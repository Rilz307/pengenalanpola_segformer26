import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import tempfile
import requests
import ee
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from scipy.ndimage import uniform_filter
from transformers import SegformerForSemanticSegmentation
import os
import datetime

# =====================================================
# PAGE CONFIG & INIT
# =====================================================
st.set_page_config(page_title="Live Flood Detection: SegFormer", page_icon="🌍", layout="wide")

st.title("🌍 Live Flood Detection (Web-GIS)")
st.markdown(
    """
    Gambarkan kotak area pada peta, atur rentang waktu, lalu **pilih tanggal spesifik** perekaman satelit. 
    Sistem akan mengunduh citra Sentinel-1 (Resolusi 10m) dan memprosesnya secara *real-time*.
    """
)

# Inisialisasi Google Earth Engine
try:
    ee.Initialize(project='project-pengenalanpola') # Ganti sesuai nama project GEE Anda
except Exception as e:
    st.warning("Meminta autentikasi Google Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project='project-pengenalanpola')

MODEL_PATH = "model/finetune_best.pth"

# =====================================================
# MODEL ARCHITECTURE (SARSegFormer)
# =====================================================
class SARSegFormer(nn.Module):
    def __init__(self, num_labels: int = 2, backbone: str = 'nvidia/mit-b2'):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone, num_labels=num_labels, ignore_mismatched_sizes=True,
            id2label={0: 'background', 1: 'water'}, label2id={'background': 0, 'water': 1},
        )
        self._adapt_patch_embedding(in_channels=2)

    def _adapt_patch_embedding(self, in_channels: int = 2):
        orig_proj = None
        patch_emb_module = None
        for name, module in self.model.named_modules():
            if "patch_embeddings" in name and hasattr(module, "proj") and isinstance(module.proj, nn.Conv2d):
                patch_emb_module = module
                orig_proj = module.proj
                break
        if orig_proj is None:
            for name, module in self.model.named_modules():
                if hasattr(module, "proj") and isinstance(module.proj, nn.Conv2d) and module.proj.in_channels == 3:
                    patch_emb_module = module
                    orig_proj = module.proj
                    break
        if orig_proj is None: raise RuntimeError("Layer patch embeddings tidak ditemukan.")
            
        old_weight = orig_proj.weight.data
        new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        new_proj = nn.Conv2d(in_channels, orig_proj.out_channels, kernel_size=orig_proj.kernel_size,
                             stride=orig_proj.stride, padding=orig_proj.padding)
        new_proj.weight = nn.Parameter(new_weight)
        if orig_proj.bias is not None: new_proj.bias = nn.Parameter(orig_proj.bias.data.clone())
        patch_emb_module.proj = new_proj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits  = outputs.logits
        H, W    = pixel_values.shape[-2:]
        logits  = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits

def sliding_window_predict(model: nn.Module, img_chw: np.ndarray, window: int = 256, overlap: int = 64, device='cpu') -> np.ndarray:
    C, H, W = img_chw.shape
    step = window - overlap
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    pad_h, pad_w = max(0, window - H), max(0, window - W)
    if pad_h > 0 or pad_w > 0:
        img_chw = np.pad(img_chw, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
        _, H_pad, W_pad = img_chw.shape
    else:
        H_pad, W_pad = H, W

    model.eval()
    with torch.no_grad():
        for y0 in range(0, H_pad - window + 1, step):
            for x0 in range(0, W_pad - window + 1, step):
                patch = img_chw[:, y0:y0+window, x0:x0+window]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(pixel_values=t)
                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                prob_map[y0:y0+window, x0:x0+window] += probs
                count_map[y0:y0+window, x0:x0+window] += 1.0

        if W_pad > window:
            for y0 in range(0, H_pad - window + 1, step):
                patch = img_chw[:, y0:y0+window, W_pad-window:W_pad]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(pixel_values=t)
                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                prob_map[y0:y0+window, W_pad-window:W_pad] += probs
                count_map[y0:y0+window, W_pad-window:W_pad] += 1.0
                
        if H_pad > window:
            for x0 in range(0, W_pad - window + 1, step):
                patch = img_chw[:, H_pad-window:H_pad, x0:x0+window]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(pixel_values=t)
                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                prob_map[H_pad-window:H_pad, x0:x0+window] += probs
                count_map[H_pad-window:H_pad, x0:x0+window] += 1.0

        if H_pad > window and W_pad > window:
            patch = img_chw[:, H_pad-window:H_pad, W_pad-window:W_pad]
            t = torch.from_numpy(patch).unsqueeze(0).to(device)
            logits = model(pixel_values=t)
            probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            prob_map[H_pad-window:H_pad, W_pad-window:W_pad] += probs
            count_map[H_pad-window:H_pad, W_pad-window:W_pad] += 1.0

    count_map = np.maximum(count_map, 1.0)
    prob_map = prob_map / count_map

    if pad_h > 0 or pad_w > 0:
        prob_map = prob_map[:H, :W]

    return (prob_map >= 0.5).astype(np.int64)

# =====================================================
# ALGORITMA PEMROSESAN (OTSU & SEGFORMER PREP)
# =====================================================
def otsu_threshold_1d(histogram, bin_edges):
    hist = histogram.astype(np.float64)
    hist /= hist.sum() + 1e-10
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    best_t, best_var = centers[0], -1.0
    for i in range(1, len(hist)):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 < 1e-8 or w1 < 1e-8: continue
        mu0 = (hist[:i] * centers[:i]).sum() / w0
        mu1 = (hist[i:] * centers[i:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var: best_var, best_t = var, centers[i]
    return best_t

def compute_otsu_label(vh_db: np.ndarray):
    vh = np.clip(vh_db.astype(np.float32), -35.0, 5.0)
    H, W = vh.shape
    cell = 100
    n_r, n_c = max(1, H // cell), max(1, W // cell)
    grid_vars, grid_patches = [], []
    for r in range(n_r):
        for c in range(n_c):
            patch = vh[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            grid_vars.append(patch.var())
            grid_patches.append(patch.flatten())
    
    if not grid_vars: return np.zeros_like(vh, dtype=np.int16), -35.0
    
    var_thr = np.percentile(grid_vars, 75)
    high_var_px = np.concatenate([grid_patches[i] for i, v in enumerate(grid_vars) if v >= var_thr])
    hist, edges = np.histogram(high_var_px, bins=256)
    threshold = otsu_threshold_1d(hist, edges)
    vh_smooth = uniform_filter(vh, size=3)
    return (vh_smooth < threshold).astype(np.int16), threshold

def speckle_filter(band: np.ndarray, ksize: int = 3) -> np.ndarray:
    linear = 10.0 ** (band / 10.0)
    mean_local = cv2.boxFilter(linear.astype(np.float32), -1, (ksize, ksize))
    mean2_local = cv2.boxFilter((linear**2).astype(np.float32), -1, (ksize, ksize))
    var_local = np.maximum(mean2_local - mean_local**2, 1e-10)
    w = var_local / (var_local + np.mean(var_local) * 0.1)
    filtered = mean_local + w * (linear - mean_local)
    return 10.0 * np.log10(np.maximum(filtered, 1e-10))

def preprocess_s1_for_model(raw_img: np.ndarray) -> np.ndarray:
    chip = raw_img.transpose(1, 2, 0)
    out = np.zeros_like(chip, dtype=np.float32)
    for c in range(chip.shape[-1]):
        b = chip[..., c].astype(np.float32)
        b = np.where(np.isfinite(b), b, -30.0)
        b = speckle_filter(b, ksize=3)
        b = np.clip(b, -30.0, 10.0)
        b_min, b_max = b.min(), b.max()
        if b_max - b_min > 1e-6: b = (b - b_min) / (b_max - b_min)
        else: b = np.zeros_like(b)
        out[..., c] = b
    return out.transpose(2, 0, 1)

# =====================================================
# LOAD MODEL & UI LAYOUT
# =====================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SARSegFormer(num_labels=2, backbone='nvidia/mit-b2')
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
    st.sidebar.success("✅ Model SegFormer Aktif")
except Exception as e:
    st.sidebar.error("Model tidak ditemukan.")
    st.stop()

# Layout Kontrol Input
st.sidebar.header("⚙️ Pencarian Citra (Waktu)")
start_date = st.sidebar.date_input("Mulai Cari Dari", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("Sampai Dengan", datetime.date(2021, 1, 31))

# Skala dikunci kaku di 10 meter/px (Konstanta Sen1Floods11)
SCALE_NATIVE = 10 
MAX_AREA_SQKM = 300 

st.sidebar.info(
    f"""
    **Constraint Model:**
    Resolusi dikunci pada **{SCALE_NATIVE} m/px**.
    Batas maksimal area penarikan: **{MAX_AREA_SQKM} km²**.
    """
)

st.subheader("1. Pilih Area of Interest (AOI)")
st.write("Gunakan alat ⬛ (*Draw a rectangle*) di peta untuk menyeleksi wilayah observasi.")

m = folium.Map(location=[-2.5489, 118.0149], zoom_start=5)
Draw(export=False, draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True}).add_to(m)
map_data = st_folium(m, width=1200, height=500)

if map_data["all_drawings"]:
    geom = map_data["all_drawings"][0]["geometry"]
    roi = ee.Geometry.Polygon(geom["coordinates"])
    
    # PROTEKSI LUAS AREA
    area_sqm = roi.area().getInfo()
    area_sqkm = area_sqm / 1_000_000

    if area_sqkm > MAX_AREA_SQKM:
        st.error(f"❌ Area terlalu luas ({area_sqkm:.2f} km²). Batas aman: {MAX_AREA_SQKM} km².")
        st.stop()
    else:
        st.success(f"✅ Luas area valid: {area_sqkm:.2f} km².")
        
        # MENCARI TANGGAL TERSEDIA DI GEE
        with st.spinner("Mencari tanggal perekaman satelit di area ini..."):
            col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                   .filterBounds(roi)
                   .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                   .filter(ee.Filter.eq('instrumentMode', 'IW'))
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                   .select(['VV', 'VH']))
            
            def get_date(img):
                return ee.Feature(None, {'date': img.date().format('YYYY-MM-DD')})
            
            try:
                date_features = col.map(get_date).distinct('date').aggregate_array('date').getInfo()
                list_tanggal = sorted(list(set(date_features)))
            except Exception as e:
                list_tanggal = []
                
        if not list_tanggal:
            st.error("❌ Satelit Sentinel-1 tidak melintasi area ini pada rentang waktu di atas. Geser area atau perlebar waktu.")
        else:
            # ---------------------------------------------------------
            # PEMILIHAN TANGGAL SPESIFIK & EKSEKUSI
            # ---------------------------------------------------------
            selected_date_str = st.selectbox("📅 Pilih Tanggal Rekaman yang Tersedia:", list_tanggal)
            
            if st.button(f"🛰️ Tarik Citra & Proses ({selected_date_str})", use_container_width=True):
                with st.spinner(f"Mengunduh citra Sentinel-1 tanggal {selected_date_str} (Resolusi {SCALE_NATIVE}m)..."):
                    try:
                        sel_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
                        next_day = sel_date + datetime.timedelta(days=1)
                        
                        target_col = col.filterDate(sel_date.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"))
                        
                        # Memakai mosaic untuk menggabungkan pass yang terpotong di hari yang sama
                        image = target_col.mosaic().clip(roi)
                        
                        url = image.getDownloadURL({
                            'scale': SCALE_NATIVE,
                            'crs': 'EPSG:4326',
                            'region': roi,
                            'format': 'GEO_TIFF'
                        })
                        
                        response = requests.get(url)
                        if response.status_code != 200:
                            st.error(f"Gagal mengunduh citra dari GEE: {response.text}")
                            st.stop()
                        
                        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                            tmp.write(response.content)
                            tmp_path = tmp.name

                        with rasterio.open(tmp_path) as src:
                            raw_img = src.read([1, 2]).astype(np.float32)
                            H, W = src.height, src.width
                        
                        os.remove(tmp_path) 
                        st.success(f"Berhasil menarik citra (Resolusi {W}x{H} piksel)")

                        # =============================================
                        # INFERENSI MODEL
                        # =============================================
                        with st.spinner("Memproses algoritma Otsu dan Deep Learning..."):
                            otsu_mask, otsu_thr = compute_otsu_label(raw_img[1])
                            
                            img_norm = preprocess_s1_for_model(raw_img)
                            dl_mask = sliding_window_predict(model, img_norm, window=256, overlap=64, device=device)

                        # =============================================
                        # VISUALISASI HASIL
                        # =============================================
                        st.subheader("2. Hasil Analisis Banjir")
                        
                        data_clean = np.nan_to_num(raw_img, nan=-30.0)
                        data_clean = np.clip(data_clean, -30.0, 10.0)
                        data_clean = (data_clean + 30.0) / 40.0
                        
                        pseudo_rgb = np.clip(np.dstack([data_clean[0], data_clean[1], data_clean[0] / (data_clean[1] + 1e-5)]), 0, 1)

                        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
                        
                        ax[0].imshow(pseudo_rgb)
                        ax[0].set_title(f"Citra Satelit - {selected_date_str}", fontweight='bold')
                        ax[0].axis("off")

                        ax[1].imshow(pseudo_rgb)
                        # Masker Otsu diwarnai Orange
                        ax[1].imshow(otsu_mask, cmap=mcolors.ListedColormap(["#00000000", "orange"]), alpha=0.6)
                        ax[1].set_title(f"Baseline (Otsu Threshold)\nThr: {otsu_thr:.2f} dB", fontweight='bold')
                        ax[1].axis("off")

                        ax[2].imshow(pseudo_rgb)
                        # Masker SegFormer diwarnai Kuning Stabilo (Sangat Kontras)
                        ax[2].imshow(dl_mask, cmap=mcolors.ListedColormap(["#00000000", "#FFFF00"]), alpha=0.6)
                        ax[2].set_title("Deep Learning (SegFormer)", fontweight='bold')
                        ax[2].axis("off")

                        st.pyplot(fig)

                        # =============================================
                        # STATISTIK KOMPARATIF
                        # =============================================
                        st.subheader("📊 Statistik Perbandingan Area")
                        total_pixels = dl_mask.size
                        otsu_flood_px = int((otsu_mask == 1).sum())
                        dl_flood_px = int((dl_mask == 1).sum())
                        
                        otsu_ha = (otsu_flood_px * (SCALE_NATIVE ** 2)) / 10000
                        dl_ha = (dl_flood_px * (SCALE_NATIVE ** 2)) / 10000

                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"**Otsu Thresholding (Orange)**\n- Total Luasan: {otsu_ha:,.2f} Hektar\n- Persentase: {(otsu_flood_px / total_pixels) * 100:.2f}%")
                        with c2:
                            st.success(f"**SegFormer (Kuning)**\n- Total Luasan: {dl_ha:,.2f} Hektar\n- Persentase: {(dl_flood_px / total_pixels) * 100:.2f}%")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")
else:
    st.caption("Silakan gambar area (kotak hitam di sebelah kiri peta) untuk mulai menganalisis.")