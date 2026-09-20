import hashlib
import html
import io
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------------
# Konfigurasi (harus sama dengan saat training di notebook)
# ----------------------------------------------------------------------------
IMG_SIZE = 128
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # urutan WAJIB sama
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_final.keras")

HIGH_CONFIDENCE = 0.85   # di atas ini: "tinggi"
LOW_CONFIDENCE = 0.60    # di bawah ini: "rendah"
PANEL = 384              # ukuran sisi gambar yang ditampilkan
DEFAULT_ALPHA = 45       # intensitas overlay Grad-CAM (%)
WIB = timezone(timedelta(hours=7))

# Warna aksen per kelas: terang untuk latar gelap, gelap untuk latar terang
ACCENT_LIGHT = {"glioma": "#E39A45", "meningioma": "#6AA8E0", "pituitary": "#A98BDC", "notumor": "#9DB0BF"}
ACCENT_DARK = {"glioma": "#B8631C", "meningioma": "#2F73B8", "pituitary": "#7F5FC0", "notumor": "#6B8294"}

INFO = {
    "glioma": {
        "label": "Glioma",
        "ringkas": "Tumor yang tumbuh dari sel glia, sel penyokong jaringan saraf di otak.",
        "sections": [
            ("Tentang tumor ini",
             "Glioma tumbuh dari sel glia, yaitu sel penyokong jaringan saraf di otak dan sumsum tulang belakang. "
             "Tingkat keganasannya bervariasi, dari derajat rendah yang tumbuh lambat hingga derajat tinggi yang "
             "tumbuh agresif, seperti glioblastoma."),
            ("Yang biasa tampak pada MRI",
             "Umumnya berupa massa di dalam jaringan otak (intra-aksial) dengan batas yang sering tidak tegas, dan "
             "bisa disertai pembengkakan (edema) di sekitarnya. Pada derajat tinggi, penyerapan kontras sering tidak merata."),
            ("Gejala yang umum",
             "Sakit kepala yang menetap atau makin berat, kejang, kelemahan pada satu sisi tubuh, gangguan bicara atau "
             "penglihatan, serta perubahan perilaku atau daya ingat. Gejalanya bergantung pada lokasi tumor."),
            ("Penanganan umum",
             "Ditentukan oleh derajat dan lokasi tumor, dan biasanya melibatkan dokter bedah saraf dan onkologi. "
             "Pilihannya dapat berupa operasi, radioterapi, dan kemoterapi."),
        ],
    },
    "meningioma": {
        "label": "Meningioma",
        "ringkas": "Tumor yang tumbuh dari selaput pelindung otak (meningen), umumnya jinak.",
        "sections": [
            ("Tentang tumor ini",
             "Meningioma tumbuh dari selaput pelindung otak dan sumsum tulang belakang (meningen). Sebagian besar "
             "bersifat jinak dan tumbuh lambat, tetapi tetap bisa menimbulkan gejala karena menekan jaringan otak di "
             "dekatnya. Lebih sering ditemukan pada perempuan dan pada usia dewasa lanjut."),
            ("Yang biasa tampak pada MRI",
             "Umumnya tampak sebagai massa di luar jaringan otak (ekstra-aksial) yang menempel pada selaput otak, "
             "dengan batas tegas dan penyerapan kontras yang merata dan kuat."),
            ("Gejala yang umum",
             "Sering tanpa gejala dan ditemukan secara tidak sengaja. Bila bergejala: sakit kepala, kejang, gangguan "
             "penglihatan, atau kelemahan yang berkembang perlahan."),
            ("Penanganan umum",
             "Meningioma kecil tanpa gejala sering hanya dipantau dengan MRI berkala. Bila membesar atau bergejala, "
             "pilihannya operasi atau radioterapi, dengan keputusan dari dokter bedah saraf."),
        ],
    },
    "pituitary": {
        "label": "Tumor pituitari",
        "ringkas": "Tumor pada kelenjar pituitari di dasar otak yang mengatur banyak hormon.",
        "sections": [
            ("Tentang tumor ini",
             "Tumor pituitari (umumnya adenoma hipofisis) tumbuh di kelenjar pituitari, kelenjar kecil di dasar otak "
             "yang mengatur banyak hormon tubuh. Sebagian besar bersifat jinak. Ada yang menghasilkan hormon berlebih "
             "dan ada yang tidak."),
            ("Yang biasa tampak pada MRI",
             "Tampak sebagai massa di area sella turcica, yaitu rongga tulang di dasar tengkorak tempat kelenjar "
             "pituitari berada, dan dapat meluas ke atas mendekati saraf penglihatan."),
            ("Gejala yang umum",
             "Gangguan penglihatan (terutama lapang pandang bagian samping), sakit kepala, serta gejala hormonal seperti "
             "haid tidak teratur, keluarnya ASI di luar masa menyusui, atau perubahan fisik akibat kelebihan hormon."),
            ("Penanganan umum",
             "Bergantung pada jenis dan ukuran tumor. Sebagian dapat diobati dengan obat, sebagian memerlukan operasi "
             "lewat hidung (transsfenoidal) atau radioterapi. Biasanya ditangani bersama dokter endokrin dan bedah saraf."),
        ],
    },
    "notumor": {
        "label": "Tidak terdeteksi tumor",
        "ringkas": "Model tidak menemukan pola glioma, meningioma, atau tumor pituitari.",
        "sections": [
            ("Hasil ini berarti",
             "Model tidak menemukan pola yang cocok dengan glioma, meningioma, maupun tumor pituitari pada citra ini."),
            ("Yang perlu diingat",
             "Hasil ini tidak menyingkirkan kelainan lain, dan model bisa saja keliru. Penilaian citra MRI tetap harus "
             "dilakukan oleh dokter spesialis radiologi."),
            ("Bila ada keluhan",
             "Sakit kepala yang menetap, kejang, gangguan penglihatan, atau kelemahan anggota gerak tetap perlu "
             "diperiksakan ke dokter meskipun hasil model tidak menunjukkan tumor."),
        ],
    },
}
DISPLAY_ORDER = ["glioma", "meningioma", "pituitary", "notumor"]

st.set_page_config(
    page_title="Klasifikasi Tumor Otak MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Spectral:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Public Sans', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.title { font-family: 'Spectral', Georgia, serif; font-size: 2.3rem; font-weight: 700;
         line-height: 1.15; color: #16212B; margin: 0 0 .35rem 0; }
.subtitle { color: #4B5A68; font-size: 1.02rem; margin-bottom: 1.4rem; line-height: 1.55; }
.section-h { font-family: 'Spectral', Georgia, serif; font-size: 1.25rem; font-weight: 600; margin: 0 0 .7rem 0; color: #16212B; }

.meta { display: flex; flex-wrap: wrap; gap: .4rem 2.2rem; padding: .7rem 0 .9rem 0; margin-bottom: .9rem;
        border-bottom: 1px solid #D5DCE2; }
.meta div { display: flex; flex-direction: column; }
.meta span { color: #6B7A88; font-size: .78rem; }
.meta b { color: #16212B; font-size: .93rem; font-weight: 500; word-break: break-all; }

.verdict { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; justify-content: space-between;
           background: #16212B; color: #F3F5F7; border-radius: 6px; border-left: 6px solid #6AA8E0;
           padding: 1.5rem 1.7rem; margin: 0 0 1.6rem 0; }
.v-main { flex: 1 1 320px; min-width: 0; }
.v-kicker { color: #9FB0BF; font-size: .85rem; margin-bottom: .25rem; }
.v-name { font-family: 'Spectral', Georgia, serif; font-size: 2.3rem; font-weight: 700; line-height: 1.1; }
.v-sub { color: #C7D2DB; font-size: .95rem; margin-top: .55rem; line-height: 1.5; max-width: 60ch; }
.v-meta { display: flex; flex-wrap: wrap; gap: .6rem 1rem; align-items: center; margin-top: 1rem; font-size: .9rem; color: #C7D2DB; }
.chip { display: inline-flex; align-items: center; gap: .5rem; background: rgba(255,255,255,.09);
        border-radius: 999px; padding: .28rem .8rem; font-size: .85rem; color: #F3F5F7; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.ring { width: 140px; height: 140px; border-radius: 50%; display: grid; place-items: center; flex: 0 0 auto; }
.ring-in { width: 110px; height: 110px; border-radius: 50%; background: #16212B; display: flex;
           flex-direction: column; align-items: center; justify-content: center; }
.ring-in .rv { font-family: 'Spectral', Georgia, serif; font-size: 1.65rem; font-weight: 700; line-height: 1; }
.ring-in .rl { color: #9FB0BF; font-size: .75rem; margin-top: .25rem; }

.bar-row { margin-bottom: .95rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .93rem; margin-bottom: .3rem; }
.bar-head .n { color: #16212B; font-weight: 500; }
.bar-head .v { color: #4B5A68; font-variant-numeric: tabular-nums; }
.bar-track { background: #DCE3E9; height: 10px; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; background: #A9B6C2; }

.caption { color: #4B5A68; font-size: .85rem; text-align: center; margin-top: .3rem; }
.legend { margin: .55rem auto 0 auto; max-width: 92%; }
.lg-bar { height: 8px; border-radius: 2px;
          background: linear-gradient(90deg, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000); }
.lg-lb { display: flex; justify-content: space-between; color: #6B7A88; font-size: .75rem; margin-top: .2rem; }

.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; margin-top: .2rem; }
.info-card { background: #FFFFFF; border: 1px solid #D5DCE2; border-top: 3px solid #0E7C86; border-radius: 4px; padding: 1rem 1.15rem 1.1rem 1.15rem; }
.info-card h4 { font-family: 'Public Sans', sans-serif; font-size: .98rem; font-weight: 600; color: #16212B; margin: 0 0 .4rem 0; }
.info-card p { color: #2B3946; font-size: .93rem; line-height: 1.6; margin: 0; }

.class-card { background: #FFFFFF; border: 1px solid #D5DCE2; border-top: 3px solid #0E7C86; border-radius: 4px; padding: 1rem 1.15rem; height: 100%; }
.class-card .cn { font-family: 'Spectral', Georgia, serif; font-size: 1.15rem; font-weight: 600; color: #16212B; margin-bottom: .3rem; }
.class-card .cd { color: #4B5A68; font-size: .9rem; line-height: 1.5; }

.disclaimer { color: #4B5A68; font-size: .83rem; border-top: 1px solid #D5DCE2; padding-top: .9rem; margin-top: 2rem; }
[data-testid="stImage"] img { border-radius: 4px; background: #0B1218; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Preprocessing (identik dengan preprocess_image() di notebook)
# ----------------------------------------------------------------------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def crop_brain_roi(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    thr = cv2.erode(thr, None, iterations=2)
    thr = cv2.dilate(thr, None, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        return image
    return image[y : y + h, x : x + w]


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Input: RGB uint8. Output: float32 [0,1] berukuran 128x128x3."""
    img = cv2.GaussianBlur(rgb, (3, 3), 0)
    img = apply_clahe(img)
    img = crop_brain_roi(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


# ----------------------------------------------------------------------------
# Model, Grad-CAM, dan pembantu gambar
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def find_last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def gradcam_heatmap(model, x, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        inp = tf.convert_to_tensor(x[np.newaxis], dtype=tf.float32)
        conv_out, preds = grad_model(inp, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def letterbox(img: np.ndarray, size: int = PANEL) -> np.ndarray:
    """Taruh gambar di kanvas persegi hitam tanpa mengubah rasio, supaya semua panel sama besar."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def overlay_heatmap(x_float: np.ndarray, heat: np.ndarray, alpha: float, size: int = PANEL) -> np.ndarray:
    base = cv2.resize((x_float * 255).astype("uint8"), (size, size), interpolation=cv2.INTER_CUBIC)
    heat_r = cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heat_r, 0, 1)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(base * (1 - alpha) + colored * alpha)


def to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def analyze(file_bytes: bytes):
    """Semua perhitungan berat dilakukan sekali per gambar, lalu di-cache."""
    try:
        rgb = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    model = load_model()
    x = preprocess(rgb)
    probs = model.predict(x[np.newaxis], verbose=0)[0]
    top = int(np.argmax(probs))
    try:
        heat = gradcam_heatmap(model, x, find_last_conv_name(model), top)
    except Exception:
        heat = None
    return {
        "size": (int(rgb.shape[1]), int(rgb.shape[0])),
        "probs": probs.astype("float64"),
        "x": x,
        "heat": heat,
        "orig": letterbox(rgb),
        "proc": letterbox((x * 255).astype("uint8")),
    }


# ----------------------------------------------------------------------------
# Laporan PDF
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_pdf(name, stamp, size_txt, probs, top, orig_png, proc_png, cam_png) -> bytes:
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ink, muted, teal = colors.HexColor("#16212B"), colors.HexColor("#4B5A68"), colors.HexColor("#0E7C86")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                        textColor=ink, alignment=0, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                        textColor=ink, spaceBefore=9, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13,
                          textColor=colors.HexColor("#2B3946"))
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11.5, textColor=muted)
    big = ParagraphStyle("big", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=26, textColor=teal)
    cap = ParagraphStyle("cap", parent=small, alignment=1)

    cls = CLASSES[top]
    info = INFO[cls]
    conf = probs[top]
    level_txt = confidence_level(conf)[0]

    els = [Paragraph("Laporan Klasifikasi Tumor Otak dari Citra MRI", h1)]

    meta = Table(
        [["Berkas", escape(name)], ["Dianalisis", stamp], ["Ukuran citra asli", size_txt]],
        colWidths=[3.6 * cm, 13.4 * cm],
    )
    meta.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"), ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (0, -1), muted),
        ("TEXTCOLOR", (1, 0), (1, -1), ink), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#D5DCE2")),
    ]))
    els += [meta, Spacer(1, 4)]

    els.append(Paragraph("Hasil prediksi", h2))
    els.append(Paragraph(escape(info["label"]), big))
    els.append(Paragraph(f"Keyakinan model {pct(conf)} ({level_txt.lower()}). {escape(info['ringkas'])}", body))
    els.append(Spacer(1, 8))

    order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])
    rows = [["Kelas", "Probabilitas"]] + [[INFO[CLASSES[i]]["label"], pct(probs[i])] for i in order]
    pt = Table(rows, colWidths=[9 * cm, 4 * cm], hAlign="LEFT")
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EBEF")), ("TEXTCOLOR", (0, 1), (-1, 1), teal),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DCE2")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    els += [pt, Spacer(1, 4)]

    panels = [(orig_png, "Citra asli"), (proc_png, "Setelah diproses")]
    if cam_png is not None:
        panels.append((cam_png, "Area perhatian model (Grad-CAM)"))
    side = 4.5 * cm
    img_tbl = Table(
        [[RLImage(io.BytesIO(p), width=side, height=side) for p, _ in panels],
         [Paragraph(c, cap) for _, c in panels]],
        colWidths=[side + 0.35 * cm] * len(panels), hAlign="LEFT",
    )
    img_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    els.append(Paragraph("Citra", h2))
    els += [img_tbl]

    els.append(Paragraph(f"Informasi: {escape(info['label'])}", h2))
    for head, text in info["sections"]:
        els.append(KeepTogether([Paragraph(f"<b>{escape(head)}</b>", body),
                                 Paragraph(escape(text), body), Spacer(1, 4)]))

    els.append(Spacer(1, 4))
    els.append(Paragraph(
        "Hasil ini adalah prediksi model untuk keperluan pembelajaran, bukan diagnosis medis. "
        "Konsultasikan citra MRI dengan dokter atau radiolog.", small))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm, title="Laporan Klasifikasi Tumor Otak").build(els)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Pembantu tampilan
# ----------------------------------------------------------------------------
def model_file_problem():
    """Periksa file model dan kembalikan pesan masalahnya (atau None kalau tampak valid).

    Keras melaporkan "File not found" untuk file .keras yang bukan zip, padahal file-nya ada.
    Pengecekan ini membedakan penyebab yang sebenarnya.
    """
    if not os.path.exists(MODEL_PATH):
        return "File `classifier_final.keras` tidak ditemukan di folder yang sama dengan `app.py`."
    if os.path.isdir(MODEL_PATH):
        return ("`classifier_final.keras` berupa folder, padahal harus satu file. "
                "Jangan extract file .keras, upload file aslinya.")
    size = os.path.getsize(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        return (f"`classifier_final.keras` hanya berisi pointer Git LFS ({size} byte), bukan model aslinya. "
                "Push ulang model tanpa Git LFS.")
    if head.startswith(b"\x89HDF"):
        return ("`classifier_final.keras` sebenarnya berformat HDF5 (.h5) yang hanya diganti namanya. "
                "Ambil file `classifier_final.keras` yang asli dari output notebook.")
    if not zipfile.is_zipfile(MODEL_PATH):
        return (f"`classifier_final.keras` ({size:,} byte) bukan arsip .keras yang valid. "
                "File kemungkinan rusak atau terpotong saat upload.")
    return None


def pct(p: float) -> str:
    return f"{p * 100:.1f}".replace(".", ",") + "%"


def confidence_level(conf: float):
    if conf >= HIGH_CONFIDENCE:
        return "Keyakinan tinggi", "#4CC38A"
    if conf >= LOW_CONFIDENCE:
        return "Keyakinan sedang", "#E3B23C"
    return "Keyakinan rendah, hasil kurang dapat diandalkan", "#E5604D"


def safe_stem(name: str) -> str:
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "citra"


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
st.markdown('<div class="title">Klasifikasi Tumor Otak dari Citra MRI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Unggah citra MRI otak untuk mendapatkan prediksi jenis tumornya.</div>',
    unsafe_allow_html=True,
)

problem = model_file_problem()
if problem:
    st.error(problem)
    st.stop()

uploaded = st.file_uploader("Pilih citra MRI (JPG, JPEG, atau PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.markdown('<div class="section-h" style="margin-top:1.4rem">Jenis yang dikenali</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, key in zip(cols, DISPLAY_ORDER):
        with col:
            st.markdown(
                f'<div class="class-card" style="border-top-color:{ACCENT_DARK[key]}">'
                f'<div class="cn">{INFO[key]["label"]}</div>'
                f'<div class="cd">{INFO[key]["ringkas"]}</div></div>',
                unsafe_allow_html=True,
            )
else:
    data = uploaded.getvalue()
    with st.spinner("Menganalisis citra..."):
        res = analyze(data)
    if res is None:
        st.error("Gambar tidak bisa dibaca. Coba file lain.")
        st.stop()

    probs = res["probs"]
    order = [int(i) for i in np.argsort(-probs)]
    top, second = order[0], order[1]
    top_cls = CLASSES[top]
    info = INFO[top_cls]
    conf = float(probs[top])
    level_txt, level_color = confidence_level(conf)

    file_key = hashlib.md5(data).hexdigest()
    stamp = st.session_state.setdefault(f"ts_{file_key}", datetime.now(WIB).strftime("%d/%m/%Y %H.%M WIB"))
    w_px, h_px = res["size"]
    size_txt = f"{w_px} x {h_px} piksel"

    # Gambar overlay bawaan untuk PDF
    cam_default = overlay_heatmap(res["x"], res["heat"], DEFAULT_ALPHA / 100) if res["heat"] is not None else None
    pdf_bytes = build_pdf(
        uploaded.name, stamp, size_txt, tuple(float(p) for p in probs), top,
        to_png(res["orig"]), to_png(res["proc"]),
        to_png(cam_default) if cam_default is not None else None,
    )

    head_l, head_r = st.columns([3, 1], vertical_alignment="center")
    with head_l:
        st.markdown('<div class="section-h" style="margin:1.2rem 0 0 0">Laporan hasil klasifikasi</div>', unsafe_allow_html=True)
    with head_r:
        st.download_button(
            "Unduh laporan PDF", data=pdf_bytes, file_name=f"laporan_{safe_stem(uploaded.name)}.pdf",
            mime="application/pdf",
        )

    st.markdown(
        '<div class="meta">'
        f'<div><span>Berkas</span><b>{html.escape(uploaded.name)}</b></div>'
        f'<div><span>Dianalisis</span><b>{stamp}</b></div>'
        f'<div><span>Ukuran citra asli</span><b>{size_txt}</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    deg = conf * 360
    accent = ACCENT_LIGHT[top_cls]
    st.markdown(
        f'<div class="verdict" style="border-left-color:{accent}">'
        f'<div class="v-main"><div class="v-kicker">Prediksi model</div>'
        f'<div class="v-name">{info["label"]}</div>'
        f'<div class="v-sub">{info["ringkas"]}</div>'
        f'<div class="v-meta"><span class="chip"><span class="dot" style="background:{level_color}"></span>{level_txt}</span>'
        f'<span>Kemungkinan kedua: {INFO[CLASSES[second]]["label"]} ({pct(float(probs[second]))})</span></div></div>'
        f'<div class="ring" style="background:conic-gradient({accent} {deg:.1f}deg, rgba(255,255,255,.12) 0)">'
        f'<div class="ring-in"><span class="rv">{pct(conf)}</span><span class="rl">keyakinan</span></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-h">Citra</div>', unsafe_allow_html=True)
        alpha = DEFAULT_ALPHA
        if res["heat"] is not None:
            alpha = st.slider("Intensitas overlay Grad-CAM", 0, 100, DEFAULT_ALPHA, format="%d%%")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(res["orig"], width="stretch")
            st.markdown('<div class="caption">Citra asli</div>', unsafe_allow_html=True)
        with c2:
            st.image(res["proc"], width="stretch")
            st.markdown('<div class="caption">Setelah diproses</div>', unsafe_allow_html=True)
        with c3:
            if res["heat"] is not None:
                st.image(overlay_heatmap(res["x"], res["heat"], alpha / 100), width="stretch")
                st.markdown('<div class="caption">Area perhatian model</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="legend"><div class="lg-bar"></div>'
                    '<div class="lg-lb"><span>pengaruh rendah</span><span>tinggi</span></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Area perhatian tidak tersedia.")

    with right:
        st.markdown('<div class="section-h">Probabilitas per kelas</div>', unsafe_allow_html=True)
        rows = ""
        for i in order:
            p = float(probs[i]) * 100
            fill = f"background:{ACCENT_DARK[CLASSES[i]]}" if i == top else ""
            rows += (
                f'<div class="bar-row"><div class="bar-head"><span class="n">{INFO[CLASSES[i]]["label"]}</span>'
                f'<span class="v">{pct(float(probs[i]))}</span></div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{p:.1f}%;{fill}"></div></div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    st.markdown(f'<div class="section-h" style="margin-top:1.6rem">Informasi: {info["label"]}</div>', unsafe_allow_html=True)
    cards = "".join(
        f'<div class="info-card" style="border-top-color:{ACCENT_DARK[top_cls]}"><h4>{h}</h4><p>{t}</p></div>'
        for h, t in info["sections"]
    )
    st.markdown(f'<div class="info-grid">{cards}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Hasil ini adalah prediksi model untuk keperluan pembelajaran, bukan diagnosis medis. '
    "Konsultasikan citra MRI dengan dokter atau radiolog.</div>",
    unsafe_allow_html=True,
)
