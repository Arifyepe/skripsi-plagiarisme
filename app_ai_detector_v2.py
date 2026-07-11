import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import PyPDF2
import re
import os
from scipy.sparse import hstack
import joblib
import string

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="AI DETECT ACADEMIC INTEGRITY SYSTEM", page_icon="🎓", layout="wide")

# =====================================================================
# KODE CSS (BACKGROUND PUTIH & DROPZONE BIRU MUDA)
# =====================================================================
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], main { background-color: #F8F9FA !important; }
    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, label, div { color: #111111 !important; }
    
    [data-testid="stTextInput"] > div > div { background-color: #FFFFFF !important; border: 1.5px solid #CED4DA !important; border-radius: 8px !important; align-items: center !important; }
    [data-testid="stTextInput"] > div > div:focus-within { border-color: #0d6efd !important; box-shadow: 0 0 0 0.2rem rgba(13,110,253,.25) !important; }
    [data-testid="stTextInput"] input { color: #000000 !important; caret-color: #000000 !important; background-color: #FFFFFF !important; -webkit-text-fill-color: #000000 !important; }
    [data-testid="stTextInput"] button { color: #000000 !important; background-color: #FFFFFF !important; transform: translateY(-2px) !important; padding-bottom: 2px !important; }
    [data-testid="stTextInput"] svg { fill: #000000 !important; color: #000000 !important; }

    div[data-testid="stForm"] { background-color: #FFFFFF !important; border: 1px solid #E5E7EB !important; border-radius: 16px !important; padding: 30px 30px !important; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08) !important; }
    div[data-testid="stForm"] button { background-color: #0d6efd !important; border-color: #0d6efd !important; border-radius: 8px !important; padding: 10px !important; margin-top: 15px !important; }
    div[data-testid="stForm"] button p { color: #FFFFFF !important; font-weight: bold !important; }

    /* PERBAIKAN AREA UPLOAD */
    [data-testid="stFileUploader"] { background-color: transparent !important; }
    
    [data-testid="stFileUploadDropzone"], 
    [data-testid="stFileUploaderDropzone"] { 
        background-color: #F8F9FA !important; 
        border: 2px dashed #CBD5E1 !important; 
        border-radius: 12px !important; 
        padding: 40px 20px !important; 
    }
    
    [data-testid="stFileUploadDropzone"] > div, [data-testid="stFileUploadDropzone"] > section,
    [data-testid="stFileUploaderDropzone"] > div, [data-testid="stFileUploaderDropzone"] > section { 
        display: flex !important; flex-direction: column !important; align-items: center !important; 
        justify-content: center !important; text-align: center !important; background-color: transparent !important; 
    }
    
    [data-testid="stFileUploadDropzone"] svg, [data-testid="stFileUploaderDropzone"] svg { display: none !important; }
    [data-testid="stFileUploadDropzone"] p, [data-testid="stFileUploaderDropzone"] p { color: #111111 !important; font-size: 18px !important; font-weight: 500 !important; margin-bottom: 5px !important; }
    [data-testid="stFileUploadDropzone"] small, [data-testid="stFileUploaderDropzone"] small { color: #8C98A4 !important; font-size: 14px !important; }
    
    [data-testid="stFileUploadDropzone"] button, [data-testid="stFileUploaderDropzone"] button { 
        background-color: #38BDF8 !important; border: none !important; border-radius: 8px !important; 
        padding: 10px 30px !important; margin-top: 15px !important; box-shadow: 0 4px 6px rgba(56, 189, 248, 0.25) !important; transition: all 0.3s ease !important; 
    }
    [data-testid="stFileUploadDropzone"] button:hover, [data-testid="stFileUploaderDropzone"] button:hover { 
        background-color: #0EA5E9 !important; transform: translateY(-2px) !important; border: none !important; 
    }
    [data-testid="stFileUploadDropzone"] button *, [data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; font-weight: 600 !important; font-size: 15px !important; }
    
    [data-testid="stUploadedFile"], [data-testid="stUploadedFile"] > div { background-color: #FFFFFF !important; }
    [data-testid="stUploadedFile"] { border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }
    [data-testid="stUploadedFile"] * { color: #111111 !important; }
    [data-testid="stUploadedFile"] svg { display: inline-block !important; fill: #111111 !important; }

    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E7EB !important; }
    [data-testid="stSidebar"] button { background-color: #dc3545 !important; border-color: #dc3545 !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] button p { color: #FFFFFF !important; font-weight: bold !important; }
    [data-testid="stSidebar"] button:hover { background-color: #c82333 !important; }

    [data-testid="stMainBlockContainer"] button[kind="secondary"] { background-color: #FFFFFF !important; border: 1px solid #CED4DA !important; border-radius: 8px !important; }
    [data-testid="stMainBlockContainer"] button[kind="secondary"] p { color: #000000 !important; font-weight: bold !important; }
    [data-testid="stMainBlockContainer"] button[kind="secondary"]:hover { background-color: #F8F9FA !important; border-color: #0d6efd !important; color: #0d6efd !important; }
    [data-testid="stMainBlockContainer"] button[kind="primary"] { background-color: #0d6efd !important; border-color: #0d6efd !important; border-radius: 8px !important; }
    [data-testid="stMainBlockContainer"] button[kind="primary"] p { color: #FFFFFF !important; font-weight: bold !important; }
    
    /* CSS Kustom Tambahan untuk Hasil Prediksi */
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        background-color: white;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Box Khusus Heuristik */
    .heuristik-box {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .heuristik-title {
        font-size: 16px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .heuristik-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #f1f3f5;
    }
    
    .heuristik-model {
        font-weight: 600;
        color: #343a40;
    }
    
    .heuristik-pct {
        font-weight: bold;
        color: #0d6efd;
        background-color: #e9ecef;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
for key in ['scan_done', 'model_trained']:
    if key not in st.session_state:
        st.session_state[key] = False
for key in ['role', 'username', 'active_filter']:
    if key not in st.session_state:
        st.session_state[key] = "Pengguna" if key == 'username' else "Semua" if key == 'active_filter' else "User" 
if 'hasil_analisis' not in st.session_state:
    st.session_state['hasil_analisis'] = []
if 'ringkasan' not in st.session_state:
    st.session_state['ringkasan'] = {}
if 'model_pack' not in st.session_state:
    st.session_state['model_pack'] = None

# =====================================================================
# FUNGSI EKSTRAKSI FITUR HYBRID (Stylometry + TF-IDF)
# =====================================================================
def hitung_stylometry(teks):
    kata = re.findall(r'\b\w+\b', teks.lower())
    jumlah_kata = len(kata) if len(kata) > 0 else 1
    kata_unik = len(set(kata))
    lexical_diversity = kata_unik / jumlah_kata
    
    kalimat = re.split(r'[.!?]+', teks)
    kalimat = [k.strip() for k in kalimat if len(k.strip()) > 0]
    jumlah_kalimat = len(kalimat) if len(kalimat) > 0 else 1
    avg_sentence_length = jumlah_kata / jumlah_kalimat
    
    tanda_baca = len(re.findall(r'[.,!?;:]', teks))
    punct_ratio = tanda_baca / jumlah_kata
    
    return [lexical_diversity, avg_sentence_length, punct_ratio]

def bersihkan_teks(teks):
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    import re
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def extract_stylometric_features(text_series):
    """Mengekstrak fitur stylometri sesuai standar train_model.py dan naskah skripsi"""
    features = []
    for text in text_series:
        if not isinstance(text, str):
            text = ""
        words = text.split()
        word_count = len(words) if len(words) > 0 else 1
        char_count = len(text)
        avg_word_len = char_count / word_count
        punct_count = sum([1 for char in text if char in string.punctuation])
        features.append([word_count, char_count, avg_word_len, punct_count])
    return np.array(features)

def tebak_sumber_ai(teks_asli, lex_div, avg_len):
    """Fungsi heuristik untuk menebak model AI berdasarkan gaya bahasa dan pola frasa"""
    teks = teks_asli.lower()
    
    score_gpt = 0
    score_gemini = 0
    score_claude = 0
    
    # 1. Analisis Metrik Stilometri (Hanya dipicu jika ekstrem)
    if lex_div >= 0.85:
        score_claude += 1
    elif lex_div <= 0.40:
        score_gpt += 1
        
    # 2. Analisis Panjang Kalimat (Hanya dipicu jika ekstrem)
    if avg_len >= 22:
        score_gemini += 1
    elif avg_len <= 10:
        score_gpt += 1
        
    # 3. Analisis Pola Frasa (Sangat kaku dan spesifik)
    gpt_phrases = ["penting untuk dicatat", "perlu diingat bahwa", "sebagai model bahasa ai"]
    gemini_phrases = ["secara komprehensif", "mari kita telusuri"]
    claude_phrases = ["patut digarisbawahi", "secara fundamental"]
    
    for phrase in gpt_phrases:
        if phrase in teks: score_gpt += 2
    for phrase in gemini_phrases:
        if phrase in teks: score_gemini += 2
    for phrase in claude_phrases:
        if phrase in teks: score_claude += 2
        
    # Penentuan Pemenang secara adil tanpa bias >=
    if score_claude > score_gpt and score_claude > score_gemini:
        return "Claude"
    elif score_gemini > score_gpt and score_gemini > score_claude:
        return "Gemini"
    elif score_gpt > score_claude and score_gpt > score_gemini:
        return "ChatGPT"
    else:
        return "Campuran (Seri)" # Jika seri, dibagi rata 33%

# =====================================================================
# PENGECUALIAN (WHITELIST) DAFTAR STRUKTURAL SKRIPSI (DIROMBAK TOTAL & SUPER AGRESIF)
# =====================================================================
def is_halaman_struktural(teks):
    t_lower = teks.lower().strip()
    
    # 1. CEK POLA TITIK-TITIK (Khas Daftar Isi, Gambar, Tabel, Lampiran)
    if re.search(r'\.{5,}\s*\d+', t_lower):
        return True
        
    # 2. CEK KATA KUNCI MUTLAK (Sangat Agresif - Jika muncul di mana saja, lewati AI)
    frasa_mutlak = [
        "kata kunci:", "keywords:", "kata kunci :", "keywords :",
        "pernyataan originalitas", "lembar pengesahan", "halaman pengesahan",
        "daftar pustaka", "daftar lampiran", "daftar gambar", "daftar tabel",
        "tulisan ilmiah", "tugas akhir", "skripsi", "tesis", "disertasi", "abstrak", "abstract"
    ]
    if any(frasa in t_lower for frasa in frasa_mutlak):
        return True
        
    # 3. CEK KATA KUNCI COVER & LEMBAR PENGESAHAN & FORM ISIAN (Kombinasi Form Isian)
    cover_keywords = [
        "nama", "npm", "nim", "jurusan", "program studi", "pembimbing", 
        "penguji", "tanggal sidang", "tanggal lulus", "ketua", "kasubag",
        "fakultas", "universitas", "menyetujui", "tanda tangan", "nip", "nidn", "judul",
        "mata pelajaran", "kelas", "absen", "hari", "tanggal", "latihan", "fase", "semester"
    ]
    # Jika dalam 1 paragraf pendek (<80 kata) memuat setidaknya 3 kata kunci dari form cover/pengesahan
    if len(t_lower.split()) < 80:
        match_count = sum(1 for k in cover_keywords if k in t_lower)
        if match_count >= 3:
            return True
        # Atau jika banyak tanda titik dua (khas formulir/biodata)
        if t_lower.count(':') >= 2 and match_count >= 2:
            return True

    # 4. DETEKSI ISI DAFTAR PUSTAKA (SITASI) YANG SANGAT AGRESIF
    has_year = bool(re.search(r'\b(19|20)\d{2}\b', t_lower)) # Deteksi adanya tahun (ex: 2024, 2025)
    has_bracket = bool(re.search(r'\[\d+\]', t_lower)) # Deteksi format IEEE (ex: [1])
    
    if has_year or has_bracket:
        # Cek jika ada elemen khas pustaka akademik
        pustaka_keywords = [
            'jurnal', 'journal', 'vol.', 'vol ', 'no.', 'hal.', 'pp.', 'doi', 'https://', 'http', 
            'diakses', 'press', 'universitas', 'university', 'edisi', 'makalah', 'proceeding'
        ]
        if sum(1 for k in pustaka_keywords if k in t_lower) >= 1:
            return True
            
        # Jika tidak ada kata kuncinya, tapi gaya formatnya padat tanda baca (Khas Daftar Pustaka Penulis, A. B. (2020).)
        if t_lower.count('.') >= 3 and t_lower.count(',') >= 2 and len(t_lower.split()) < 100:
            return True

    # 5. DETEKSI DAFTAR LAMPIRAN (Tanpa Titik-titik, contoh: "Lampiran 1 Wawancara")
    if len(re.findall(r'lampiran\s+\d+', t_lower)) >= 1 or len(re.findall(r'lampiran\s+[a-z]', t_lower)) >= 1:
        return True
        
    # 6. DETEKSI ISI LAMPIRAN LISTING PROGRAM (Source Code)
    kode_keywords = ['if ', 'else:', 'for ', 'while ', 'def ', 'class ', 'import ', 'include ', 'void ', 'public ', 'echo ', 'return ', 'function ']
    simbol_koding = t_lower.count('{') + t_lower.count('}') + t_lower.count(';') + t_lower.count('()')
    if sum(1 for k in kode_keywords if k in t_lower) >= 2 or simbol_koding >= 3:
        return True

    # 7. DETEKSI HURUF KAPITAL SEMUA (JUDUL / NAMA KAMPUS / KOP SOAL)
    # Jika teks pendek (< 40 kata) dan sebagian besar hurufnya kapital
    words = teks.split()
    if len(words) > 0 and len(words) < 40:
        huruf_alpha = [c for c in teks if c.isalpha()]
        if len(huruf_alpha) > 0:
            huruf_kapital = [c for c in huruf_alpha if c.isupper()]
            if (len(huruf_kapital) / len(huruf_alpha)) > 0.6:
                return True

    # 8. DETEKSI NAMA & GELAR AKADEMIK (Halaman Pengesahan)
    gelar_keywords = ['s.kom', 'm.kom', 's.si', 'm.si', 'dr.', 'prof.', 's.t.', 'm.t.', 's.e.', 'm.e.', 'mmsi', 'm.i.kom', 's.pd', 'm.pd', 'ph.d', 'b.sc', 'm.sc', 'ir.']
    if len(words) < 50:
        match_gelar = sum(1 for g in gelar_keywords if g in t_lower)
        if match_gelar >= 2: # Ada minimal 2 gelar akademik
            return True

    # 9. DETEKSI KALIMAT FORMAL PENGAJUAN SKRIPSI
    frasa_pengajuan = [
        "diajukan guna melengkapi",
        "syarat dalam mencapai gelar",
        "sarjana strata",
        "untuk memenuhi salah satu syarat",
        "memperoleh gelar sarjana",
        "telah dipertahankan di depan",
        "guna mencapai gelar"
    ]
    if any(frasa in t_lower for frasa in frasa_pengajuan):
        return True

    # 10. DETEKSI FORM ISIAN TITIK-TITIK KOSONG (Contoh: "Nama : .........")
    # Jika titiknya lebih dari 10 dan kata-katanya sedikit
    if t_lower.count('.') > 10 and len(words) < 30:
        return True

    return False

# =====================================================================
# MEMUAT MODEL (LOAD MODEL YANG SUDAH DILATIH OLEH TRAIN_MODEL.PY)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Memuat model SVM dan Feature Extractor yang sudah dilatih"""
    try:
        vectorizer = joblib.load('vectorizer.joblib')
        scaler = joblib.load('scaler.joblib')
        svm_model = joblib.load('svm_model.joblib')
        return {'tfidf': vectorizer, 'scaler': scaler, 'model': svm_model}
    except Exception as e:
        st.error(f"⚠️ Error memuat model: {e}. Pastikan Anda sudah menjalankan file 'train_model.py'!")
        return None

def ekstrak_dengan_posisi(file_upload):
    hasil_ekstraksi = []
    para_global_count = 1
    
    if file_upload.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file_upload)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                current_para = ""
                start_line = 1
                for line_num, line in enumerate(lines, 1):
                    if line.strip() == "":
                        if len(current_para.strip()) > 50:
                            hasil_ekstraksi.append({'teks': current_para.strip(), 'halaman': page_num, 'baris': start_line, 'paragraf': para_global_count})
                            para_global_count += 1
                        current_para = ""
                        start_line = line_num + 1
                    else:
                        if current_para == "": start_line = line_num
                        current_para += line + " "
                if len(current_para.strip()) > 50:
                    hasil_ekstraksi.append({'teks': current_para.strip(), 'halaman': page_num, 'baris': start_line, 'paragraf': para_global_count})
                    para_global_count += 1
    else:
        text = file_upload.getvalue().decode("utf-8")
        lines = text.split('\n')
        current_para = ""
        start_line = 1
        for line_num, line in enumerate(lines, 1):
            if line.strip() == "":
                if len(current_para.strip()) > 50:
                    hasil_ekstraksi.append({'teks': current_para.strip(), 'halaman': 1, 'baris': start_line, 'paragraf': para_global_count})
                    para_global_count += 1
                current_para = ""
                start_line = line_num + 1
            else:
                if current_para == "": start_line = line_num
                current_para += line + " "
        if len(current_para.strip()) > 50:
            hasil_ekstraksi.append({'teks': current_para.strip(), 'halaman': 1, 'baris': start_line, 'paragraf': para_global_count})
            para_global_count += 1
    return hasil_ekstraksi

# =====================================================================
# LOGIKA UTAMA APLIKASI
# =====================================================================
if True:
    # Memuat Model secara Otomatis
    if not st.session_state['model_trained']:
        st.session_state['model_pack'] = load_models()
        st.session_state['model_trained'] = True
    
    model_pack = st.session_state['model_pack']
    
    with st.sidebar:
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            try:
                if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
            except Exception: pass
            
        st.markdown("<h3 style='text-align: center;'>Portal Akademik</h3>", unsafe_allow_html=True)
        st.success(f"Selamat datang,\n**{st.session_state['username']}**")
        st.caption(f"Role Anda: {st.session_state['role']}")
        st.markdown("---")
        st.info("✨ **Keaslian Karya Terjamin!**\nSistem cerdas kami siap mendeteksi AI pada dokumen Anda. Pastikan karya Anda 100% orisinal dan tingkatkan kredibilitas akademik Anda!")
        if model_pack is not None:
            st.success("🚀 **Sistem Siap Digunakan!**\nUnggah dokumen Anda sekarang dan biarkan teknologi kami bekerja untuk Anda.")
        else:
            st.error("❌ **Sistem Sedang Gangguan!**\nMohon hubungi administrator.")
        st.markdown("---")
        


    col_t1, col_t2 = st.columns([1, 4], vertical_alignment="center")
    with col_t1:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
    with col_t2:
        st.markdown("<h1 style='margin-top: 0px; margin-bottom: 0px;'>AI DETECT ACADEMIC INTEGRITY SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📄 Upload Dokumen")
        file_tugas = st.file_uploader("Upload File (TXT, PDF)", type=["txt", "pdf"])
        tombol_scan = st.button("🚀 Mulai Pemindaian", use_container_width=True, type="primary")

    with col2:
        if tombol_scan:
            if file_tugas is not None:
                with st.spinner('Mengekstrak Fitur Stylometry & Menjalankan Prediksi SVM...'):
                    kumpulan_paragraf_posisi = ekstrak_dengan_posisi(file_tugas)
                    
                    if len(kumpulan_paragraf_posisi) == 0:
                        st.error("Teks tidak dapat dibaca atau terlalu pendek.")
                    else:
                        hasil_analisis = []
                        paragraf_ai = 0
                        paragraf_abu = 0
                        paragraf_manusia = 0
                        
                        count_chatgpt = 0
                        count_gemini = 0
                        count_claude = 0
                        
                        for item in kumpulan_paragraf_posisi:
                            teks_asli = item['teks']
                            teks_bersih = bersihkan_teks(teks_asli)
                            
                            if not teks_bersih.strip():
                                continue
                                
                            stylo_features = hitung_stylometry(teks_asli)
                            lex_div = stylo_features[0]
                            avg_len = stylo_features[1]
                            
                            # LOGIKA BYPASS (Mencegah Halaman 8 Daftar Struktural dan Kodingan terdeteksi AI)
                            if is_halaman_struktural(teks_asli):
                                status = "MANUSIA"
                                paragraf_manusia += 1
                                skor_final_tampil = 100.0 # Skor Manusia
                                sumber_ai = "-"
                                # Nilai default untuk bypass
                                word_count, char_count, avg_word_len, punct_count = 0, 0, 0, 0
                            else:
                                # Prediksi Normal SVM menggunakan fitur persis seperti train_model.py
                                stylo_new = extract_stylometric_features([teks_asli])[0]
                                word_count, char_count, avg_word_len, punct_count = stylo_new
                                
                                feat_tfidf = model_pack['tfidf'].transform([teks_bersih])
                                feat_stylo_scaled = model_pack['scaler'].transform([stylo_new])
                                fitur_gabungan = hstack([feat_tfidf, feat_stylo_scaled])
                                
                                probabilitas = model_pack['model'].predict_proba(fitur_gabungan)[0]
                                prob_manusia_pred = probabilitas[0] * 100
                                prob_ai = probabilitas[1] * 100
                                
                                sumber_ai = "-"
                                
                                if prob_ai > 60:
                                    status = "AI"
                                    paragraf_ai += 1
                                    skor_final_tampil = prob_ai
                                    
                                    # MENGIRIM TEKS ASLI KE FUNGSI TEBAK SUMBER AI
                                    sumber_ai = tebak_sumber_ai(teks_asli, lex_div, avg_len)
                                    if sumber_ai == "Campuran (Seri)":
                                        count_chatgpt += 1/3
                                        count_gemini += 1/3
                                        count_claude += 1/3
                                    else:
                                        if "ChatGPT" in sumber_ai: count_chatgpt += 1
                                        elif "Gemini" in sumber_ai: count_gemini += 1
                                        elif "Claude" in sumber_ai: count_claude += 1
                                    
                                elif prob_ai > 40:
                                    status = "ABU-ABU"
                                    paragraf_abu += 1
                                    skor_final_tampil = prob_ai
                                else:
                                    status = "MANUSIA"
                                    paragraf_manusia += 1
                                    skor_final_tampil = prob_manusia_pred
                            
                            hasil_analisis.append({
                                'teks': item['teks'],
                                'halaman': item['halaman'],
                                'paragraf': item['paragraf'],
                                'baris': item['baris'],
                                'skor': skor_final_tampil,
                                'status': status,
                                'sumber_ai': sumber_ai,
                                'lex_div': lex_div,
                                'avg_len': avg_len,
                                'word_count': word_count,
                                'char_count': char_count,
                                'avg_word_len': avg_word_len,
                                'punct_count': punct_count
                            })
                        
                        jumlah_p_valid = len(hasil_analisis)
                        skor_ai_dokumen = (paragraf_ai / jumlah_p_valid) * 100 if jumlah_p_valid > 0 else 0
                        skor_abu_dokumen = (paragraf_abu / jumlah_p_valid) * 100 if jumlah_p_valid > 0 else 0
                        skor_manusia_dokumen = 100.0 - skor_ai_dokumen - skor_abu_dokumen if jumlah_p_valid > 0 else 0
                        
                        st.session_state['hasil_analisis'] = hasil_analisis
                        st.session_state['ringkasan'] = {
                            'jumlah_p_valid': jumlah_p_valid,
                            'skor_ai_dokumen': skor_ai_dokumen,
                            'skor_abu_dokumen': skor_abu_dokumen,
                            'skor_manusia_dokumen': skor_manusia_dokumen,
                            'jml_ai': paragraf_ai,
                            'jml_abu': paragraf_abu,
                            'jml_manusia': paragraf_manusia,
                            'ai_stats': {'ChatGPT': count_chatgpt, 'Gemini': count_gemini, 'Claude': count_claude}
                        }
                        st.session_state['scan_done'] = True
                        st.session_state['active_filter'] = "Semua"

            else:
                st.warning("⚠️ Mohon upload file dokumen terlebih dahulu!")

        # =========================================================
        # TAMPILAN HASIL ANALISIS
        # =========================================================
        if st.session_state['scan_done']:
            tab1, tab2 = st.tabs(["📊 Ringkasan Eksekutif", "🔍 Analisis Per Paragraf (SVM)"])
            
            with tab1:
                st.subheader("Kesimpulan Algoritma SVM")
                r = st.session_state['ringkasan']
                
                # BLOK TEKS KESIMPULAN
                if r['skor_ai_dokumen'] > 40:
                    st.error("🚨 KESIMPULAN: TERDETEKSI AI TINGKAT TINGGI!")
                elif r['skor_ai_dokumen'] > 15:
                    st.warning("⚠️ KESIMPULAN: INDIKASI CAMPURAN (ABU-ABU)")
                else:
                    st.success("✅ KESIMPULAN: AMAN (MAYORITAS TULISAN MANUSIA)")
                
                # METRIK MENGGUNAKAN PERSENTASE & CUSTOM HTML
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #0d6efd;">{r['jumlah_p_valid']}</div>
                            <div class="metric-label">Total Paragraf</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_r2:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #dc3545;">{r['skor_ai_dokumen']:.1f}%</div>
                            <div class="metric-label">Terindikasi AI</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_r3:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #f59f00;">{r['skor_abu_dokumen']:.1f}%</div>
                            <div class="metric-label">Kemungkinan AI</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_r4:
                    st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #198754;">{r['skor_manusia_dokumen']:.1f}%</div>
                            <div class="metric-label">Tulisan Manusia</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.info("💡 **Informasi:** Persentase **Kemungkinan AI** (warna kuning) mengindikasikan paragraf yang terdeteksi sebagai campuran, hasil parafrase dari AI, atau ketidakyakinan algoritma antara AI/Manusia.")
                
                # INFORMASI HEURISTIK AI (DIPERBAIKI)
                if r['jml_ai'] > 0:
                    total_ai = r['jml_ai']
                    pct_chatgpt = (r['ai_stats']['ChatGPT'] / total_ai) * 100 if total_ai > 0 else 0
                    pct_gemini = (r['ai_stats']['Gemini'] / total_ai) * 100 if total_ai > 0 else 0
                    pct_claude = (r['ai_stats']['Claude'] / total_ai) * 100 if total_ai > 0 else 0

                    st.markdown(f"""
                        <div class="heuristik-box">
                            <div class="heuristik-title">
                                🤖 Analisis Heuristik (Probabilitas Model Bahasa AI yang Digunakan):
                            </div>
                            <div class="heuristik-item">
                                <span class="heuristik-model">ChatGPT (OpenAI)</span>
                                <span class="heuristik-pct">{pct_chatgpt:.1f}%</span>
                            </div>
                            <div class="heuristik-item">
                                <span class="heuristik-model">Gemini (Google)</span>
                                <span class="heuristik-pct">{pct_gemini:.1f}%</span>
                            </div>
                            <div class="heuristik-item">
                                <span class="heuristik-model">Claude (Anthropic)</span>
                                <span class="heuristik-pct">{pct_claude:.1f}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                hasil_semua = st.session_state['hasil_analisis']
                
                if not hasil_semua:
                    st.info("Tidak ada paragraf yang dianalisis.")
                else:
                    # IMPLEMENTASI FILTER DENGAN BUTTONS
                    st.markdown("### Filter Analisis")
                    
                    # Menggunakan columns untuk menjejalkan button secara horizontal
                    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1.2, 1])
                    
                    with f_col1:
                        if st.button("Semua Paragraf", use_container_width=True, 
                                     type="primary" if st.session_state['active_filter'] == "Semua" else "secondary"):
                            st.session_state['active_filter'] = "Semua"
                            st.rerun()
                    with f_col2:
                        if st.button("🔴 AI", use_container_width=True,
                                     type="primary" if st.session_state['active_filter'] == "AI" else "secondary"):
                            st.session_state['active_filter'] = "AI"
                            st.rerun()
                    with f_col3:
                        if st.button("🟡 Kemungkinan AI", use_container_width=True,
                                     type="primary" if st.session_state['active_filter'] == "ABU-ABU" else "secondary"):
                            st.session_state['active_filter'] = "ABU-ABU"
                            st.rerun()
                    with f_col4:
                        if st.button("🟢 Manusia", use_container_width=True,
                                     type="primary" if st.session_state['active_filter'] == "MANUSIA" else "secondary"):
                            st.session_state['active_filter'] = "MANUSIA"
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Tentukan data mana yang akan ditampilkan berdasarkan filter aktif
                    if st.session_state['active_filter'] == "Semua":
                        data_tampil = hasil_semua
                        st.markdown(f"Menampilkan **Semua Paragraf** ({len(data_tampil)} hasil)")
                    else:
                        data_tampil = [item for item in hasil_semua if item['status'] == st.session_state['active_filter']]
                        st.markdown(f"Menampilkan kategori **{st.session_state['active_filter']}** ({len(data_tampil)} hasil)")
                    
                    if len(data_tampil) == 0:
                        st.info("Tidak ada data untuk kategori ini.")
                    else:
                        # Render Kartu Paragraf (Tanpa Expander, langsung tampil)
                        for item in data_tampil:
                            if item['status'] == "AI":
                                warna_border = "#dc3545"
                                badge_html = f"<span style='background-color: #dc3545; color: white !important; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 13px;'>🤖 Prediksi SVM: AI ({item['skor']:.1f}%) | Model: {item['sumber_ai']}</span>"
                            elif item['status'] == "ABU-ABU":
                                warna_border = "#ffc107"
                                badge_html = f"<span style='background-color: #ffc107; color: black !important; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 13px;'>⚠️ Kemungkinan AI / Parafrase ({item['skor']:.1f}%)</span>"
                            else:
                                warna_border = "#198754"
                                badge_html = f"<span style='background-color: #198754; color: white !important; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 13px;'>🟢 Prediksi SVM: Manusia ({item['skor']:.1f}%)</span>"
                            
                            st.markdown(f"""
                            <div style='background-color: #ffffff; color: #111111; padding: 20px; border-radius: 10px; border-left: 6px solid {warna_border}; border-top: 1px solid #dee2e6; border-right: 1px solid #dee2e6; border-bottom: 1px solid #dee2e6; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;'>
                                    <div>{badge_html}</div>
                                    <div style='background-color: #f8f9fa; border: 1px solid #ddd; padding: 5px 12px; border-radius: 6px; font-size: 13px; color: #333; font-weight: bold;'>
                                        📄 Hal: {item['halaman']} &nbsp;|&nbsp; 📝 Para: {item['paragraf']}
                                    </div>
                                </div>
                                <div style='line-height: 1.7; font-size: 15px; text-align: justify; border-top: 1px dashed #eee; padding-top: 10px; margin-bottom: 10px;'>
                                    {item['teks']}
                                </div>
                                <div style='background-color: #E3F2FD; border-radius: 6px; padding: 10px; font-size: 12px; color: #000; border: 1px solid #90CAF9;'>
                                    <b>🔍 Ekstraksi Fitur Hybrid (Input Aktual untuk SVM):</b><br>
                                    • Jumlah Kata: <b>{int(item.get('word_count', 0))}</b> | Karakter: <b>{int(item.get('char_count', 0))}</b><br>
                                    • Rata-rata Panjang Kata: <b>{item.get('avg_word_len', 0):.2f}</b><br>
                                    • Jumlah Tanda Baca: <b>{int(item.get('punct_count', 0))}</b><br>
                                    • Term Frequency - Inverse Document Frequency (TF-IDF): <b>Aktif</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)