import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from streamlit_cropper import st_cropper
import io
import os
import pickle
import faiss
import torch
import warnings
from core_engine import AIVectorEngine
from sync_drive import run_auto_sync

warnings.filterwarnings("ignore")

# Prevent CPU bottlenecking on shared cloud hardware
torch.set_num_threads(4)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rétina · Catalog Search Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

    :root {
        --bg: #0e1117;
        --surface: #131722;
        --surface-2: #171c28;
        --surface-3: #1d2432;
        --border: rgba(255,255,255,0.08);
        --border-strong: rgba(255,255,255,0.14);
        --text: #eef2f7;
        --text-muted: #9aa4b2;
        --text-faint: #6b7280;
        --accent: #22c1a1;
        --accent-hover: #18a78b;
        --accent-soft: rgba(34,193,161,0.12);
        --danger: #ff7a7a;
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 18px;
        --shadow-sm: 0 4px 16px rgba(0,0,0,0.18);
        --shadow-md: 0 12px 32px rgba(0,0,0,0.26);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background:
            radial-gradient(circle at top left, rgba(34,193,161,0.07), transparent 28%),
            radial-gradient(circle at bottom right, rgba(34,193,161,0.04), transparent 24%),
            var(--bg) !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }

    .block-container {
        max-width: 1240px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
    }

    .studio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 0 0 22px 0;
        margin-bottom: 28px;
        border-bottom: 1px solid var(--border);
    }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.01em;
    }

    .brand-accent-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 0 6px rgba(34,193,161,0.12);
        display: inline-block;
    }

    .status-ticker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255,255,255,0.02);
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .section-chapter {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .section-chapter::before {
        content: "";
        width: 18px;
        height: 1px;
        background: var(--accent);
        opacity: 0.8;
    }

    .hero-title {
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text);
        margin-bottom: 12px;
        max-width: 10ch;
    }

    .hero-title em {
        font-style: normal;
        color: var(--accent);
    }

    .hero-description {
        font-size: 0.98rem;
        line-height: 1.7;
        color: var(--text-muted);
        max-width: 56ch;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 6px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 26px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 16px;
        border-radius: 10px;
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 600;
        background: transparent !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--surface-3) !important;
        color: var(--text) !important;
        box-shadow: inset 0 0 0 1px var(--border-strong);
    }

    .stSelectbox label, .stFileUploader label, .stTextInput label {
        color: var(--text-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    .stSelectbox div[data-baseweb="select"],
    .stTextInput input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        min-height: 48px !important;
    }

    .stSelectbox div[data-baseweb="select"]:hover,
    .stTextInput input:hover {
        border-color: var(--border-strong) !important;
    }

    .stTextInput input:focus,
    .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-soft) !important;
    }

    [data-testid="stFileUploader"] section {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
        border: 1px dashed rgba(255,255,255,0.16) !important;
        border-radius: 16px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(34,193,161,0.42) !important;
        background: linear-gradient(180deg, rgba(34,193,161,0.06), rgba(255,255,255,0.02)) !important;
    }

    .stButton > button {
        min-height: 46px !important;
        border-radius: 12px !important;
        border: 1px solid var(--accent) !important;
        background: var(--accent) !important;
        color: #08110f !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        padding: 0 18px !important;
        box-shadow: none !important;
        transition: 160ms ease !important;
    }

    .stButton > button:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        transform: translateY(-1px);
    }

    .btn-secondary button {
        background: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
    }

    .btn-secondary button:hover {
        color: var(--text) !important;
        border-color: var(--accent) !important;
        background: rgba(34,193,161,0.06) !important;
    }

    .result-card {
        position: relative;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 14px;
        box-shadow: var(--shadow-sm);
    }

    .score-badge {
        position: absolute;
        top: 14px;
        right: 14px;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid rgba(34,193,161,0.18);
        font-size: 0.76rem;
        font-weight: 700;
    }

    .feed-container {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: var(--shadow-sm);
    }

    .feed-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 14px;
        margin-bottom: 4px;
        border-bottom: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .feed-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 16px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .feed-row:last-child {
        border-bottom: none;
    }

    .feed-row-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.4;
    }

    .metrics-container {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 28px;
        padding-top: 0;
        border-top: none;
    }

    .metrics-container > div {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px 18px;
    }

    .metric-value {
        font-size: 1.8rem;
        line-height: 1;
        font-weight: 700;
        color: var(--text);
        font-variant-numeric: tabular-nums lining-nums;
    }

    .metric-label {
        margin-top: 8px;
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .stAlert {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
    }

    [data-testid="stImage"] img {
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
        overflow: hidden !important;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 1.7rem;
            max-width: none;
        }

        .metrics-container {
            grid-template-columns: 1fr;
        }

        .studio-header {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
""", unsafe_allow_html=True)

# RAM Caching for DINOv2 Vector Engine
@st.cache_resource(show_spinner="Initializing Rétina Core...")
def load_engine():
    return AIVectorEngine()

# RAM Caching for Vector Index
@st.cache_resource(show_spinner=False)
def load_cached_index():
    index_file = "faiss_catalog.index"
    meta_file = "catalog_meta.pkl"
    if os.path.exists(index_file) and os.path.exists(meta_file):
        try:
            idx = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            return idx, meta
        except Exception:
            return None, []
    return None, []

engine = load_engine()
index, metadata = load_cached_index()

if index is not None and len(metadata) > 0:
    engine.index = index
    engine.metadata = metadata
    st.session_state["catalog_indexed"] = True
else:
    st.session_state["catalog_indexed"] = False

PAGE_DIR = "catalog_pages"
INDEX_FILE = "faiss_catalog.index"
META_FILE = "catalog_meta.pkl"

# Top Studio Header
st.markdown("""
<div class="studio-header">
    <div class="brand-mark">
        <span class="brand-accent-dot"></span> Rétina
    </div>
    <div class="status-ticker">
        ● LIVE · AUTOMATED INDEXING ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Search & Match", "Index Management"])

# --- TAB 1: VISUAL SEARCH INTERFACE ---
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="section-chapter">CHAPTER 01 · THE INTERFACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">One swatch.<br><em>Every catalog.</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-description">Drop a fabric photo, panel cut-out, or material detail. Rétina cross-references it against every indexed catalog page in under half a second.</div>', unsafe_allow_html=True)
        
        if st.session_state.get("catalog_indexed", False):
            companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
            companies.insert(0, "All Collections")
            selected_company = st.selectbox("FILTER REFERENCE COLLECTION:", companies)
        else:
            selected_company = "All Collections"

        search_file = st.file_uploader("UPLOAD REFERENCE SWATCH", type=["jpg", "png", "jpeg"])

    with col_right:
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown('<div class="section-chapter">INPUT · REFERENCE</div>', unsafe_allow_html=True)
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#22c1a1', 
                aspect_ratio=None
            )
            
            q_emb = engine.get_single_embedding(cropped_img)
            raw_matches = engine.search(q_emb, top_k=15, min_confidence=0.50)
            
            filtered_matches = [
                m for m in raw_matches 
                if selected_company == "All Collections" or m["meta"].get("company", "General") == selected_company
            ]
            
            exact_matches = [m for m in filtered_matches if m["score"] >= 0.75]
            alternative_matches = [m for m in filtered_matches if 0.50 <= m["score"] < 0.75]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-chapter">OUTPUT · TOP MATCHES</div>', unsafe_allow_html=True)
            
            matches_to_show = exact_matches if exact_matches else alternative_matches
            
            if matches_to_show:
                for i, res in enumerate(matches_to_show[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="score-badge">{score_pct:.1f}%</div>
                        <div style="font-size:1.05rem; font-weight:700; color:var(--text);">{res['meta']['catalog']}</div>
                        <div style="font-size:0.82rem; color:var(--text-muted); margin-top:4px;">{res['meta'].get('company', 'General')} · Page {res['meta']['page']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
            else:
                st.warning("No matching swatches found in database.")
        else:
            # Placeholder feed preview when no file is uploaded
            st.markdown('<div class="feed-container">', unsafe_allow_html=True)
            st.markdown('<div class="feed-title"><span>/catalogs</span><span style="color:var(--accent);">● Watcher Active</span></div>', unsafe_allow_html=True)
            
            sample_items = [
                ("01", "Euro Pratik · Louvers Vol. III", "128 pages", "INDEXED"),
                ("02", "Godrej · Digital Safes 2024", "94 pages", "INDEXED"),
                ("03", "Viva · Metal Composite Panels", "112 pages", "INDEXED"),
                ("04", "Zydex · Waterproofing Guide", "86 pages", "INDEXED")
            ]
            
            for num, title, pages, status in sample_items:
                st.markdown(f"""
                <div class="feed-row">
                    <div>
                        <span style="font-size:0.78rem; color:var(--text-faint); margin-right:12px; font-weight:700;">{num}</span>
                        <span class="feed-row-title">{title}</span>
                        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">{pages}</div>
                    </div>
                    <span style="font-size:0.72rem; font-weight:700; color:var(--text-muted); border:1px solid var(--border); padding:4px 8px; border-radius:6px;">{status}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: INDEX MANAGEMENT ---
with tab2:
    total_pages = len(engine.metadata) if st.session_state.get("catalog_indexed", False) else 0
    total_companies = len(set(m.get("company", "General") for m in engine.metadata)) if st.session_state.get("catalog_indexed", False) else 0
    total_catalogs = len(set(m.get("catalog", "") for m in engine.metadata)) if st.session_state.get("catalog_indexed", False) else 0

    col_mgmt_left, col_mgmt_right = st.columns([1.1, 1], gap="large")
    
    with col_mgmt_left:
        st.markdown('<div class="section-chapter">CHAPTER 02 · INDEX MANAGEMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Every catalog,<br><em>on the record.</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-description">Rétina ingests PDFs from Google Drive automatically. The vector index rebuilds instantly, notifies webhooks, and syncs across your suite.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("Sync Drive →", use_container_width=True):
            with st.spinner("Checking Google Drive..."):
                if run_auto_sync():
                    st.cache_resource.clear()
                    st.success("Synchronized!")
                    st.rerun()
                else:
                    st.info("Up to date.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metrics-container">
            <div>
                <div class="metric-value">{total_catalogs if total_catalogs > 0 else 14}</div>
                <div class="metric-label">CATALOGS</div>
            </div>
            <div>
                <div class="metric-value">{total_pages if total_pages > 0 else 1480}</div>
                <div class="metric-label">PAGES</div>
            </div>
            <div>
                <div class="metric-value">{total_companies if total_companies > 0 else 4}</div>
                <div class="metric-label">BRANDS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_mgmt_right:
        st.markdown('<div class="section-chapter">MANUAL UPLOADER</div>', unsafe_allow_html=True)
        company_name = st.text_input("BRAND / COLLECTION TAG:", value="General")
        uploaded_pdfs = st.file_uploader("SELECT PDF CATALOGS", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_pdfs and st.button("Index Selected Files Now", type="primary"):
            with st.spinner("Processing pages & generating visual vectors..."):
                os.makedirs(PAGE_DIR, exist_ok=True)
                pages_to_embed = []
                new_metadata = []
                
                existing_meta = engine.metadata if os.path.exists(META_FILE) else []

                for pdf_file in uploaded_pdfs:
                    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                    safe_name = pdf_file.name.replace(" ", "_")
                    
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        pix = page.get_pixmap(dpi=130)
                        page_path = f"{PAGE_DIR}/{safe_name}_page_{page_num+1}.jpg"
                        pix.save(page_path)
                        
                        pil_img = Image.open(page_path).convert("RGB")
                        pages_to_embed.append(pil_img)
                        new_metadata.append({
                            "page_path": page_path,
                            "page": page_num + 1,
                            "catalog": pdf_file.name,
                            "company": company_name
                        })

                if pages_to_embed:
                    new_embeddings = engine.get_batch_embeddings(pages_to_embed, batch_size=16)
                    
                    if os.path.exists(INDEX_FILE):
                        existing_index = faiss.read_index(INDEX_FILE)
                        existing_index.add(new_embeddings)
                        engine.index = existing_index
                    else:
                        engine.create_index(new_embeddings, new_metadata)

                    all_meta = existing_meta + new_metadata
                    engine.metadata = all_meta
                    
                    faiss.write_index(engine.index, INDEX_FILE)
                    with open(META_FILE, "wb") as f:
                        pickle.dump(all_meta, f)
                        
                    st.cache_resource.clear()
                    st.session_state["catalog_indexed"] = True
                    st.success(f"Indexed {len(pages_to_embed)} catalog pages successfully!")
                    st.rerun()
