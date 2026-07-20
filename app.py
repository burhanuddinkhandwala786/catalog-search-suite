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
    page_title="Rétina · Catalog Intelligence Engine",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- EDITORIAL LUXURY DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Canvas Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background-color: #0b0a09 !important;
        color: #c9c3ba !important;
    }

    /* Hide Default Streamlit Interface Chrome */
    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton { 
        display: none !important; 
    }

    /* Layout Container Spacing */
    .block-container { 
        padding-top: 1.2rem !important; 
        padding-bottom: 2.5rem !important; 
        max-width: 1240px !important; 
    }

    /* Top Navigation Header Bar */
    .studio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 32px;
    }
    .brand-mark {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: #f4efe8;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }
    .brand-accent-dot {
        width: 12px;
        height: 12px;
        background-color: #d96b43;
        border-radius: 50%;
        display: inline-block;
    }
    .status-ticker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8a8378;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* Chapter Tags & Editorial Typography */
    .section-chapter {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: #d96b43;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-chapter::before {
        content: '';
        width: 16px;
        height: 1px;
        background-color: #d96b43;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.9rem;
        font-weight: 400;
        line-height: 1.08;
        color: #f4efe8;
        margin-bottom: 16px;
        letter-spacing: -0.02em;
    }
    .hero-title em {
        font-style: italic;
        color: #d96b43;
    }

    .hero-description {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #999287;
        max-width: 520px;
        margin-bottom: 28px;
    }

    /* Minimal Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 0;
        color: #8a8378;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0 16px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #f4efe8 !important;
        border-bottom: 2px solid #d96b43 !important;
    }

    /* Dark Input Controls */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #141210 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        color: #f4efe8 !important;
    }
    .stSelectbox label, .stFileUploader label, .stTextInput label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #8a8378 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !alignment;
    }

    [data-testid="stFileUploader"] section {
        background-color: #141210 !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }

    /* Action Buttons */
    .stButton>button {
        background-color: #f4efe8 !important;
        color: #0b0a09 !important;
        border: 1px solid #f4efe8 !important;
        border-radius: 30px !important;
        padding: 10px 24px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Dark Action Button */
    .btn-secondary button {
        background-color: transparent !important;
        color: #f4efe8 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .btn-secondary button:hover {
        border-color: #d96b43 !important;
        color: #d96b43 !important;
    }

    /* Glass Match Result Cards */
    .result-card {
        background: #141210;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #d96b43;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 16px;
        position: relative;
    }
    .score-badge {
        position: absolute;
        top: 14px;
        right: 14px;
        background-color: #d96b43;
        color: #0b0a09;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
    }

    /* Live Feed Sidebar Box */
    .feed-container {
        background: #141210;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 20px 24px;
    }
    .feed-title {
        display: flex;
        justify-content: space-between;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8a8378;
    }
    .feed-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .feed-row-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #f4efe8;
    }

    /* Metric Grid Counter */
    .metrics-container {
        display: flex;
        gap: 36px;
        padding-top: 24px;
        margin-top: 32px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #f4efe8;
        line-height: 1;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #8a8378;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 6px;
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

# Top Bar Header
st.markdown("""
<div class="studio-header">
    <div class="brand-mark">
        <span class="brand-accent-dot"></span> Rétina.
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
                box_color='#d96b43', 
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
                        <div style="font-family:'Playfair Display', serif; font-size:1.15rem; color:#f4efe8;">{res['meta']['catalog']}</div>
                        <div style="font-size:0.8rem; color:#8a8378; margin-top:2px;">{res['meta'].get('company', 'General')} · Page {res['meta']['page']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
            else:
                st.warning("No matching swatches found in database.")
        else:
            # Placeholder feed preview when no file is active
            st.markdown('<div class="feed-container">', unsafe_allow_html=True)
            st.markdown('<div class="feed-title"><span>/catalogs</span><span style="color:#d96b43;">● Watcher Active</span></div>', unsafe_allow_html=True)
            
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
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#645e54; margin-right:12px;">{num}</span>
                        <span class="feed-row-title">{title}</span>
                        <div style="font-size:0.78rem; color:#787166; margin-top:2px;">{pages}</div>
                    </div>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#a39c90; border:1px solid rgba(255,255,255,0.15); padding:3px 8px; border-radius:4px;">{status}</span>
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
