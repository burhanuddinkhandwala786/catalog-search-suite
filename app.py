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
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- EDITORIAL LUXURY UI STYLING (MATCHING REFERENCE EXACTLY) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background-color: #0d0c0b !important;
        color: #c5bebe !important;
    }

    /* Hide Streamlit Native Chrome */
    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton { 
        display: none !important; 
    }

    /* Main Canvas Spacing */
    .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 2.5rem !important; 
        max-width: 1240px !important; 
    }

    /* Brand Top Navigation Bar */
    .retina-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0 24px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 30px;
    }
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #f3efe6;
        letter-spacing: -0.01em;
    }
    .brand-dot {
        width: 14px;
        height: 14px;
        background-color: #e06338;
        border-radius: 50%;
        display: inline-block;
    }
    .nav-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8c857b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Typography Hierarchy */
    .chapter-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: #e06338;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .chapter-tag::before {
        content: '';
        width: 18px;
        height: 1px;
        background-color: #e06338;
    }

    .editorial-heading {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 400;
        line-height: 1.1;
        color: #f3efe6;
        margin-bottom: 16px;
        letter-spacing: -0.02em;
    }
    .editorial-heading em {
        font-style: italic;
        color: #e06338;
    }

    .editorial-desc {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #9c958a;
        max-width: 520px;
        margin-bottom: 24px;
    }

    /* Pill Buttons (Primary Cream & Secondary Dark) */
    .stButton>button {
        border-radius: 30px !important;
        padding: 10px 24px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    
    /* Primary Pill Button */
    .primary-pill button {
        background-color: #f3efe6 !important;
        color: #0d0c0b !important;
        border: 1px solid #f3efe6 !important;
    }
    .primary-pill button:hover {
        background-color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Pill Button */
    .secondary-pill button {
        background-color: transparent !important;
        color: #f3efe6 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .secondary-pill button:hover {
        border-color: #e06338 !important;
        color: #e06338 !important;
    }

    /* Stats Counter Footer */
    .stats-row {
        display: flex;
        gap: 40px;
        padding-top: 24px;
        margin-top: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #f3efe6;
        line-height: 1;
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #8c857b;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* Right Panel Catalog Feed */
    .feed-panel {
        background: #121110;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 20px 24px;
    }
    .feed-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8c857b;
    }

    .feed-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .feed-item-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        color: #f3efe6;
    }
    .feed-item-sub {
        font-size: 0.78rem;
        color: #787268;
        margin-top: 2px;
    }

    /* Tag Badges */
    .badge-indexed {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #a8a196;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.08em;
    }
    .badge-syncing {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #e06338;
        border: 1px solid rgba(224, 99, 56, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.08em;
    }

    /* Input & Select Box Customization */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #141312 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        color: #f3efe6 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #121110 !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }

    /* Match Result Card Overlay */
    .match-card {
        background: #121110;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 16px;
        position: relative;
    }
    .match-card-exact {
        border-left: 3px solid #e06338;
    }
    .score-overlay {
        position: absolute;
        top: 12px;
        right: 12px;
        background-color: #e06338;
        color: #0d0c0b;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 2px;
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

# --- BRAND HEADER BAR ---
total_pages = len(engine.metadata) if st.session_state.get("catalog_indexed", False) else 0
total_companies = len(set(m.get("company", "General") for m in engine.metadata)) if st.session_state.get("catalog_indexed", False) else 0
total_catalogs = len(set(m.get("catalog", "") for m in engine.metadata)) if st.session_state.get("catalog_indexed", False) else 0

st.markdown("""
<div class="retina-nav">
    <div class="brand-logo">
        <span class="brand-dot"></span> Rétina.
    </div>
    <div class="nav-status">
        ● LIVE · AUTOMATED INDEXING ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN INTERFACE TABS
tab1, tab2 = st.tabs(["Search & Match", "Index Management"])

# --- TAB 1: VISUAL SEARCH INTERFACE ---
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="chapter-tag">CHAPTER 01 · THE INTERFACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="editorial-heading">One swatch.<br><em>Every catalog.</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="editorial-desc">Drop a fabric photo, panel cut-out, or finish detail. Rétina cross-references it against every indexed catalog page in under half a second.</div>', unsafe_allow_html=True)
        
        if st.session_state.get("catalog_indexed", False):
            companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
            companies.insert(0, "All Collections")
            selected_company = st.selectbox("Filter Reference Collection:", companies)
        else:
            selected_company = "All Collections"

        search_file = st.file_uploader("Upload Reference Swatch", type=["jpg", "png", "jpeg"])

    with col_right:
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown('<div class="chapter-tag">INPUT · REFERENCE</div>', unsafe_allow_html=True)
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#e06338', 
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
            st.markdown('<div class="chapter-tag">OUTPUT · TOP MATCHES</div>', unsafe_allow_html=True)
            
            matches_to_show = exact_matches if exact_matches else alternative_matches
            
            if matches_to_show:
                for i, res in enumerate(matches_to_show[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="match-card match-card-exact">
                        <div class="score-overlay">{score_pct:.1f}%</div>
                        <div style="font-family:'Playfair Display', serif; font-size:1.1rem; color:#f3efe6;">{res['meta']['catalog']}</div>
                        <div style="font-size:0.8rem; color:#8c857b; margin-top:2px;">{res['meta'].get('company', 'General')} · Page {res['meta']['page']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
            else:
                st.warning("No matching swatches found in catalog index.")
        else:
            # Placeholder feed preview when no file uploaded
            st.markdown('<div class="feed-panel">', unsafe_allow_html=True)
            st.markdown('<div class="feed-header"><span>/catalogs</span><span style="color:#e06338;">● Watcher active</span></div>', unsafe_allow_html=True)
            
            sample_items = [
                ("01", "Marmi Firenze · Vol. IV", "312 pages", "INDEXED"),
                ("02", "Kvadrat · Wool 2024", "184 pages", "INDEXED"),
                ("03", "Fornace Brioni · Terracotta", "96 pages", "INDEXED"),
                ("04", "Listone Giordano · Oak", "148 pages", "SYNCING")
            ]
            
            for num, title, pages, status in sample_items:
                badge_class = "badge-syncing" if status == "SYNCING" else "badge-indexed"
                st.markdown(f"""
                <div class="feed-item">
                    <div>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#646059; margin-right:12px;">{num}</span>
                        <span class="feed-item-title">{title}</span>
                        <div class="feed-item-sub">{pages}</div>
                    </div>
                    <span class="{badge_class}">{status}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: INDEX MANAGEMENT ---
with tab2:
    col_mgmt_left, col_mgmt_right = st.columns([1.1, 1], gap="large")
    
    with col_mgmt_left:
        st.markdown('<div class="chapter-tag">CHAPTER 05 · INDEX MANAGEMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="editorial-heading">Every catalog,<br><em>on the record.</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="editorial-desc">Rétina ingests PDFs from Drive automatically. The vector index rebuilds instantly, notifies webhooks, and syncs across your entire suite.</div>', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.markdown('<div class="secondary-pill">', unsafe_allow_html=True)
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
        <div class="stats-row">
            <div>
                <div class="stat-number">{total_catalogs if total_catalogs > 0 else 14}</div>
                <div class="stat-label">CATALOGS</div>
            </div>
            <div>
                <div class="stat-number">{total_pages if total_pages > 0 else 1480}</div>
                <div class="stat-label">PAGES</div>
            </div>
            <div>
                <div class="stat-number">{total_companies if total_companies > 0 else 4}</div>
                <div class="stat-label">BRANDS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_mgmt_right:
        st.markdown('<div class="chapter-tag">MANUAL UPLOADER</div>', unsafe_allow_html=True)
        company_name = st.text_input("Brand / Collection Tag:", value="General")
        uploaded_pdfs = st.file_uploader("Select PDF Catalogs", type=["pdf"], accept_multiple_files=True)
        
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
