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
    page_title="AI Visual Catalog Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LUXURY ENTERPRISE CSS STYLING ---
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hide Default Streamlit Chrome */
    [data-testid="stSidebar"], #MainMenu, footer, header { display: none !important; }
    
    /* Global Container Setup */
    .stApp { 
        background-color: #ffffff !important; 
    }
    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1100px !important; 
    }

    /* Primary Accent Color Settings */
    :root {
        --brand-gold: #b8976c;
        --brand-gold-hover: #9e7f55;
        --brand-dark: #0f172a;
        --border-color: #e2e8f0;
    }

    /* Custom Header Banner */
    .enterprise-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 400;
    }
    .header-badge {
        background: rgba(184, 151, 108, 0.15);
        border: 1px solid rgba(184, 151, 108, 0.4);
        color: #d4af37;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Custom Input & Select Box Styling */
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 0.9rem !important;
    }
    
    /* Styled Buttons */
    .stButton>button {
        background-color: var(--brand-gold) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(184, 151, 108, 0.25) !important;
    }
    .stButton>button:hover {
        background-color: var(--brand-gold-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(184, 151, 108, 0.35) !important;
    }

    /* Result Card Styling */
    .result-card-exact {
        background: #faf8f5;
        border: 1px solid #e8dfd1;
        border-left: 5px solid #15803d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .result-card-alt {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0284c7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .card-badge-exact {
        background-color: #15803d;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: inline-block;
        margin-bottom: 12px;
    }
    .card-badge-alt {
        background-color: #0284c7;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Meta Info Grid */
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 8px;
    }
    .meta-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #334155;
    }
    .meta-box b {
        color: #0f172a;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #f1f5f9;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        border-radius: 8px 8px 0px 0px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--brand-gold) !important;
        border-bottom-color: var(--brand-gold) !important;
    }
</style>
""", unsafe_allow_html=True)

# RAM Caching for DINOv2 Vector Engine
@st.cache_resource(show_spinner="Initializing Neural Vector Engine...")
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

# --- BRAND HEADER ---
st.markdown("""
<div class="enterprise-header">
    <div>
        <div class="header-title">✨ AI Visual Pattern & Catalog Search Engine</div>
        <div class="header-subtitle">Neural Visual Retrieval System • Universal Hardware Products</div>
    </div>
    <div class="header-badge">Enterprise Edition</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Visual Pattern Matcher", "⚡ Instant PDF Indexer"])

# TAB 1: PRODUCTION SEARCH
with tab1:
    if st.session_state.get("catalog_indexed", False):
        companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
        companies.insert(0, "All Companies")
        
        col_filter, col_sync = st.columns([3.2, 1])
        with col_filter:
            selected_company = st.selectbox("Filter Search by Brand Library:", companies)
        with col_sync:
            st.write("")
            st.write("")
            if st.button("🔄 Sync Google Drive", use_container_width=True):
                with st.spinner("Synchronizing Drive catalogs..."):
                    if run_auto_sync():
                        st.cache_resource.clear()
                        st.success("Database updated successfully!")
                        st.rerun()
                    else:
                        st.info("Catalogs are fully up to date.")

        search_file = st.file_uploader("Upload or Capture Reference Photo / Texture", type=["jpg", "png", "jpeg"])
        
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown("<p style='font-weight:600; color:#1e293b; margin-top:15px;'>✂️ Frame / Crop Pattern Area:</p>", unsafe_allow_html=True)
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#b8976c', 
                aspect_ratio=None
            )
            
            with st.spinner("Extracting visual features & matching index..."):
                q_emb = engine.get_single_embedding(cropped_img)
                raw_matches = engine.search(q_emb, top_k=15, min_confidence=0.50)
                
                filtered_matches = [
                    m for m in raw_matches 
                    if selected_company == "All Companies" or m["meta"].get("company", "General") == selected_company
                ]
                
                exact_matches = [m for m in filtered_matches if m["score"] >= 0.75]
                alternative_matches = [m for m in filtered_matches if 0.50 <= m["score"] < 0.75]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if exact_matches:
                st.markdown("<h4 style='color:#0f172a; font-weight:700;'>🎯 Direct Match Results</h4>", unsafe_allow_html=True)
                for i, res in enumerate(exact_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="result-card-exact">
                        <span class="card-badge-exact">Match #{i+1} — {score_pct:.1f}% Confidence</span>
                        <div class="meta-grid">
                            <div class="meta-box">🏢 <b>Brand:</b> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-box">📖 <b>Catalog:</b> {res['meta']['catalog']}</div>
                            <div class="meta-box">📄 <b>Page:</b> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
                    
            elif alternative_matches:
                st.info("💡 **Exact match not found. Displaying closest visually similar alternatives:**")
                st.markdown("<h4 style='color:#0f172a; font-weight:700;'>🎨 Recommended Alternatives</h4>", unsafe_allow_html=True)
                for i, res in enumerate(alternative_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="result-card-alt">
                        <span class="card-badge-alt">Alternative #{i+1} — {score_pct:.1f}% Visual Similarity</span>
                        <div class="meta-grid">
                            <div class="meta-box">🏢 <b>Brand:</b> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-box">📖 <b>Catalog:</b> {res['meta']['catalog']}</div>
                            <div class="meta-box">📄 <b>Page:</b> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
            else:
                st.warning(f"❌ No matching pattern found under '{selected_company}'. Try adjusting the crop area or set filter to 'All Companies'.")
    else:
        st.info("No catalogs indexed yet. Use 'Sync Google Drive' or the PDF Indexer tab to populate the database.")

# TAB 2: STANDALONE UPLOADER
with tab2:
    st.markdown("<h4 style='color:#0f172a; font-weight:700; margin-top:10px;'>⚡ Direct PDF Catalog Indexer</h4>", unsafe_allow_html=True)
    company_name = st.text_input("Brand Tag / Manufacturer Name:", value="General")
    uploaded_pdfs = st.file_uploader("Select PDF Catalog Files to Vectorize", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs and st.button("Process & Generate Vector Index", type="primary"):
        with st.spinner("Rendering PDF pages & computing visual embeddings..."):
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
                st.success(f"Successfully processed {len(pages_to_embed)} pages!")
                st.rerun()