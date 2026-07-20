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
    page_title="Visual Catalog Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ELEGANT ENTERPRISE UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Hide Streamlit Native Chrome */
    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton { 
        display: none !important; 
    }

    /* Container Spacing */
    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1000px !important; 
    }

    /* Custom Minimal Header */
    .app-header {
        text-align: center;
        padding: 10px 0 25px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 25px;
    }
    .app-header-subtitle {
        color: #b8976c;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .app-header-title {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }

    /* Streamlit Select Box & Inputs Styling */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #faf8f5 !important;
    }
    .stSelectbox label, .stFileUploader label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em;
    }

    /* Primary Accent Button */
    .stButton>button {
        background-color: #b8976c !important;
        color: #ffffff !important;
        border: 1px solid #b8976c !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(184, 151, 108, 0.2) !important;
    }
    .stButton>button:hover {
        background-color: #a38258 !important;
        border-color: #a38258 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(184, 151, 108, 0.3) !important;
    }

    /* Result Cards Design */
    .match-container-exact {
        background: #fcfbf9;
        border: 1px solid #e2d9cd;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    .match-container-alt {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    
    .match-header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .tag-exact {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .tag-alt {
        background-color: #f0f9ff;
        color: #0284c7;
        border: 1px solid #bae6fd;
    }

    /* Meta Information Grid */
    .meta-details-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .meta-item-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #475569;
    }
    .meta-item-box strong {
        color: #0f172a;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px 6px 0 0;
        color: #64748b;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        color: #b8976c !important;
        border-bottom-color: #b8976c !important;
    }

    /* Image Display Wrapper */
    .result-image-frame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# RAM Caching for DINOv2 Vector Engine
@st.cache_resource(show_spinner="Loading Visual Recognition Engine...")
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

# --- ELEGANT MINIMAL HEADER ---
st.markdown("""
<div class="app-header">
    <div class="app-header-subtitle">INSTANT PATTERN RECOGNITION</div>
    <div class="app-header-title">AI Catalog Search Engine</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Visual Pattern Search", "⚙️ Index Management"])

# TAB 1: PRODUCTION SEARCH
with tab1:
    if st.session_state.get("catalog_indexed", False):
        companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
        companies.insert(0, "All Brand Libraries")
        
        col_filter, col_sync = st.columns([3.2, 1])
        with col_filter:
            selected_company = st.selectbox("Select Brand Collection:", companies)
        with col_sync:
            st.write("")
            st.write("")
            if st.button("🔄 Sync Drive", use_container_width=True):
                with st.spinner("Checking Google Drive..."):
                    if run_auto_sync():
                        st.cache_resource.clear()
                        st.success("Catalogs synchronized!")
                        st.rerun()
                    else:
                        st.info("Database is up to date.")

        search_file = st.file_uploader("Upload or Capture Reference Image", type=["jpg", "png", "jpeg"])
        
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown("<p style='font-weight:600; color:#334155; font-size:0.88rem; margin-top:16px;'>Crop Target Texture or Pattern Area:</p>", unsafe_allow_html=True)
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#b8976c', 
                aspect_ratio=None
            )
            
            with st.spinner("Searching neural index for visual matches..."):
                q_emb = engine.get_single_embedding(cropped_img)
                raw_matches = engine.search(q_emb, top_k=15, min_confidence=0.50)
                
                filtered_matches = [
                    m for m in raw_matches 
                    if selected_company == "All Brand Libraries" or m["meta"].get("company", "General") == selected_company
                ]
                
                exact_matches = [m for m in filtered_matches if m["score"] >= 0.75]
                alternative_matches = [m for m in filtered_matches if 0.50 <= m["score"] < 0.75]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if exact_matches:
                st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.1rem;'>🎯 Exact Match Results</h4>", unsafe_allow_html=True)
                for i, res in enumerate(exact_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="match-container-exact">
                        <div class="match-header-tag tag-exact">
                            <span>Direct Match #{i+1}</span> • <span>{score_pct:.1f}% Confidence</span>
                        </div>
                        <div class="meta-details-grid">
                            <div class="meta-item-box">🏢 <strong>Brand:</strong> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-item-box">📖 <strong>Catalog:</strong> {res['meta']['catalog']}</div>
                            <div class="meta-item-box">📄 <strong>Location:</strong> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
                    
            elif alternative_matches:
                st.info("💡 **No exact product match found. Displaying closest matching alternatives:**")
                st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.1rem;'>🎨 Recommended Alternatives</h4>", unsafe_allow_html=True)
                for i, res in enumerate(alternative_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="match-container-alt">
                        <div class="match-header-tag tag-alt">
                            <span>Alternative #{i+1}</span> • <span>{score_pct:.1f}% Visual Similarity</span>
                        </div>
                        <div class="meta-details-grid">
                            <div class="meta-item-box">🏢 <strong>Brand:</strong> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-item-box">📖 <strong>Catalog:</strong> {res['meta']['catalog']}</div>
                            <div class="meta-item-box">📄 <strong>Location:</strong> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
            else:
                st.warning(f"❌ No matching pattern found under '{selected_company}'. Try adjusting the crop area or selecting 'All Brand Libraries'.")
    else:
        st.info("No catalog index loaded. Use 'Sync Drive' or Tab 2 to index catalog files.")

# TAB 2: STANDALONE UPLOADER
with tab2:
    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem; margin-top:10px;'>⚡ Manual PDF Indexer</h4>", unsafe_allow_html=True)
    company_name = st.text_input("Brand / Manufacturer Name Tag:", value="General")
    uploaded_pdfs = st.file_uploader("Upload PDF Catalogs to Generate Embeddings", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs and st.button("Process & Update Vector Database", type="primary"):
        with st.spinner("Extracting catalog pages & generating visual embeddings..."):
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
