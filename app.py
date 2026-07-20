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

# --- UI STYLING ---
st.set_page_config(
    page_title="Catalog Search Engine | Burhanuddin Khandwala",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 16px 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 24px;
    }
    .brand-title { color: #f8fafc; font-size: 1.4rem; font-weight: 700; margin: 0; }
    .brand-subtitle { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }
    .match-card-exact { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 18px; margin-bottom: 12px; }
    .match-card-alt { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 18px; margin-bottom: 12px; }
    .badge-exact { background-color: #15803d; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: inline-block; margin-bottom: 8px; }
    .badge-alt { background-color: #0369a1; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: inline-block; margin-bottom: 8px; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 10px; font-size: 0.95rem; color: #1e293b; }
    .meta-item { background: rgba(255, 255, 255, 0.7); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(0, 0, 0, 0.05); }
</style>
""", unsafe_allow_html=True)

# RAM Caching for DINOv2 Vector Engine
@st.cache_resource(show_spinner="Initializing Neural Search Core...")
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

st.markdown("""
<div class="brand-header">
    <div class="brand-title">📦 Enterprise Catalog Visual Search Engine</div>
    <div class="brand-subtitle">Integrated Intelligence System • Burhanuddin Khandwala</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Visual Pattern Matcher", "⚡ On-The-Fly PDF Search"])

# TAB 1: PRODUCTION SEARCH
with tab1:
    if st.session_state.get("catalog_indexed", False):
        companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
        companies.insert(0, "All Companies")
        
        col_filter, col_sync = st.columns([3, 1])
        with col_filter:
            selected_company = st.selectbox("Filter Brand Catalog:", companies)
        with col_sync:
            st.write("")
            if st.button("🔄 Sync Google Drive", use_container_width=True):
                with st.spinner("Checking Drive updates..."):
                    if run_auto_sync():
                        st.cache_resource.clear()
                        st.success("Database synchronized!")
                        st.rerun()
                    else:
                        st.info("Catalogs are up to date.")

        search_file = st.file_uploader("Upload or Snap Reference Photo", type=["jpg", "png", "jpeg"])
        
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown("#### ✂️ Crop Material Pattern / Texture Area:")
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#00FF00', 
                aspect_ratio=None
            )
            
            q_emb = engine.get_single_embedding(cropped_img)
            raw_matches = engine.search(q_emb, top_k=15, min_confidence=0.50)
            
            filtered_matches = [
                m for m in raw_matches 
                if selected_company == "All Companies" or m["meta"].get("company", "General") == selected_company
            ]
            
            exact_matches = [m for m in filtered_matches if m["score"] >= 0.75]
            alternative_matches = [m for m in filtered_matches if 0.50 <= m["score"] < 0.75]
            
            st.markdown("---")
            
            if exact_matches:
                st.markdown("### 🎯 Direct Catalog Matches")
                for i, res in enumerate(exact_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="match-card-exact">
                        <span class="badge-exact">EXACT MATCH #{i+1} — {score_pct:.1f}% ACCURACY</span>
                        <div class="meta-grid">
                            <div class="meta-item">🏢 <b>Brand:</b> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-item">📖 <b>Catalog File:</b> {res['meta']['catalog']}</div>
                            <div class="meta-item">📄 <b>Page Number:</b> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
                    
            elif alternative_matches:
                st.info("💡 **No exact product match found in catalog. Showing closest similar designs:**")
                st.markdown("### 🎨 Similar Catalog Alternatives")
                for i, res in enumerate(alternative_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="match-card-alt">
                        <span class="badge-alt">RECOMMENDED ALTERNATIVE #{i+1} — {score_pct:.1f}% SIMILARITY</span>
                        <div class="meta-grid">
                            <div class="meta-item">🏢 <b>Brand:</b> {res['meta'].get('company', 'General')}</div>
                            <div class="meta-item">📖 <b>Catalog File:</b> {res['meta']['catalog']}</div>
                            <div class="meta-item">📄 <b>Page Number:</b> Page {res['meta']['page']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
            else:
                st.warning(f"❌ No matching materials found under '{selected_company}'. Adjust crop area or choose 'All Companies'.")
    else:
        st.info("No catalogs indexed yet. Click 'Sync Google Drive' or use Tab 2 for quick uploads.")

# TAB 2: STANDALONE UPLOADER
with tab2:
    st.markdown("### ⚡ Quick On-The-Fly PDF Processing")
    company_name = st.text_input("Brand Tag:", value="General")
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
                st.success(f"Indexed {len(pages_to_embed)} pages successfully!")
                st.rerun()