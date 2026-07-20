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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DARK ENTERPRISE CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

    /* Global Resets & Background Grid Overlay */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #07090f !important;
        color: #f1f5f9 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(34, 211, 238, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(212, 165, 116, 0.06) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
    }

    /* Hide Streamlit Native Chrome */
    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton { 
        display: none !important; 
    }

    /* Container Spacing */
    .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 2.5rem !important; 
        max-width: 1050px !important; 
    }

    /* Glass-Morphism Command Header */
    .command-header {
        background: rgba(11, 15, 24, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
    }
    .header-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        color: #22d3ee;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .header-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }

    /* Live Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 30px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-dot-online {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }
    .status-dot-offline {
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        box-shadow: 0 0 10px #ef4444;
    }

    /* Inputs & Selectbox Styling */
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.01em;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #0b0f18 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #22d3ee !important;
        box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.18) !important;
    }

    /* Dashed Upload Zone */
    [data-testid="stFileUploader"] section {
        background-color: #0b0f18 !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #22d3ee !important;
        background-color: rgba(34, 211, 238, 0.02) !important;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.1) !important;
    }

    /* Custom Pill-Style Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0b0f18;
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0 18px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(34, 211, 238, 0.12) !important;
        color: #22d3ee !important;
        border: 1px solid rgba(34, 211, 238, 0.3) !important;
    }

    /* Action Buttons Styling */
    .stButton>button {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        border-color: #22d3ee !important;
        color: #22d3ee !important;
        box-shadow: 0 0 16px rgba(34, 211, 238, 0.2) !important;
    }

    /* Process CTA Button (Gold Accent) */
    .cta-gold button {
        background: linear-gradient(135deg, #d4a574 0%, #b8860b 100%) !important;
        color: #07090f !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(212, 165, 116, 0.25) !important;
    }
    .cta-gold button:hover {
        box-shadow: 0 6px 24px rgba(212, 165, 116, 0.4) !important;
        transform: translateY(-1px) !important;
        color: #000000 !important;
    }

    /* Result Glass Cards */
    .card-exact {
        background: rgba(11, 15, 24, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-exact:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.4);
    }

    .card-alt {
        background: rgba(11, 15, 24, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #0284c7;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-alt:hover {
        transform: translateY(-2px);
        border-color: rgba(2, 132, 199, 0.4);
    }

    /* Confidence Progress Bar */
    .progress-bar-bg {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }
    .progress-bar-fill-exact {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        height: 100%;
        border-radius: 6px;
    }
    .progress-bar-fill-alt {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%);
        height: 100%;
        border-radius: 6px;
    }

    /* Monospace Grid Labels */
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 14px;
    }
    .meta-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .meta-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .meta-value {
        font-size: 0.88rem;
        color: #f1f5f9;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# RAM Caching for DINOv2 Vector Engine
@st.cache_resource(show_spinner="Loading Neural Vector Core...")
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

# --- GLASS COMMAND HEADER ---
total_pages = len(engine.metadata) if st.session_state.get("catalog_indexed", False) else 0

status_html = f"""
<div class="status-pill">
    <div class="status-dot-online"></div>
    <span style="color: #f8fafc;">Index Online</span>
    <span style="color: #64748b;">·</span>
    <span style="color: #22d3ee;">{total_pages} pages</span>
</div>
""" if st.session_state.get("catalog_indexed", False) else """
<div class="status-pill">
    <div class="status-dot-offline"></div>
    <span style="color: #ef4444;">Index Offline</span>
</div>
"""

st.markdown(f"""
<div class="command-header">
    <div>
        <div class="header-eyebrow">Enterprise Intelligence Suite</div>
        <div class="header-title">AI Catalog Visual Matcher</div>
    </div>
    {status_html}
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Visual Pattern Matcher", "⚡ Instant Indexer"])

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
            
            st.markdown("<p style='font-family:\"JetBrains Mono\", monospace; font-size:0.75rem; color:#22d3ee; letter-spacing:0.08em; text-transform:uppercase; margin-top:16px;'>Crop Target Texture Area:</p>", unsafe_allow_html=True)
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#22d3ee', 
                aspect_ratio=None
            )
            
            with st.spinner("Executing neural vector search..."):
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
                st.markdown("<p style='font-family:\"Space Grotesk\", sans-serif; font-size:1.1rem; color:#f8fafc; font-weight:700;'>🎯 Direct Matches</p>", unsafe_allow_html=True)
                for i, res in enumerate(exact_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="card-exact">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#10b981; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">EXACT MATCH #{i+1}</span>
                            <span style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:#f8fafc; font-weight:600;">{score_pct:.1f}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill-exact" style="width: {score_pct}%;"></div>
                        </div>
                        <div class="meta-grid">
                            <div class="meta-box">
                                <div class="meta-label">BRAND</div>
                                <div class="meta-value">{res['meta'].get('company', 'General')}</div>
                            </div>
                            <div class="meta-box">
                                <div class="meta-label">CATALOG</div>
                                <div class="meta-value">{res['meta']['catalog']}</div>
                            </div>
                            <div class="meta-box">
                                <div class="meta-label">LOCATION</div>
                                <div class="meta-value">Page {res['meta']['page']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(res["meta"]["page_path"]):
                        st.image(res["meta"]["page_path"], use_container_width=True)
                    st.divider()
                    
            elif alternative_matches:
                st.info("💡 **Exact match not found. Displaying closest matching alternatives:**")
                st.markdown("<p style='font-family:\"Space Grotesk\", sans-serif; font-size:1.1rem; color:#f8fafc; font-weight:700;'>🎨 Recommended Alternatives</p>", unsafe_allow_html=True)
                for i, res in enumerate(alternative_matches[:3]):
                    score_pct = res["score"] * 100
                    st.markdown(f"""
                    <div class="card-alt">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#38bdf8; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">ALTERNATIVE #{i+1}</span>
                            <span style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:#f8fafc; font-weight:600;">{score_pct:.1f}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill-alt" style="width: {score_pct}%;"></div>
                        </div>
                        <div class="meta-grid">
                            <div class="meta-box">
                                <div class="meta-label">BRAND</div>
                                <div class="meta-value">{res['meta'].get('company', 'General')}</div>
                            </div>
                            <div class="meta-box">
                                <div class="meta-label">CATALOG</div>
                                <div class="meta-value">{res['meta']['catalog']}</div>
                            </div>
                            <div class="meta-box">
                                <div class="meta-label">LOCATION</div>
                                <div class="meta-value">Page {res['meta']['page']}</div>
                            </div>
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
    st.markdown("<p style='font-family:\"Space Grotesk\", sans-serif; font-size:1.05rem; color:#f8fafc; font-weight:700; margin-top:10px;'>⚡ Direct PDF Catalog Indexer</p>", unsafe_allow_html=True)
    company_name = st.text_input("Brand / Manufacturer Tag:", value="General")
    uploaded_pdfs = st.file_uploader("Upload PDF Catalogs to Vectorize", type=["pdf"], accept_multiple_files=True)
    
    st.markdown('<div class="cta-gold">', unsafe_allow_html=True)
    if uploaded_pdfs and st.button("Process & Update Database", type="primary", use_container_width=True):
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
    st.markdown('</div>', unsafe_allow_html=True)
