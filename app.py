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
import requests
import base64
from core_engine import AIVectorEngine
from sync_drive import run_auto_sync, fetch_pdf_bytes_from_drive

warnings.filterwarnings("ignore")

# Prevent CPU bottlenecking on shared cloud hardware
torch.set_num_threads(4)

REPO_OWNER = "burhanuddinkhandwala786"
REPO_NAME = "catalog-search-suite"
RAW_GITHUB_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main"

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

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    [data-testid="stSidebar"], #MainMenu, footer, header, .stDeployButton { 
        display: none !important; 
    }

    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1000px !important; 
    }

    .app-header {
        text-align: center;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 20px;
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

    /* --- DROPDOWN (SELECTBOX) STYLING --- */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
        border: 1.5px solid #cbd5e1 !important;
        background-color: #f8fafc !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: #b8976c !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* --- LABELS --- */
    .stSelectbox label, .stFileUploader label {
        font-weight: 700 !important;
        color: #1e293b !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em;
        margin-bottom: 6px !important;
    }

    /* --- BUTTON STYLING --- */
    .stButton>button {
        background-color: #b8976c !important;
        color: #ffffff !important;
        border: 1px solid #a38258 !important;
        border-radius: 8px !important;
        height: 42px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 5px rgba(184, 151, 108, 0.25) !important;
    }
    .stButton>button:hover {
        background-color: #a38258 !important;
        border-color: #8c6d46 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(184, 151, 108, 0.35) !important;
    }

    /* --- FILE UPLOADER STYLING --- */
    [data-testid="stFileUploader"] {
        background-color: #f8fafc !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #b8976c !important;
    }

    /* --- MATCH CARDS & TAGS --- */
    .match-container-exact {
        background: #fcfbf9;
        border: 1px solid #e2d9cd;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .match-container-alt {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
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
</style>
""", unsafe_allow_html=True)


# Fetch latest commit SHA directly from GitHub API
def get_latest_github_commit():
    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/main"
        headers = {"User-Agent": "Streamlit-Catalog-App"}
        gh_token = st.secrets.get("GITHUB_TOKEN")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
            
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json().get("sha", "latest")
    except Exception:
        pass
    return "latest"


# Render match image directly from GitHub raw media URL
def render_match_image(meta_dict):
    raw_path = meta_dict.get("page_path", "")
    filename = os.path.basename(raw_path) if raw_path else ""
    latest_sha = get_latest_github_commit()
    
    if filename:
        github_img_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{latest_sha}/catalog_pages/{filename}"
        try:
            res = requests.get(github_img_url, timeout=10)
            if res.status_code == 200:
                st.image(res.content, width="stretch")
                return
        except Exception:
            pass

    # Local disk fallback
    local_img_path = os.path.join("catalog_pages", filename) if filename else ""
    if local_img_path and os.path.exists(local_img_path):
        st.image(local_img_path, width="stretch")
        return

    st.info(f"📍 **Match Reference:** {meta_dict.get('catalog', '')} — **Page {meta_dict.get('page', 1)}**")


# Engine Initialization
@st.cache_resource(show_spinner="Loading Visual Recognition Engine...")
def load_engine():
    return AIVectorEngine()


# Direct GitHub API index loader using the commit SHA as cache key
@st.cache_resource(show_spinner="Downloading latest database from GitHub...")
def load_remote_index_via_api(commit_sha):
    gh_token = st.secrets.get("GITHUB_TOKEN")
    headers = {"User-Agent": "Streamlit-Catalog-App"}
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
        
    index_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{commit_sha}/faiss_catalog.index"
    meta_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{commit_sha}/catalog_meta.pkl"

    try:
        idx_res = requests.get(index_url, headers=headers, timeout=30)
        meta_res = requests.get(meta_url, headers=headers, timeout=30)

        if idx_res.status_code == 200 and meta_res.status_code == 200:
            tmp_index_path = f"/tmp/faiss_{commit_sha[:7]}.index"
            with open(tmp_index_path, "wb") as f:
                f.write(idx_res.content)

            meta = pickle.loads(meta_res.content)
            idx = faiss.read_index(tmp_index_path)
            return idx, meta
    except Exception as e:
        st.error(f"Error downloading vector database: {e}")

    # Local disk fallback
    if os.path.exists("faiss_catalog.index") and os.path.exists("catalog_meta.pkl"):
        try:
            idx = faiss.read_index("faiss_catalog.index")
            with open("catalog_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            return idx, meta
        except Exception:
            pass

    return None, []


engine = load_engine()
latest_sha = get_latest_github_commit()
index, metadata = load_remote_index_via_api(latest_sha)

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
<div class="app-header">
    <div class="app-header-subtitle">INSTANT PATTERN RECOGNITION</div>
    <div class="app-header-title">AI Catalog Search Engine</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Visual Pattern Search", "⚙️ Index Management"])

# TAB 1: VISUAL SEARCH
with tab1:
    if st.session_state.get("catalog_indexed", False):
        companies = sorted(list(set(m.get("company", "General") for m in engine.metadata)))
        companies.insert(0, "All Brand Libraries")
        
        col_filter, col_sync = st.columns([3.5, 1], vertical_alignment="bottom")
        with col_filter:
            selected_company = st.selectbox("Select Brand Collection:", companies)
        with col_sync:
            if st.button("🔄 Sync Drive", use_container_width=True):
                gh_token = st.secrets.get("GITHUB_TOKEN")
                
                if gh_token:
                    with st.spinner("Triggering GitHub Actions cloud sync..."):
                        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
                        headers = {
                            "Authorization": f"Bearer {gh_token}",
                            "Accept": "application/vnd.github.v3+json",
                            "User-Agent": "Streamlit-Catalog-App"
                        }
                        data = {"event_type": "drive-updated"}
                        res = requests.post(url, json=data, headers=headers)
                        
                        if res.status_code == 204:
                            st.cache_resource.clear()
                            st.success("Sync triggered! New catalogs will load in ~1–2 minutes.")
                        else:
                            st.error(f"Failed to trigger sync: {res.status_code}")
                else:
                    with st.spinner("Syncing Google Drive catalogs..."):
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
                    render_match_image(res["meta"])
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
                    render_match_image(res["meta"])
                    st.divider()
            else:
                st.warning(f"❌ No matching pattern found under '{selected_company}'. Try adjusting crop or selecting 'All Brand Libraries'.")
    else:
        st.info("No catalog index loaded. Click 'Sync Drive' above or use Tab 2 to process PDFs dynamically.")
        if st.button("🚀 Initial Cloud Sync", type="primary"):
            with st.spinner("Processing PDF catalogs directly in web app..."):
                if run_auto_sync():
                    st.cache_resource.clear()
                    st.success("Indexing complete!")
                    st.rerun()

# TAB 2: MANUAL WEB UPLOADER
with tab2:
    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem; margin-top:10px;'>⚡ Cloud PDF Indexer</h4>", unsafe_allow_html=True)
    company_name = st.text_input("Brand / Manufacturer Name Tag:", value="General")
    uploaded_pdfs = st.file_uploader("Upload PDF Catalogs to Generate Embeddings", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs and st.button("Process & Update Vector Database", type="primary"):
        with st.spinner("Extracting catalog pages & generating visual embeddings in web app memory..."):
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
