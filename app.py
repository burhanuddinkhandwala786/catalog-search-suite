import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from streamlit_cropper import st_cropper
import io
import os
import requests
import warnings
from core_engine import AIVectorEngine, COLLECTION_NAME
from sync_drive import run_auto_sync, fetch_pdf_bytes_from_drive

warnings.filterwarnings("ignore")

REPO_OWNER = "burhanuddinkhandwala786"
REPO_NAME = "catalog-search-suite"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Visual Catalog Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def st_session_state_wrapper():
    return st.session_state

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

    div[data-baseweb="select"], input {
        border-radius: 8px !important;
        border: 1.5px solid #cbd5e1 !important;
        background-color: #f8fafc !important;
    }

    .stButton>button {
        background-color: #b8976c !important;
        color: #ffffff !important;
        border: 1px solid #a38258 !important;
        border-radius: 8px !important;
        height: 42px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }

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
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdf_bytes_cached(file_id):
    return fetch_pdf_bytes_from_drive(file_id)


def render_match_image(meta_dict):
    raw_path = meta_dict.get("page_path", "")
    filename = os.path.basename(raw_path) if raw_path else ""

    local_img_path = os.path.join("catalog_pages", filename) if filename else ""
    if local_img_path and os.path.exists(local_img_path):
        st.image(local_img_path, use_container_width=True)
        return
    elif raw_path and os.path.exists(raw_path):
        st.image(raw_path, use_container_width=True)
        return

    if "file_id" in meta_dict and meta_dict["file_id"]:
        pdf_bytes = fetch_pdf_bytes_cached(meta_dict["file_id"])
        if pdf_bytes:
            try:
                page_num = meta_dict.get("page", 1) - 1
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = doc[page_num]
                
                # Render directly to PIL Image for 100% web container reliability
                pix = page.get_pixmap(dpi=110)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                st.image(img, use_container_width=True)
                return
            except Exception:
                pass

    st.info(f"📍 **Match Reference:** {meta_dict.get('catalog', '')} — **Page {meta_dict.get('page', 1)}**")


@st.cache_resource(show_spinner="Connecting to Visual Search Engine...")
def load_engine():
    try:
        return AIVectorEngine()
    except Exception as e:
        st.error(f"Engine connection failed: {e}")
        return None


engine = load_engine()

st.markdown("""
<div class="app-header">
    <div class="app-header-subtitle">INSTANT PATTERN RECOGNITION</div>
    <div class="app-header-title">AI Catalog Search Engine</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Visual Pattern Search", "⚙️ Index Management"])

with tab1:
    if engine is not None:
        try:
            companies = engine.get_all_brands()
        except Exception:
            companies = []
            
        companies.insert(0, "All Brand Libraries")
        
        col_filter, col_search_kw, col_sync = st.columns([2.5, 2, 1], vertical_alignment="bottom")
        with col_filter:
            selected_company = st.selectbox("Select Brand Collection:", companies)
        with col_search_kw:
            catalog_keyword = st.text_input("Filter Catalog Name (Optional):", placeholder="e.g. Marbelo, Louvers")
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
                        res = requests.post(url, json={"event_type": "drive-updated"}, headers=headers)
                        if res.status_code == 204:
                            st.cache_resource.clear()
                            st.success("Sync triggered!")
                        else:
                            st.error(f"Failed: {res.status_code}")

        with st.expander("⚙️ Search Sensitivity Controls", expanded=False):
            min_confidence_slider = st.slider(
                "Minimum Confidence Cutoff (%)",
                min_value=30,
                max_value=80,
                value=40,
                step=5,
                help="Filters out weak or unrelated results. High values ensure only true matching catalog pages are shown."
            )

        search_file = st.file_uploader("Upload or Capture Reference Image", type=["jpg", "png", "jpeg"])
        
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            
            st.markdown("<p style='font-weight:600; color:#334155; font-size:0.88rem; margin-top:16px;'>1. Adjust Crop Area over Pattern / Product:</p>", unsafe_allow_html=True)
            
            cropped_img = st_cropper(
                raw_pil_img, 
                realtime_update=True, 
                box_color='#b8976c', 
                aspect_ratio=None,
                return_type='image'
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            trigger_search = st.button("🔍 Search Cropped Pattern", type="primary", use_container_width=True)
            
            if trigger_search or "last_search_executed" in st_session_state_wrapper():
                st_session_state_wrapper()["last_search_executed"] = True
                
                # Upsample small crops for high-precision shape & texture detail
                if cropped_img.width < 224 or cropped_img.height < 224:
                    proc_img = cropped_img.resize((448, 448), Image.Resampling.BICUBIC)
                else:
                    proc_img = cropped_img

                with st.spinner("Searching neural database for visual matches..."):
                    query_vector = engine.get_single_embedding(proc_img)
                    confidence_threshold = min_confidence_slider / 100.0
                    
                    # Native Hybrid Search (Vector + Qdrant Payload Filter)
                    matches = engine.search(
                        query_vector=query_vector, 
                        top_k=25, 
                        min_confidence=confidence_threshold,
                        brand_filter=selected_company,
                        keyword_filter=catalog_keyword
                    )
                    
                    exact_matches = [m for m in matches if m["score"] >= 0.50]
                    high_confidence_matches = [m for m in matches if confidence_threshold <= m["score"] < 0.50]
                
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
                                <div class="meta-item-box">📖 <strong>Catalog:</strong> {res['meta'].get('catalog', 'N/A')}</div>
                                <div class="meta-item-box">📄 <strong>Location:</strong> Page {res['meta'].get('page', 1)}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        render_match_image(res["meta"])
                        st.divider()
                        
                elif high_confidence_matches:
                    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.1rem;'>🎨 High Confidence Alternatives</h4>", unsafe_allow_html=True)
                    for i, res in enumerate(high_confidence_matches[:3]):
                        score_pct = res["score"] * 100
                        st.markdown(f"""
                        <div class="match-container-alt">
                            <div class="match-header-tag tag-alt">
                                <span>Candidate #{i+1}</span> • <span>{score_pct:.1f}% Visual Similarity</span>
                            </div>
                            <div class="meta-details-grid">
                                <div class="meta-item-box">🏢 <strong>Brand:</strong> {res['meta'].get('company', 'General')}</div>
                                <div class="meta-item-box">📖 <strong>Catalog:</strong> {res['meta'].get('catalog', 'N/A')}</div>
                                <div class="meta-item-box">📄 <strong>Location:</strong> Page {res['meta'].get('page', 1)}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        render_match_image(res["meta"])
                        st.divider()
                else:
                    st.info(f"ℹ️ **No matching product found in current catalogs above {min_confidence_slider}% confidence.**\n\nIf this is a new product, please ensure its PDF catalog is uploaded to Google Drive and synced.")

with tab2:
    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem; margin-top:10px;'>⚡ Cloud Index & Fast PDF Lookup</h4>", unsafe_allow_html=True)
    
    # 1. LIVE DATABASE METRICS DISPLAY
    if engine is not None:
        try:
            col_info = engine.client.get_collection(COLLECTION_NAME)
            total_vectors = col_info.points_count
            brands_list = engine.get_all_brands()
            total_brands = len(brands_list)
        except Exception:
            total_vectors = 0
            total_brands = 0

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Total Indexed Brands", value=total_brands)
        with m_col2:
            st.metric(label="Searchable Vector Patches", value=f"{total_vectors:,}")
            
    st.divider()

    # 2. QUICK KEYWORD CATALOG SEARCH
    st.markdown("##### 🔎 Quick Catalog Keyword Search")
    quick_kw = st.text_input("Search catalog name or brand directly:", placeholder="e.g. Louvers, Marbelo, Euro Pratik", key="quick_search_kw")
    
    if quick_kw:
        if engine is not None:
            # Query Qdrant for matching text payload entries
            scroll_res, _ = engine.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter={"must": [{"key": "catalog", "match": {"text": quick_kw.strip()}}]},
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            if scroll_res:
                st.success(f"Found {len(scroll_res)} catalog reference matches for '{quick_kw}':")
                for point in scroll_res:
                    meta = point.payload
                    st.markdown(f"📖 **Catalog:** `{meta.get('catalog')}` | 🏢 **Brand:** `{meta.get('company')}` | 📄 **Page:** {meta.get('page')}")
            else:
                st.info(f"No catalog names matching '{quick_kw}' found in current index.")

    st.divider()

    # 3. GOOGLE DRIVE AUTO-SYNC TRIGGER WITH PROGRESS STATUS
    st.markdown("##### 🔄 Sync Google Drive Catalogs")
    if st.button("🚀 Trigger Cloud PDF Indexing", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Connecting to Google Drive...")
        
        with st.spinner("Processing new PDF catalogs into Qdrant Cloud..."):
            progress_bar.progress(30, text="Downloading new PDFs from Google Drive...")
            
            if run_auto_sync():
                progress_bar.progress(80, text="Updating neural vector embeddings...")
                st.cache_resource.clear()
                progress_bar.progress(100, text="Indexing complete!")
                st.success("✅ Database synchronized successfully! All new PDFs are live.")
                st.rerun()
            else:
                progress_bar.empty()
                st.error("❌ Sync failed. Please check Google Drive file permissions or GitHub Actions log.")
