import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageOps
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

# --- RESPONSIVE & HIGH-PERFORMANCE UI STYLING ---
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

    /* Fluid App Container */
    .block-container { 
        padding-top: 0.8rem !important; 
        padding-bottom: 2rem !important; 
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 900px !important; 
    }

    /* Header Styling */
    .app-header {
        text-align: center;
        padding: 5px 0 10px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 12px;
    }
    .app-header-subtitle {
        color: #b8976c;
        font-size: clamp(0.65rem, 2vw, 0.75rem);
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .app-header-title {
        color: #0f172a;
        font-size: clamp(1.2rem, 4vw, 1.5rem);
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
        min-height: 42px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        width: 100% !important;
    }

    /* Cropper Canvas Container Bounds */
    canvas, .stCropper {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 8px !important;
    }

    /* Match Results Containers */
    .match-container-exact {
        background: #fcfbf9;
        border: 1px solid #e2d9cd;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .match-container-alt {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .match-header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
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
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 8px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .meta-item-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 0.8rem;
        color: #475569;
        word-break: break-word;
    }
    .meta-item-box strong {
        color: #0f172a;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 0.4rem !important;
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pdf_page_image_cached(file_id, page_num):
    """Downloads & converts PDF page to compressed JPEG with 24h cache for fast loading."""
    pdf_bytes = fetch_pdf_bytes_from_drive(file_id)
    if pdf_bytes:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=100)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            return img_byte_arr.getvalue()
        except Exception:
            pass
    return None


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
        img_bytes = fetch_pdf_page_image_cached(meta_dict["file_id"], meta_dict.get("page", 1))
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
            return

    cat_title = meta_dict.get('catalog', '')
    cat_page = meta_dict.get('page', 1)
    st.info(f"📍 **Match Reference:** {cat_title} — **Page {cat_page}**")


@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        return AIVectorEngine()
    except Exception as e:
        st.error(f"Engine connection failed: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_brands(_engine):
    try:
        return _engine.get_all_brands()
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_tab2_metrics(_engine):
    """Caches database count queries so app startup remains instant."""
    try:
        col_info = _engine.client.get_collection(COLLECTION_NAME)
        total_vectors = col_info.points_count
        
        scroll_res, _ = _engine.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=["company", "catalog"],
            with_vectors=False
        )
        
        all_brands = set()
        all_catalogs = set()
        for point in scroll_res:
            if point.payload:
                if "company" in point.payload:
                    all_brands.add(point.payload["company"])
                if "catalog" in point.payload:
                    all_catalogs.add(point.payload["catalog"])

        return len(all_catalogs), len(all_brands), total_vectors, sorted(list(all_catalogs))
    except Exception:
        return 0, 0, 0, []


def make_cropper_safe_image(pil_img, target_size=800):
    """
    Pads wide or tall images into a square 1:1 container.
    This prevents react-cropper from clipping horizontal edges on mobile screens.
    """
    w, h = pil_img.size
    max_dim = max(w, h)
    
    scale = target_size / float(max_dim)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas = Image.new("RGB", (target_size, target_size), (248, 250, 252))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized_img, (pad_x, pad_y))
    
    return canvas, pad_x, pad_y, scale


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
        companies = get_cached_brands(engine)
        companies_list = ["All Brand Libraries"] + companies
        
        col_filter, col_search_kw, col_sync = st.columns([2.5, 2, 1], vertical_alignment="bottom")
        with col_filter:
            selected_company = st.selectbox("Select Brand Collection:", companies_list)
        with col_search_kw:
            catalog_keyword = st.text_input("Filter Catalog Name (Optional):", placeholder="e.g. Marbelo, Louvers")
        with col_sync:
            if st.button("🔄 Sync Drive", use_container_width=True):
                gh_token = st.secrets.get("GITHUB_TOKEN")
                if gh_token:
                    with st.spinner("Triggering cloud sync..."):
                        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
                        headers = {
                            "Authorization": f"Bearer {gh_token}",
                            "Accept": "application/vnd.github.v3+json",
                            "User-Agent": "Streamlit-Catalog-App"
                        }
                        res = requests.post(url, json={"event_type": "drive-updated"}, headers=headers)
                        if res.status_code == 204:
                            st.cache_resource.clear()
                            st.cache_data.clear()
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
                help="Filters out weak or unrelated results."
            )

        search_file = st.file_uploader("Upload or Capture Reference Image", type=["jpg", "png", "jpeg"])
        
        if search_file:
            raw_pil_img = Image.open(io.BytesIO(search_file.getvalue())).convert("RGB")
            raw_pil_img = ImageOps.exif_transpose(raw_pil_img)
            
            # Format image into square container (guarantees 100% horizontal & vertical fit on phones)
            canvas_img, pad_x, pad_y, scale = make_cropper_safe_image(raw_pil_img, target_size=800)
            
            st.markdown("<p style='font-weight:600; color:#334155; font-size:0.85rem; margin-top:8px; margin-bottom:4px;'>1. Adjust Crop Area over Pattern / Product:</p>", unsafe_allow_html=True)
            
            crop_box = st_cropper(
                canvas_img, 
                realtime_update=True, 
                box_color='#b8976c', 
                aspect_ratio=None,
                return_type='box'
            )
            
            trigger_search = st.button("🔍 Search Cropped Pattern", type="primary", use_container_width=True)
            
            if trigger_search or "last_search_executed" in st_session_state_wrapper():
                st_session_state_wrapper()["last_search_executed"] = True
                
                orig_w, orig_h = raw_pil_img.size
                
                if crop_box:
                    # Reverse map canvas crop box coordinates back to original high-res image
                    res_left = crop_box['left'] - pad_x
                    res_top = crop_box['top'] - pad_y
                    res_w = crop_box['width']
                    res_h = crop_box['height']
                    
                    orig_left = int(res_left / scale)
                    orig_top = int(res_top / scale)
                    orig_right = int((res_left + res_w) / scale)
                    orig_bottom = int((res_top + res_h) / scale)
                    
                    # Clamp boundaries safely
                    orig_left = max(0, min(orig_w, orig_left))
                    orig_top = max(0, min(orig_h, orig_top))
                    orig_right = max(0, min(orig_w, orig_right))
                    orig_bottom = max(0, min(orig_h, orig_bottom))
                    
                    if orig_right > orig_left and orig_bottom > orig_top:
                        high_res_crop = raw_pil_img.crop((orig_left, orig_top, orig_right, orig_bottom))
                    else:
                        high_res_crop = raw_pil_img
                else:
                    high_res_crop = raw_pil_img

                # Upsample small crops for optimal neural vector extraction
                if high_res_crop.width < 224 or high_res_crop.height < 224:
                    proc_img = high_res_crop.resize((448, 448), Image.Resampling.BICUBIC)
                else:
                    proc_img = high_res_crop

                with st.spinner("Searching neural database..."):
                    query_vector = engine.get_single_embedding(proc_img)
                    confidence_threshold = min_confidence_slider / 100.0
                    
                    try:
                        matches = engine.search(
                            query_vector=query_vector, 
                            top_k=25, 
                            min_confidence=confidence_threshold,
                            brand_filter=selected_company,
                            keyword_filter=catalog_keyword
                        )
                    except TypeError:
                        matches = engine.search(
                            query_vector=query_vector, 
                            top_k=25, 
                            min_confidence=confidence_threshold
                        )
                    
                    exact_matches = [m for m in matches if m["score"] >= 0.50]
                    high_confidence_matches = [m for m in matches if confidence_threshold <= m["score"] < 0.50]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if exact_matches:
                    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem;'>🎯 Exact Match Results</h4>", unsafe_allow_html=True)
                    for i, res in enumerate(exact_matches[:3]):
                        score_pct = res["score"] * 100
                        b_name = res['meta'].get('company', 'General')
                        c_name = res['meta'].get('catalog', 'N/A')
                        p_num = res['meta'].get('page', 1)
                        st.markdown(f"""
                        <div class="match-container-exact">
                            <div class="match-header-tag tag-exact">
                                <span>Direct Match #{i+1}</span> • <span>{score_pct:.1f}% Confidence</span>
                            </div>
                            <div class="meta-details-grid">
                                <div class="meta-item-box">🏢 <strong>Brand:</strong> {b_name}</div>
                                <div class="meta-item-box">📖 <strong>Catalog:</strong> {c_name}</div>
                                <div class="meta-item-box">📄 <strong>Location:</strong> Page {p_num}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        render_match_image(res["meta"])
                        st.divider()
                        
                elif high_confidence_matches:
                    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem;'>🎨 High Confidence Alternatives</h4>", unsafe_allow_html=True)
                    for i, res in enumerate(high_confidence_matches[:3]):
                        score_pct = res["score"] * 100
                        b_name = res['meta'].get('company', 'General')
                        c_name = res['meta'].get('catalog', 'N/A')
                        p_num = res['meta'].get('page', 1)
                        st.markdown(f"""
                        <div class="match-container-alt">
                            <div class="match-header-tag tag-alt">
                                <span>Candidate #{i+1}</span> • <span>{score_pct:.1f}% Visual Similarity</span>
                            </div>
                            <div class="meta-details-grid">
                                <div class="meta-item-box">🏢 <strong>Brand:</strong> {b_name}</div>
                                <div class="meta-item-box">📖 <strong>Catalog:</strong> {c_name}</div>
                                <div class="meta-item-box">📄 <strong>Location:</strong> Page {p_num}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        render_match_image(res["meta"])
                        st.divider()
                else:
                    st.info(f"ℹ️ **No matching product found in current catalogs above {min_confidence_slider}% confidence.**")

with tab2:
    st.markdown("<h4 style='color:#0f172a; font-weight:700; font-size:1.05rem; margin-top:10px;'>⚡ Cloud Index & Fast PDF Lookup</h4>", unsafe_allow_html=True)
    
    if engine is not None:
        total_catalogs, total_brands, total_vectors, catalog_list = get_tab2_metrics(engine)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="PDF Catalogs Indexed", value=total_catalogs)
        with m_col2:
            st.metric(label="Unique Brands", value=total_brands)
        with m_col3:
            st.metric(label="Searchable Vector Patches", value=f"{total_vectors:,}")
            
    st.divider()

    st.markdown("##### 🔎 Quick Catalog Keyword Search")
    quick_kw = st.text_input("Search catalog name or brand directly:", placeholder="e.g. Louvers, Marbelo, Euro Pratik", key="quick_search_kw")
    
    if quick_kw:
        if engine is not None:
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
                    c_title = meta.get('catalog', '')
                    b_title = meta.get('company', '')
                    p_no = meta.get('page', 1)
                    st.write(f"📖 Catalog: {c_title} | 🏢 Brand: {b_title} | 📄 Page: {p_no}")
            else:
                st.info(f"No catalog names matching '{quick_kw}' found in current index.")

    if 'catalog_list' in locals() and catalog_list:
        with st.expander(f"📄 View All {len(catalog_list)} Currently Indexed PDF Files"):
            for idx, cat_name in enumerate(catalog_list, 1):
                st.write(f"{idx}. {cat_name}")

    st.divider()

    st.markdown("##### 🔄 Sync Google Drive Catalogs")
    if st.button("🚀 Trigger Cloud PDF Indexing", type="primary", use_container_width=True):
        gh_token = st.secrets.get("GITHUB_TOKEN")
        if not gh_token:
            st.error("❌ Missing `GITHUB_TOKEN` in Streamlit secrets.")
        else:
            with st.spinner("Triggering GitHub Actions cloud indexing server..."):
                url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
                headers = {
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Streamlit-Catalog-App"
                }
                res = requests.post(url, json={"event_type": "drive-updated"}, headers=headers)
                
                if res.status_code == 204:
                    st.success("✅ Sync successfully triggered on GitHub Actions!")
                else:
                    st.error(f"❌ Failed to trigger GitHub workflow: {res.status_code}")
