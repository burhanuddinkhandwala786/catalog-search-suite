import os
import json
import pickle
import faiss
import fitz  # PyMuPDF
from PIL import Image
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from core_engine import AIVectorEngine

PAGE_DIR = "catalog_pages"
PDF_DIR = "pdf_catalogs"
INDEX_FILE = "faiss_catalog.index"
META_FILE = "catalog_meta.pkl"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def ensure_directories():
    os.makedirs(PAGE_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def fetch_pdf_bytes_from_drive(file_id):
    """Downloads raw PDF bytes from public export or Drive link."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return res.content
    except Exception:
        pass
    return None


def get_drive_service():
    service_account_info = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not service_account_info:
        cred_path = "credentials.json"
        if os.path.exists(cred_path):
            try:
                creds = Credentials.from_service_account_file(cred_path, scopes=SCOPES)
                return build("drive", "v3", credentials=creds)
            except Exception as e:
                print(f"❌ Failed to load credentials.json: {e}")
                return None
        print("❌ No GOOGLE_SERVICE_ACCOUNT_JSON env var or credentials.json found.")
        return None

    try:
        info = json.loads(service_account_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ Failed to parse Service Account JSON: {e}")
        return None


def fetch_all_pdfs(service):
    """Recursively fetches all PDFs across Drive root and subfolders."""
    all_files = []
    page_token = None
    query = "mimeType='application/pdf' and trashed=false"

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        files = response.get("files", [])
        all_files.extend(files)

        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    return all_files


def download_unindexed_pdfs(service, indexed_catalogs):
    """Only downloads PDFs that are NOT already in the index."""
    ensure_directories()
    files = fetch_all_pdfs(service)

    if not files:
        print("⚠️ 0 PDF files returned by Drive API.")
        return []

    new_files = []
    for f in files:
        safe_filename = f["name"].replace(" ", "_")
        
        # Check if catalog is already processed
        if f["name"] in indexed_catalogs or safe_filename in indexed_catalogs:
            print(f"⚡ Skipping already indexed catalog: '{safe_filename}'")
            continue

        file_id = f["id"]
        local_pdf_path = os.path.join(PDF_DIR, safe_filename)

        print(f"⬇️ New Catalog Detected! Downloading '{safe_filename}'...")
        request = service.files().get_media(fileId=file_id)
        with open(local_pdf_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        
        new_files.append(safe_filename)

    return new_files


def extract_and_index_incremental():
    ensure_directories()
    
    # Load existing metadata & FAISS index if present
    existing_meta = []
    existing_catalogs = set()
    index = None

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "rb") as f:
                existing_meta = pickle.load(f)
                existing_catalogs = set(m.get("catalog", "") for m in existing_meta)
            print(f"📦 Loaded existing index metadata ({len(existing_meta)} pages across {len(existing_catalogs)} catalogs).")
        except Exception as e:
            print(f"⚠️ Failed to load existing metadata, starting fresh: {e}")

    if os.path.exists(INDEX_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
        except Exception as e:
            print(f"⚠️ Failed to load existing index: {e}")

    # Connect to Drive and download ONLY new PDFs
    service = get_drive_service()
    if not service:
        raise RuntimeError("Drive Service authentication failed.")

    new_pdf_names = download_unindexed_pdfs(service, existing_catalogs)

    # Check for any local unindexed PDFs sitting in pdf_catalogs/
    local_pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    unindexed_local_pdfs = [f for f in local_pdfs if f not in existing_catalogs and f.replace(" ", "_") not in existing_catalogs]

    all_new_pdfs = list(set(new_pdf_names + unindexed_local_pdfs))

    if not all_new_pdfs:
        print("✅ No new PDF catalogs detected! All files are already indexed. Exiting fast.")
        return True

    print(f"🚀 Processing {len(all_new_pdfs)} NEW catalog PDF file(s)...")

    engine = AIVectorEngine()
    pages_to_embed = []
    new_metadata = []

    for pdf_filename in sorted(all_new_pdfs):
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        safe_name = pdf_filename.replace(" ", "_")

        company = "Godrej" if "godrej" in pdf_filename.lower() else ("Viva" if "viva" in pdf_filename.lower() or "hpl" in pdf_filename.lower() else "General")

        try:
            doc = fitz.open(pdf_path)
            print(f"📖 Extracting '{pdf_filename}' ({len(doc)} pages)...")

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=130)

                img_filename = f"{safe_name}_page_{page_num + 1}.jpg"
                rel_image_path = os.path.join(PAGE_DIR, img_filename)
                pix.save(rel_image_path)

                pil_img = Image.open(rel_image_path).convert("RGB")
                pages_to_embed.append(pil_img)

                new_metadata.append({
                    "page_path": rel_image_path,
                    "page": page_num + 1,
                    "catalog": pdf_filename,
                    "company": company
                })

        except Exception as e:
            print(f"❌ Error reading '{pdf_filename}': {e}")

    if pages_to_embed:
        print(f"⚡ Generating embeddings ONLY for {len(pages_to_embed)} new page(s)...")
        new_embeddings = engine.get_batch_embeddings(pages_to_embed, batch_size=16)

        if index is not None:
            print("➕ Appending new vectors into existing FAISS index...")
            index.add(new_embeddings)
            final_index = index
        else:
            engine.create_index(new_embeddings, new_metadata)
            final_index = engine.index

        combined_metadata = existing_meta + new_metadata

        faiss.write_index(final_index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(combined_metadata, f)

        print(f"✅ Incremental indexing complete! Added {len(pages_to_embed)} pages. Total pages in index: {len(combined_metadata)}.")
        return True

    return True


def run_auto_sync():
    try:
        return extract_and_index_incremental()
    except Exception as e:
        print(f"❌ Incremental Sync Failed: {e}")
        return False


if __name__ == "__main__":
    extract_and_index_incremental()
