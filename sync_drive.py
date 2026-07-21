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


def cleanup_deleted_catalogs(drive_files, existing_meta, engine):
    """Removes local images and metadata for PDFs deleted from Google Drive."""
    active_pdf_names = set(f["name"] for f in drive_files) | set(f["name"].replace(" ", "_") for f in drive_files)
    
    updated_meta = []
    deleted_catalogs = set()

    for m in existing_meta:
        catalog_name = m.get("catalog", "")
        safe_catalog_name = catalog_name.replace(" ", "_")

        if catalog_name in active_pdf_names or safe_catalog_name in active_pdf_names:
            updated_meta.append(m)
        else:
            deleted_catalogs.add(catalog_name)
            # Delete corresponding extracted image from catalog_pages/
            img_path = m.get("page_path", "")
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"⚠️ Could not delete image '{img_path}': {e}")

    if deleted_catalogs:
        print(f"🗑️ Detected {len(deleted_catalogs)} deleted PDF(s) from Drive: {list(deleted_catalogs)}")
        print("🔄 Rebuilding FAISS index for active catalogs to stay under storage limits...")
        
        # Re-embed remaining active images to ensure clean vector index alignment
        if updated_meta:
            images = []
            valid_meta = []
            for m in updated_meta:
                p_path = m.get("page_path", "")
                if os.path.exists(p_path):
                    try:
                        images.append(Image.open(p_path).convert("RGB"))
                        valid_meta.append(m)
                    except Exception:
                        pass

            if images:
                new_embeddings = engine.get_batch_embeddings(images, batch_size=16)
                engine.create_index(new_embeddings, valid_meta)
                faiss.write_index(engine.index, INDEX_FILE)
                with open(META_FILE, "wb") as f:
                    pickle.dump(valid_meta, f)
                print(f"✅ Clean index rebuilt with {len(valid_meta)} active pages.")
                return valid_meta, True
        else:
            # If all PDFs were deleted from Drive
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            if os.path.exists(META_FILE):
                os.remove(META_FILE)
            print("🧹 All catalogs deleted from Drive. Cleared index completely.")
            return [], True

    return existing_meta, False


def download_unindexed_pdfs(service, drive_files, indexed_catalogs):
    """Only downloads PDFs that are NOT already in the index."""
    ensure_directories()

    if not drive_files:
        print("⚠️ 0 PDF files returned by Drive API.")
        return []

    new_files = []
    for f in drive_files:
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

    # Connect to Drive
    service = get_drive_service()
    if not service:
        raise RuntimeError("Drive Service authentication failed.")

    engine = AIVectorEngine()
    drive_files = fetch_all_pdfs(service)

    # Clean up any PDFs deleted from Drive before processing new ones
    existing_meta, cleaned_up = cleanup_deleted_catalogs(drive_files, existing_meta, engine)
    existing_catalogs = set(m.get("catalog", "") for m in existing_meta)

    # Download new PDFs
    new_pdf_names = download_unindexed_pdfs(service, drive_files, existing_catalogs)

    # Check for any local unindexed PDFs sitting in pdf_catalogs/
    local_pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    unindexed_local_pdfs = [f for f in local_pdfs if f not in existing_catalogs and f.replace(" ", "_") not in existing_catalogs]

    all_new_pdfs = list(set(new_pdf_names + unindexed_local_pdfs))

    if not all_new_pdfs:
        print("✅ No new PDF catalogs detected! All files are up to date.")
        return True

    print(f"🚀 Processing {len(all_new_pdfs)} NEW catalog PDF file(s)...")

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

        if os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
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


def run_auto_sync():
    try:
        return extract_and_index_incremental()
    except Exception as e:
        print(f"❌ Incremental Sync Failed: {e}")
        return False


if __name__ == "__main__":
    extract_and_index_incremental()
