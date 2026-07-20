import os
import json
import pickle
import faiss
import fitz  # PyMuPDF
from PIL import Image
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
    """Ensure local storage folders exist."""
    os.makedirs(PAGE_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def get_drive_service():
    """Builds and returns the Google Drive API service."""
    service_account_info = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not service_account_info:
        cred_path = "credentials.json"
        if os.path.exists(cred_path):
            try:
                creds = Credentials.from_service_account_file(cred_path, scopes=SCOPES)
                return build("drive", "v3", credentials=creds)
            except Exception as e:
                print(f"⚠️ Failed to load local credentials.json: {e}")
                return None
        print("⚠️ No credentials found in environment or credentials.json.")
        return None

    try:
        info = json.loads(service_account_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ Failed to parse Service Account JSON: {e}")
        return None


def download_pdfs_from_drive():
    """Syncs PDF files from Google Drive if service is available."""
    ensure_directories()
    service = get_drive_service()
    if not service:
        print("⚠️ Skipping Drive download step. Using local files in 'pdf_catalogs/'.")
        return

    print("☁️ Querying Google Drive for catalog PDFs...")
    try:
        response = service.files().list(
            q="mimeType='application/pdf' and trashed=false",
            fields="files(id, name)"
        ).execute()

        files = response.get("files", [])

        if not files:
            print("⚠️ No PDF files returned by Google Drive.")
            return

        print(f"📥 Found {len(files)} PDF(s) in Drive. Syncing...")

        for f in files:
            file_id = f["id"]
            safe_filename = f["name"].replace(" ", "_")
            local_pdf_path = os.path.join(PDF_DIR, safe_filename)

            if not os.path.exists(local_pdf_path):
                print(f"⬇️ Downloading '{safe_filename}'...")
                request = service.files().get_media(fileId=file_id)
                with open(local_pdf_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                print(f"✅ Downloaded '{safe_filename}'")
            else:
                print(f"⚡ File cached: '{safe_filename}'")

    except Exception as e:
        print(f"❌ Drive sync error: {e}")


def extract_and_index_all():
    """Processes all local PDFs in pdf_catalogs/, extracts pages, and creates the index."""
    ensure_directories()
    
    # Try downloading from Drive first
    download_pdfs_from_drive()

    engine = AIVectorEngine()
    pages_to_embed = []
    new_metadata = []

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDFs found in '{PDF_DIR}/'. Place PDF files in '{PDF_DIR}/' or connect Drive.")
        return False

    print(f"🔄 Processing {len(pdf_files)} PDF catalog(s)...")

    for pdf_filename in sorted(pdf_files):
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
            print(f"❌ Error rendering '{pdf_filename}': {e}")

    if pages_to_embed:
        print(f"⚡ Generating embeddings for {len(pages_to_embed)} total pages...")
        embeddings = engine.get_batch_embeddings(pages_to_embed, batch_size=16)

        engine.create_index(embeddings, new_metadata)

        faiss.write_index(engine.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(new_metadata, f)

        print(f"✅ Indexing complete! Created '{INDEX_FILE}' and '{META_FILE}'.")
        return True

    return False


def run_auto_sync():
    """Wrapper function required by app.py."""
    try:
        return extract_and_index_all()
    except Exception as e:
        print(f"❌ Auto Sync Pipeline Error: {e}")
        return False


if __name__ == "__main__":
    run_auto_sync()
