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

# Google Drive API Scopes
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def ensure_directories():
    """Ensure local output directories exist."""
    os.makedirs(PAGE_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def get_drive_service():
    """Builds and returns the Google Drive API service using GitHub Secrets."""
    service_account_info = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    
    if not service_account_info:
        # Fallback to local credentials file if running locally
        cred_path = "credentials.json"
        if os.path.exists(cred_path):
            creds = Credentials.from_service_account_file(cred_path, scopes=SCOPES)
            return build("drive", "v3", credentials=creds)
        print("⚠️ No Google Drive credentials found in env or credentials.json.")
        return None

    try:
        info = json.loads(service_account_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ Failed to parse Service Account JSON: {e}")
        return None


def download_pdfs_from_drive():
    """Queries Google Drive for all PDF files and downloads new or missing ones."""
    ensure_directories()
    service = get_drive_service()
    if not service:
        print("⚠️ Skipping Drive download, scanning local 'pdf_catalogs/' folder instead.")
        return

    print("☁️ Querying Google Drive for PDF catalogs...")
    try:
        # Query all non-trashed PDF files accessible to the service account
        query = "mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            print("⚠️ No PDF files found in accessible Google Drive folders.")
            return

        print(f"📥 Found {len(files)} PDF(s) in Drive. Syncing...")

        for f in files:
            file_id = f["id"]
            file_name = f["name"].replace(" ", "_")
            local_pdf_path = os.path.join(PDF_DIR, file_name)

            # Download if file does not exist locally
            if not os.path.exists(local_pdf_path):
                print(f"⬇️ Downloading '{file_name}' from Drive...")
                request = service.files().get_media(fileId=file_id)
                with open(local_pdf_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                print(f"✅ Downloaded '{file_name}'")
            else:
                print(f"⚡ File '{file_name}' already downloaded locally.")

    except Exception as e:
        print(f"❌ Error downloading PDFs from Drive: {e}")


def extract_and_index_all():
    """Extracts all PDF pages in PDF_DIR, normalizes paths to relative strings,
    and rebuilds the FAISS index and metadata pickle file from scratch.
    """
    # Step 1: Sync all PDFs from Google Drive
    download_pdfs_from_drive()

    engine = AIVectorEngine()
    pages_to_embed = []
    new_metadata = []

    # Step 2: Check for PDFs in the local directory
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDF files found in '{PDF_DIR}/'.")
        return False

    print(f"🔄 Processing {len(pdf_files)} PDF catalog(s)...")

    for pdf_filename in sorted(pdf_files):
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        safe_name = pdf_filename.replace(" ", "_")

        # Determine brand tag from filename
        if "godrej" in pdf_filename.lower():
            company = "Godrej"
        elif "viva" in pdf_filename.lower() or "hpl" in pdf_filename.lower():
            company = "Viva"
        else:
            company = "General"

        try:
            doc = fitz.open(pdf_path)
            print(f"📖 Extracting '{pdf_filename}' ({len(doc)} pages)...")

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=130)

                # Save extracted page image with clean relative path
                img_filename = f"{safe_name}_page_{page_num + 1}.jpg"
                rel_image_path = os.path.join(PAGE_DIR, img_filename)
                pix.save(rel_image_path)

                # Load image for visual embedding generation
                pil_img = Image.open(rel_image_path).convert("RGB")
                pages_to_embed.append(pil_img)

                # Append clean metadata record with RELATIVE path
                new_metadata.append(
                    {
                        "page_path": rel_image_path,  # E.g., 'catalog_pages/file_page_1.jpg'
                        "page": page_num + 1,
                        "catalog": pdf_filename,
                        "company": company,
                    }
                )

        except Exception as e:
            print(f"❌ Error reading PDF '{pdf_filename}': {e}")

    # Step 3: Generate vector embeddings and persist index/metadata
    if pages_to_embed:
        print(f"⚡ Generating vector embeddings for {len(pages_to_embed)} total pages...")
        embeddings = engine.get_batch_embeddings(pages_to_embed, batch_size=16)

        # Create fresh index and metadata
        engine.create_index(embeddings, new_metadata)

        # Save index and metadata to disk
        faiss.write_index(engine.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(new_metadata, f)

        print(f"✅ Successfully built fresh index ({len(pages_to_embed)} total pages)!")
        return True

    return False


def run_auto_sync():
    """Main entry point for catalog sync pipeline."""
    try:
        return extract_and_index_all()
    except Exception as e:
        print(f"❌ Auto Sync Failed: {e}")
        return False


if __name__ == "__main__":
    run_auto_sync()
