import os
import json
import re
import fitz  # PyMuPDF
from PIL import Image
import requests
import uuid
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from qdrant_client.models import PointStruct
from core_engine import AIVectorEngine, COLLECTION_NAME

PAGE_DIR = "catalog_pages"
PDF_DIR = "pdf_catalogs"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def ensure_directories():
    os.makedirs(PAGE_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def extract_brand_name(pdf_filename):
    clean_name = os.path.splitext(pdf_filename)[0]
    clean_name = re.sub(r'[-_]', ' ', clean_name)
    ignore_words = {'catalog', 'catalogue', 'brochure', 'folder', 'mobile', 'pdf', 'v1', 'v2', 'v3', 'vol1', 'vol2', 'compressed', 'for', 'men', 'women'}
    words = [w for w in clean_name.split() if w.lower() not in ignore_words]
    if words:
        return " ".join(words[:2]).title()
    return "General"


def fetch_pdf_bytes_from_drive(file_id):
    """Fallback helper used by frontend to stream missing page images."""
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
                print(f"❌ Failed credentials.json: {e}")
                return None
        return None

    try:
        info = json.loads(service_account_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ Failed Service Account JSON: {e}")
        return None


def fetch_all_pdfs(service):
    all_files = []
    page_token = None
    query = "mimeType='application/pdf' and trashed=false"

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        all_files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break
    return all_files


def get_indexed_catalogs_from_qdrant(engine):
    """Safely retrieves all indexed catalog names using cursor-based pagination."""
    indexed_catalogs = set()
    next_page_offset = None

    while True:
        scroll_res, next_page_offset = engine.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=next_page_offset,
            with_payload=["catalog"],
            with_vectors=False
        )

        for point in scroll_res:
            if point.payload and "catalog" in point.payload:
                indexed_catalogs.add(point.payload["catalog"])

        if next_page_offset is None:
            break

    return indexed_catalogs


def extract_and_index_qdrant():
    ensure_directories()
    service = get_drive_service()
    if not service:
        raise RuntimeError("Drive Service authentication failed.")

    engine = AIVectorEngine()
    drive_files = fetch_all_pdfs(service)

    # Paginated retrieval of already indexed catalogs
    indexed_catalogs = get_indexed_catalogs_from_qdrant(engine)

    # Filter out already indexed files
    new_drive_files = [
        f for f in drive_files 
        if f["name"] not in indexed_catalogs and f["name"].replace(" ", "_") not in indexed_catalogs
    ]

    if not new_drive_files:
        print("✅ All catalogs up to date in Qdrant Cloud!")
        return True

    print(f"🚀 Processing {len(new_drive_files)} NEW catalog(s) for Qdrant Cloud...")

    for f in new_drive_files:
        pdf_filename = f["name"].replace(" ", "_")
        local_pdf_path = os.path.join(PDF_DIR, pdf_filename)
        brand = extract_brand_name(pdf_filename)

        print(f"⬇️ Downloading '{pdf_filename}'...")
        request = service.files().get_media(fileId=f["id"])
        with open(local_pdf_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        try:
            doc = fitz.open(local_pdf_path)
            print(f"📖 Processing '{pdf_filename}' -> Brand: [{brand}] ({len(doc)} pages)...")

            # Process in memory-safe batches of 16 pages
            BATCH_SIZE = 16
            for start_idx in range(0, len(doc), BATCH_SIZE):
                batch_doc = doc[start_idx : start_idx + BATCH_SIZE]
                pil_images = []
                payloads = []

                for page_offset, page in enumerate(batch_doc):
                    actual_page_num = start_idx + page_offset + 1
                    pix = page.get_pixmap(dpi=130)

                    img_filename = f"{pdf_filename}_page_{actual_page_num}.jpg"
                    rel_image_path = os.path.join(PAGE_DIR, img_filename)
                    pix.save(rel_image_path)

                    img = Image.open(rel_image_path).convert("RGB")
                    pil_images.append(img)
                    payloads.append({
                        "page_path": rel_image_path,
                        "page": actual_page_num,
                        "catalog": pdf_filename,
                        "company": brand,
                        "file_id": f["id"]
                    })

                if pil_images:
                    embeddings = engine.get_batch_embeddings(pil_images, batch_size=BATCH_SIZE)
                    points_to_upsert = [
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload=payload
                        )
                        for vector, payload in zip(embeddings, payloads)
                    ]
                    engine.upsert_points(points_to_upsert)

                    # Explicitly close image pointers to prevent RAM bloat
                    for img in pil_images:
                        img.close()

            print(f"✅ Finished indexing '{pdf_filename}'!")

        except Exception as e:
            print(f"❌ Error extracting '{pdf_filename}': {e}")
        finally:
            # Clean up raw local PDF after processing
            if os.path.exists(local_pdf_path):
                try:
                    os.remove(local_pdf_path)
                except Exception:
                    pass

    print("✅ All new catalogs processed and pushed to Qdrant Cloud!")
    return True


def run_auto_sync():
    try:
        return extract_and_index_qdrant()
    except Exception as e:
        print(f"❌ Qdrant Sync Failed: {e}")
        return False


if __name__ == "__main__":
    extract_and_index_qdrant()
