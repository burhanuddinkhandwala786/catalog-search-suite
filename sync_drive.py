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
    """
    Recursively scans Google Drive including subfolders, shared drives, and shortcuts.
    """
    all_files = []
    page_token = None
    query = "mimeType='application/pdf' and trashed=false"

    while True:
        try:
            response = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageSize=1000
            ).execute()
            
            files = response.get("files", [])
            all_files.extend(files)
            
            page_token = response.get("nextPageToken", None)
            if not page_token:
                break
        except Exception as e:
            print(f"⚠️ Drive API listing warning: {e}")
            break

    print(f"📂 Total PDF files discovered in Google Drive: {len(all_files)}")
    return all_files


def extract_page_tiles(pil_img):
    """
    Extracts multi-scale region patches (Full Page + 4 Quadrants + Center Swatch)
    allowing fine-grained patch-level visual matching.
    """
    w, h = pil_img.size
    tiles = [pil_img]  # Full Page

    # Quadrant Crops
    half_w, half_h = w // 2, h // 2
    tiles.append(pil_img.crop((0, 0, half_w, half_h)))            # Top-Left
    tiles.append(pil_img.crop((half_w, 0, w, half_h)))           # Top-Right
    tiles.append(pil_img.crop((0, half_h, half_w, h)))           # Bottom-Left
    tiles.append(pil_img.crop((half_w, half_h, w, h)))          # Bottom-Right

    # Center Swatch Focus
    margin_w, margin_h = w // 4, h // 4
    tiles.append(pil_img.crop((margin_w, margin_h, w - margin_w, h - margin_h)))

    return tiles


def extract_and_index_qdrant():
    ensure_directories()
    service = get_drive_service()
    if not service:
        raise RuntimeError("Drive Service authentication failed.")

    engine = AIVectorEngine()
    drive_files = fetch_all_pdfs(service)

    # Fetch indexed catalogs from Qdrant and normalize names (lowercase & handled spaces)
    scroll_res, _ = engine.client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=["catalog"],
        with_vectors=False
    )
    
    indexed_catalogs = set()
    for point in scroll_res:
        if point.payload and "catalog" in point.payload:
            cat = point.payload["catalog"]
            indexed_catalogs.add(cat.lower().strip())
            indexed_catalogs.add(cat.replace(" ", "_").lower().strip())

    # Safely identify unindexed files
    new_drive_files = []
    for f in drive_files:
        raw_name = f["name"]
        norm_name = raw_name.replace(" ", "_").lower().strip()
        clean_name = raw_name.lower().strip()
        
        if norm_name not in indexed_catalogs and clean_name not in indexed_catalogs:
            new_drive_files.append(f)

    if not new_drive_files:
        print("✅ All catalogs up to date in Qdrant Cloud!")
        return True

    print(f"🚀 Found {len(new_drive_files)} NEW catalog(s) to index out of {len(drive_files)} total files.")

    for f in new_drive_files:
        pdf_filename = f["name"].replace(" ", "_")
        local_pdf_path = os.path.join(PDF_DIR, pdf_filename)
        brand = extract_brand_name(pdf_filename)

        print(f"⬇️ Downloading '{pdf_filename}'...")
        try:
            request = service.files().get_media(fileId=f["id"])
            with open(local_pdf_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except Exception as dl_err:
            print(f"❌ Download failed for '{pdf_filename}': {dl_err}")
            continue

        try:
            doc = fitz.open(local_pdf_path)
            print(f"📖 Processing '{pdf_filename}' -> Brand: [{brand}] ({len(doc)} pages)...")

            points_to_upsert = []

            # Start at page_num = 1 (Page 2) to skip cover pages
            for page_num in range(1, len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=130)

                img_filename = f"{pdf_filename}_page_{page_num + 1}.jpg"
                rel_image_path = os.path.join(PAGE_DIR, img_filename)
                pix.save(rel_image_path)

                full_page_img = Image.open(rel_image_path).convert("RGB")
                
                tiles = extract_page_tiles(full_page_img)
                tile_embeddings = engine.get_batch_embeddings(tiles, batch_size=len(tiles))

                for tile_idx, vector in enumerate(tile_embeddings):
                    points_to_upsert.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "page_path": rel_image_path,
                                "page": page_num + 1,
                                "catalog": pdf_filename,
                                "company": brand,
                                "file_id": f["id"],
                                "patch_type": "full" if tile_idx == 0 else f"patch_{tile_idx}"
                            }
                        )
                    )

                full_page_img.close()

            if points_to_upsert:
                print(f"⚡ Uploading {len(points_to_upsert)} patch vectors to Qdrant Cloud...")
                engine.upsert_points(points_to_upsert)

            print(f"✅ Finished indexing '{pdf_filename}'!")

        except Exception as e:
            print(f"❌ Error extracting '{pdf_filename}': {e}")
        finally:
            if os.path.exists(local_pdf_path):
                try:
                    os.remove(local_pdf_path)
                except Exception:
                    pass

    print("✅ Patch-level indexing complete!")
    return True


def run_auto_sync():
    try:
        return extract_and_index_qdrant()
    except Exception as e:
        print(f"❌ Qdrant Sync Failed: {e}")
        return False


if __name__ == "__main__":
    extract_and_index_qdrant()
