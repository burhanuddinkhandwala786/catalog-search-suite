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


def extract_page_tiles(pil_img):
    """
    Generates sub-crop tiles (swatches) from a catalog page so textures/products
    are indexed individually alongside full-page views.
    """
    w, h = pil_img.size
    tiles = [pil_img]  # Full page image

    # 2x2 Grid Crop (4 tiles)
    half_w, half_h = w // 2, h // 2
    tiles.append(pil_img.crop((0, 0, half_w, half_h)))            # Top-Left
    tiles.append(pil_img.crop((half_w, 0, w, half_h)))           # Top-Right
    tiles.append(pil_img.crop((0, half_h, half_w, h)))           # Bottom-Left
    tiles.append(pil_img.crop((half_w, half_h, w, h)))          # Bottom-Right

    # Center Tile Crop (where swatches usually sit)
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

    # Fetch already indexed catalog names from Qdrant
    scroll_res, _ = engine.client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=["catalog"],
        with_vectors=False
    )
    indexed_catalogs = set(point.payload.get("catalog") for point in scroll_res if point.payload)

    new_drive_files = [
        f for f in drive_files 
        if f["name"] not in indexed_catalogs and f["name"].replace(" ", "_") not in indexed_catalogs
    ]

    if not new_drive_files:
        print("✅ All catalogs up to date in Qdrant Cloud!")
        return True

    print(f"🚀 Processing {len(new_drive_files)} NEW catalog(s) with Multi-Tile Indexing...")

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

            points_to_upsert = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=130)

                img_filename = f"{pdf_filename}_page_{page_num + 1}.jpg"
                rel_image_path = os.path.join(PAGE_DIR, img_filename)
                pix.save(rel_image_path)

                full_page_img = Image.open(rel_image_path).convert("RGB")
                
                # Extract sub-tiles (swatches/regions) for precision matching
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
                                "is_tile": tile_idx > 0
                            }
                        )
                    )

                full_page_img.close()

            if points_to_upsert:
                print(f"⚡ Uploading {len(points_to_upsert)} multi-tile vectors to Qdrant...")
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

    print("✅ Multi-tile indexing complete!")
    return True


def run_auto_sync():
    try:
        return extract_and_index_qdrant()
    except Exception as e:
        print(f"❌ Qdrant Sync Failed: {e}")
        return False


if __name__ == "__main__":
    extract_and_index_qdrant()
