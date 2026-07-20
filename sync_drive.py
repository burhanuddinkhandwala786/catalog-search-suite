import os
import pickle
import warnings
import subprocess
import faiss
import gdown
import fitz  # PyMuPDF
from PIL import Image
from core_engine import AIVectorEngine

warnings.filterwarnings("ignore")

COMPANY_FOLDER_IDS = {
    "Euro Pratik": "11G38ebefaIGrLf1tA6UitQ7DCdpMn3li",
    "Godrej": "1yOeTLCU5rDpttlzjQjo-ylumPv2U6frA",
    "Viva": "1FXjmmY15rcowASGOD-FxnUF2ET6IV5H3",
    "Zydex": "13auyeFy5sxjW4Ny6vni5nD91UUDD6bzf",
}

TEMP_PDF_DIR = "drive_pdfs"
PAGE_DIR = "catalog_pages"
INDEX_FILE = "faiss_catalog.index"
META_FILE = "catalog_meta.pkl"


def download_drive_folder(folder_id, target_dir):
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    rclone_path = os.path.join(os.getcwd(), "rclone.exe")

    if os.path.exists(rclone_path):
        try:
            rclone_cmd = [
                rclone_path, "copy",
                ":http:", target_dir,
                "--http-url", folder_url,
                "--quiet"
            ]
            res = subprocess.run(rclone_cmd, capture_output=True)
            if res.returncode == 0 and os.listdir(target_dir):
                return
        except Exception:
            pass

    try:
        gdown.download_folder(url=folder_url, output=target_dir, quiet=True, use_cookies=False)
    except Exception as e:
        print(f"  Note downloading [{target_dir}]: {e}")


def append_single_pdf_to_index(pdf_path, company_name, engine, index_file=INDEX_FILE, meta_file=META_FILE, page_dir=PAGE_DIR):
    """Processes only the newly added PDF and appends its vectors directly."""
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    safe_name = file_name.replace(" ", "_")

    new_pages = []
    new_meta = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=130)
        page_path = os.path.join(page_dir, f"{company_name}_{safe_name}_page_{page_num+1}.jpg")
        pix.save(page_path)

        pil_img = Image.open(page_path).convert("RGB")
        new_pages.append(pil_img)
        new_meta.append({
            "page_path": page_path,
            "page": page_num + 1,
            "catalog": file_name,
            "company": company_name
        })

    if new_pages:
        new_embeddings = engine.get_batch_embeddings(new_pages, batch_size=16)

        # Handle index creation vs update
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            index.add(new_embeddings)
        else:
            index = engine.create_index(new_embeddings, new_meta)

        # Handle metadata creation vs update
        existing_meta = []
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "rb") as f:
                    existing_meta = pickle.load(f)
            except Exception:
                existing_meta = []

        updated_meta = existing_meta + new_meta

        # Persist to disk
        faiss.write_index(index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(updated_meta, f)

        print(f"✅ Instantly added '{file_name}' ({len(new_pages)} pages) to live database.")


def run_auto_sync():
    os.makedirs(TEMP_PDF_DIR, exist_ok=True)
    os.makedirs(PAGE_DIR, exist_ok=True)

    existing_metadata = []
    indexed_catalogs = set()

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "rb") as f:
                existing_metadata = pickle.load(f)
                indexed_catalogs = set(m["catalog"] for m in existing_metadata)
        except Exception:
            existing_metadata = []

    print("⚡ Starting Automated Catalog Synchronization...")

    for company_name, folder_id in COMPANY_FOLDER_IDS.items():
        if folder_id:
            company_dir = os.path.join(TEMP_PDF_DIR, company_name)
            os.makedirs(company_dir, exist_ok=True)
            print(f"Checking updates for [{company_name}]...")
            download_drive_folder(folder_id, company_dir)

    engine = AIVectorEngine()
    processed_any = False

    for root, _, files in os.walk(TEMP_PDF_DIR):
        for file in files:
            if file.lower().endswith(".pdf") and file not in indexed_catalogs:
                pdf_path = os.path.join(root, file)
                company_name = os.path.basename(root)

                print(f"--> Extracting pages from new PDF [{company_name}]: {file}...")
                try:
                    append_single_pdf_to_index(
                        pdf_path=pdf_path,
                        company_name=company_name,
                        engine=engine,
                        index_file=INDEX_FILE,
                        meta_file=META_FILE,
                        page_dir=PAGE_DIR
                    )
                    processed_any = True
                except Exception as pdf_err:
                    print(f"  [Warning] Could not parse {file}: {pdf_err}")

    if processed_any:
        print("✅ Sync completed successfully! Database updated.")
        return True
    else:
        print("✅ All catalog files are up to date.")
        return False


if __name__ == "__main__":
    run_auto_sync()