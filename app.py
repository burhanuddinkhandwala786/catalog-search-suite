import os
import io
import pickle
import faiss
import fitz  # PyMuPDF
from PIL import Image
import gdown
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from core_engine import AIVectorEngine

PAGE_DIR = "catalog_pages"
PDF_DIR = "pdf_catalogs"
INDEX_FILE = "faiss_catalog.index"
META_FILE = "catalog_meta.pkl"

def run_auto_sync():
    os.makedirs(PAGE_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    
    engine = AIVectorEngine()
    
    # 1. Fetch files from Google Drive / Service Account
    # (Your existing drive fetching logic here)
    
    # 2. When processing each PDF:
    pages_to_embed = []
    new_metadata = []
    
    # Iterate over downloaded PDFs
    for pdf_filename in os.listdir(PDF_DIR):
        if not pdf_filename.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        doc = fitz.open(pdf_path)
        safe_name = pdf_filename.replace(" ", "_")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=130)
            
            # Save image using CLEAN RELATIVE PATH
            img_filename = f"{safe_name}_page_{page_num+1}.jpg"
            rel_image_path = os.path.join(PAGE_DIR, img_filename)
            pix.save(rel_image_path)
            
            pil_img = Image.open(rel_image_path).convert("RGB")
            pages_to_embed.append(pil_img)
            new_metadata.append({
                "page_path": rel_image_path,  # Stores 'catalog_pages/filename.jpg'
                "page": page_num + 1,
                "catalog": pdf_filename,
                "company": "Viva" if "HPL" in pdf_filename else "Godrej"
            })

    if pages_to_embed:
        embeddings = engine.get_batch_embeddings(pages_to_embed, batch_size=16)
        engine.create_index(embeddings, new_metadata)
        
        faiss.write_index(engine.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(new_metadata, f)
            
        print("✅ Re-indexed all catalog pages from scratch successfully!")
        return True
    return False

if __name__ == "__main__":
    run_auto_sync()
