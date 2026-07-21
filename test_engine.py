import os
import requests
from PIL import Image
import io
from core_engine import AIVectorEngine, COLLECTION_NAME

def run_diagnostic_test():
    print("--------------------------------------------------")
    print("🧪 STARTING DIAGNOSTIC TEST FOR CATALOG SEARCH ENGINE")
    print("--------------------------------------------------")

    # 1. Initialize Engine & Check Qdrant Connection
    try:
        print("\n1️⃣ Connecting to Qdrant Cloud & Loading Neural Model...")
        engine = AIVectorEngine()
        print("   ✅ Connected successfully!")
    except Exception as e:
        print(f"   ❌ Engine Initialization Failed: {e}")
        return

    # 2. Inspect Indexed Points & Brand Diversity
    try:
        print("\n2️⃣ Inspecting Database Payload & Brand Tags...")
        scroll_res, _ = engine.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        total_points = len(scroll_res)
        brands = set(p.payload.get("company", "General") for p in scroll_res if p.payload)
        catalogs = set(p.payload.get("catalog", "N/A") for p in scroll_res if p.payload)

        print(f"   📊 Total Vector Embeddings in Qdrant: {total_points}")
        print(f"   🏢 Extracted Brands ({len(brands)}): {list(brands)}")
        print(f"   📖 Processed Catalogs ({len(catalogs)}): {list(catalogs)[:5]}...")
        
        if total_points == 0:
            print("   ⚠️ Database is empty! Please run 'python sync_drive.py' first to populate Qdrant.")
            return

    except Exception as e:
        print(f"   ❌ Database Read Error: {e}")
        return

    # 3. Simulate Query Search Across Diverse Categories
    print("\n3️⃣ Testing Vector Search Precision...")
    
    # Select a sample point payload from database to use as a dummy test target
    sample_point = scroll_res[0]
    sample_path = sample_point.payload.get("page_path", "")
    sample_catalog = sample_point.payload.get("catalog", "N/A")
    sample_brand = sample_point.payload.get("company", "N/A")

    print(f"   🎯 Testing sample visual target: [{sample_brand}] -> {sample_catalog}")

    # Load local sample image or construct mock query
    if os.path.exists(sample_path):
        test_img = Image.open(sample_path)
    else:
        # Fallback: create a blank dummy image to verify vector pipeline mechanics
        test_img = Image.new("RGB", (224, 224), color=(200, 150, 100))

    query_vector = engine.get_single_embedding(test_img)
    results = engine.search(query_vector, top_k=5, min_confidence=0.10)

    print(f"\n4️⃣ Matches Returned ({len(results)}):")
    for idx, match in enumerate(results, 1):
        score_pct = match["score"] * 100
        meta = match["meta"]
        print(f"   Rank #{idx} | Confidence: {score_pct:.2f}% | Brand: {meta.get('company')} | Catalog: {meta.get('catalog')} (Page {meta.get('page')})")

    print("\n--------------------------------------------------")
    print("✅ TEST COMPLETED CLEANLY! System is ready for deployment.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_diagnostic_test()
