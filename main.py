"""
main.py — Elwood Clothing full scraper entry point.

Pipeline:
  1. Fetch all products from Elwood's Shopify JSON API (paginated, no browser).
  2. Transform each product into a Supabase-ready record.
  3. Generate embeddings only for new/changed products:
     - New products: generate both image and text embeddings
     - Changed image_url: regenerate both embeddings
     - Unchanged products: skip embeddings entirely
  4. Batch upsert to Supabase (50 per batch).
  5. Mark products as seen and clean up stale products.

Usage:
  python main.py
"""

import logging
import time
from datetime import datetime, timezone

from tqdm import tqdm

from config import MODEL_NAME
from db import SupabaseClient, BATCH_SIZE
from embedder import SigLIPEmbedder
from processor import transform_product
from scraper import fetch_all_products

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  Elwood Scraper  —  started at %s", start_time.isoformat())
    logger.info("=" * 60)

    embedder = SigLIPEmbedder(MODEL_NAME)
    db = SupabaseClient()

    products = fetch_all_products()

    if not products:
        logger.error("No products fetched. Aborting.")
        return

    total = len(products)
    logger.info(f"Processing {total} products …\n")

    new_count = 0
    updated_count = 0
    unchanged_count = 0
    error_count = 0

    records_to_upsert = []
    product_urls = []
    
    transformed_products = []
    for product in products:
        title = product.get("title", "UNKNOWN")
        try:
            record, info_text = transform_product(product)
            transformed_products.append((record, info_text, title))
            product_urls.append(record.get("product_url", ""))
        except Exception as exc:
            logger.error(f"  ✗ Transform failed: {title} — {exc}")
            error_count += 1

    logger.info(f"Fetching existing products from database for change detection …")
    existing_products = db.get_existing_products(product_urls)
    logger.info(f"Found {len(existing_products)} existing products.\n")

    for record, info_text, title in tqdm(transformed_products, desc="Products", unit="item"):
        product_url = record.get("product_url", "")
        
        try:
            image_url = record.get("image_url", "")
            
            if not image_url:
                logger.warning(f"  ⚠ No image_url for '{title}' — skipping.")
                unchanged_count += 1
                continue

            existing = existing_products.get(product_url)
            
            needs_embedding = False
            is_new = existing is None
            
            if is_new:
                needs_embedding = True
                logger.info(f"  [{title}] New product — generating embeddings …")
            elif existing.get("image_url") != image_url:
                needs_embedding = True
                logger.info(f"  [{title}] Image URL changed — regenerating embeddings …")
            else:
                existing_record = {k: existing.get(k) for k in record.keys() if k not in ("image_embedding", "info_embedding", "updated_at", "created_at")}
                new_record = {k: record.get(k) for k in record.keys() if k not in ("image_embedding", "info_embedding", "updated_at", "created_at")}
                
                if existing_record == new_record:
                    unchanged_count += 1
                    logger.info(f"  [{title}] Unchanged — skipping.")
                    continue
                else:
                    logger.info(f"  [{title}] Product data changed — updating without new embeddings.")

            if needs_embedding:
                logger.info(f"  [{title}] Embedding image …")
                image_emb = embedder.embed_image(image_url)
                if image_emb:
                    record["image_embedding"] = image_emb
                else:
                    logger.warning(f"  ⚠ Image embedding failed for '{title}'")

                if info_text:
                    logger.info(f"  [{title}] Embedding info text ({len(info_text)} chars) …")
                    time.sleep(0.5)
                    info_emb = embedder.embed_text(info_text)
                    if info_emb:
                        record["info_embedding"] = info_emb
                    else:
                        logger.warning(f"  ⚠ Info text embedding failed for '{title}'")

            records_to_upsert.append(record)
            
            if is_new:
                new_count += 1
            else:
                updated_count += 1

        except Exception as exc:
            error_count += 1
            logger.error(f"  ✗ FAILED: {title} — {exc}", exc_info=True)

    batches = [records_to_upsert[i:i + BATCH_SIZE] for i in range(0, len(records_to_upsert), BATCH_SIZE)]
    
    successful_batch_count = 0
    failed_records = []
    
    for batch in tqdm(batches, desc="DB Batches", unit="batch"):
        success, failed = db.batch_upsert(batch)
        if failed:
            failed_records.extend(failed)
            logger.error(f"Batch failed with {len(failed)} records")
        else:
            successful_batch_count += 1
            for rec in success:
                logger.debug(f"  DB upserted: {rec.get('id')} — {rec.get('title')}")

    db.mark_products_seen(product_urls)
    
    stale_urls = db.get_stale_product_urls()
    stale_count = 0
    if stale_urls:
        logger.info(f"Removing {len(stale_urls)} stale products (not seen for 2+ runs) …")
        stale_count = db.delete_products(list(stale_urls))
        logger.info(f"Deleted {stale_count} stale products.")

    if failed_records:
        logger.error(f"Logging {len(failed_records)} failed records to failed_products.log")
        with open("failed_products.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Run: {datetime.now(timezone.utc).isoformat()} ---\n")
            for rec in failed_records:
                f.write(f"{rec.get('id')} | {rec.get('title')} | {rec.get('product_url')}\n")

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  ✓ New products    : {new_count}")
    logger.info(f"  ↻ Products updated: {updated_count}")
    logger.info(f"  ⊘ Unchanged (skipped): {unchanged_count}")
    logger.info(f"  ✗ Errors          : {error_count}")
    logger.info(f"  🗑 Stale deleted  : {stale_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
