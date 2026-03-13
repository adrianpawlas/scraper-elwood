"""
main.py — Elwood Clothing full scraper entry point.

Pipeline:
  1. Fetch all products from Elwood's Shopify JSON API (paginated, no browser).
  2. Transform each product into a Supabase-ready record.
  3. Generate a 768-dim SigLIP image embedding from the primary product image.
  4. Generate a 768-dim SigLIP text embedding from all product info.
  5. Upsert the record into the `products` table.

Usage:
  python main.py
"""

import logging
import time
from datetime import datetime, timezone

from tqdm import tqdm

from config import MODEL_NAME
from db import SupabaseClient
from embedder import SigLIPEmbedder
from processor import transform_product
from scraper import fetch_all_products

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  Elwood Scraper  —  started at %s", start_time.isoformat())
    logger.info("=" * 60)

    # ── 1. Load SigLIP model (once, shared for image + text) ─────────────────
    embedder = SigLIPEmbedder(MODEL_NAME)

    # ── 2. Connect to Supabase ────────────────────────────────────────────────
    db = SupabaseClient()

    # ── 3. Fetch all products from Shopify ───────────────────────────────────
    products = fetch_all_products()

    if not products:
        logger.error("No products fetched. Aborting.")
        return

    total = len(products)
    logger.info(f"Processing {total} products …\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    # ── 4. Process each product ───────────────────────────────────────────────
    for product in tqdm(products, desc="Products", unit="item"):
        title = product.get("title", "UNKNOWN")

        try:
            # 4a. Transform Shopify dict → Supabase record + info text
            record, info_text = transform_product(product)

            image_url = record.get("image_url", "")

            if not image_url:
                logger.warning(f"  ⚠ No image_url for '{title}' — skipping.")
                skip_count += 1
                continue

            # 4b. Image embedding
            logger.info(f"  [{title}] Embedding image …")
            image_emb = embedder.embed_image(image_url)
            if image_emb:
                record["image_embedding"] = image_emb
            else:
                logger.warning(f"  ⚠ Image embedding failed for '{title}'")

            # 4c. Info text embedding
            if info_text:
                logger.info(f"  [{title}] Embedding info text ({len(info_text)} chars) …")
                info_emb = embedder.embed_text(info_text)
                if info_emb:
                    record["info_embedding"] = info_emb
                else:
                    logger.warning(f"  ⚠ Info text embedding failed for '{title}'")

            # 4d. Upsert to Supabase
            db.upsert(record)
            success_count += 1
            logger.info(f"  ✓ Upserted: {record['id']} — {title}")

        except Exception as exc:
            error_count += 1
            logger.error(f"  ✗ FAILED: {title} — {exc}", exc_info=True)

        # Small polite delay between products
        time.sleep(0.05)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  ✓ Success : {success_count}")
    logger.info(f"  ⚠ Skipped : {skip_count}")
    logger.info(f"  ✗ Errors  : {error_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
