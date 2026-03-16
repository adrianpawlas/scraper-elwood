"""
db.py — Supabase client with batch upsert and stale product management.

pgvector columns (image_embedding, info_embedding) are serialised as the
bracket-string format expected by PostgREST: "[0.1,0.2,...]"
"""

import logging
import time
from datetime import datetime, timezone
from supabase import Client, create_client

from config import SUPABASE_KEY, SUPABASE_URL, TABLE_NAME, SOURCE

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_RETRIES = 3


def _format_vector(v: list[float]) -> str:
    """Convert a Python float list to the pgvector wire format '[x,y,z,…]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _records_equal(existing: dict, new: dict) -> bool:
    """Compare key fields to determine if product has changed."""
    fields = [
        "title", "description", "category", "gender", "product_url",
        "image_url", "additional_images", "price", "sale", "size",
        "tags", "metadata", "second_hand", "country"
    ]
    for field in fields:
        existing_val = existing.get(field)
        new_val = new.get(field)
        if existing_val != new_val:
            return False
    return True


class SupabaseClient:
    """Thin wrapper around supabase-py with batch upsert and stale product helpers."""

    def __init__(self) -> None:
        logger.info(f"Connecting to Supabase: {SUPABASE_URL}")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client ready.")

    def get_existing_products(self, product_urls: list[str]) -> dict[str, dict]:
        """Fetch existing products by product_url for change detection."""
        if not product_urls:
            return {}
        
        result = (
            self.client.table(TABLE_NAME)
            .select("*")
            .in_("product_url", product_urls)
            .eq("source", SOURCE)
            .execute()
        )
        
        return {p["product_url"]: p for p in result.data if p.get("product_url")}

    def batch_upsert(self, records: list[dict], on_conflict_keys: list[str] | None = None) -> tuple[list[dict], list[dict]]:
        """
        Insert or update multiple product rows in a single request.
        Returns tuple of (success_records, failed_records).
        """
        if not records:
            return [], []
        
        if on_conflict_keys is None:
            on_conflict_keys = ["id"]
        
        rows = []
        for record in records:
            row = dict(record)
            for col in ("image_embedding", "info_embedding"):
                val = row.get(col)
                if isinstance(val, list):
                    row[col] = _format_vector(val)
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

        conflict_cols = ",".join(on_conflict_keys)
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = (
                    self.client.table(TABLE_NAME)
                    .upsert(rows, on_conflict=conflict_cols)
                    .execute()
                )
                if result.data:
                    return rows, []
                else:
                    logger.warning(f"Batch upsert returned no data, attempt {attempt}")
                    
            except Exception as exc:
                logger.warning(f"[Batch upsert attempt {attempt}/{MAX_RETRIES}] Failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
        
        logger.error(f"Batch upsert failed after {MAX_RETRIES} attempts for {len(rows)} records")
        return [], rows

    def upsert(self, record: dict) -> None:
        """Single record upsert for backward compatibility."""
        rows, failed = self.batch_upsert([record])
        if failed:
            raise Exception(f"Failed to upsert record: {record.get('id')}")

    def get_stale_product_urls(self) -> set[str]:
        """Get product URLs that haven't been seen in the last 2 consecutive runs."""
        result = (
            self.client.table(TABLE_NAME)
            .select("product_url, stale_count")
            .eq("source", SOURCE)
            .execute()
        )
        
        stale = set()
        for p in result.data:
            url = p.get("product_url")
            stale_count = p.get("stale_count", 0) or 0
            
            if stale_count >= 2:
                stale.add(url)
        
        return stale

    def mark_products_seen(self, product_urls: list[str]) -> None:
        """Mark products as seen in current run (reset stale_count)."""
        if not product_urls:
            return
        
        existing = (
            self.client.table(TABLE_NAME)
            .select("product_url")
            .in_("product_url", product_urls)
            .eq("source", SOURCE)
            .execute()
        )
        
        seen_urls = {p["product_url"] for p in existing.data}
        
        reset_records = []
        increment_records = []
        
        all_urls = set(product_urls)
        
        for url in seen_urls:
            reset_records.append({
                "product_url": url,
                "source": SOURCE,
                "stale_count": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        
        for url in all_urls - seen_urls:
            increment_records.append({
                "product_url": url,
                "source": SOURCE,
                "stale_count": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        
        if reset_records:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    for rec in reset_records:
                        (
                            self.client.table(TABLE_NAME)
                            .upsert(rec, on_conflict="product_url")
                            .execute()
                        )
                    break
                except Exception as exc:
                    logger.warning(f"[Mark seen reset attempt {attempt}] Failed: {exc}")
                    if attempt == MAX_RETRIES:
                        logger.error(f"Failed to reset stale_count for seen products")
        
        if increment_records:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    for rec in increment_records:
                        (
                            self.client.table(TABLE_NAME)
                            .upsert(rec, on_conflict="product_url")
                            .execute()
                        )
                    break
                except Exception as exc:
                    logger.warning(f"[Mark seen increment attempt {attempt}] Failed: {exc}")
                    if attempt == MAX_RETRIES:
                        logger.error(f"Failed to increment stale_count for new products")

    def delete_products(self, product_urls: list[str]) -> int:
        """Delete products by product_url. Returns count of deleted products."""
        if not product_urls:
            return 0
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = (
                    self.client.table(TABLE_NAME)
                    .delete()
                    .in_("product_url", product_urls)
                    .eq("source", SOURCE)
                    .execute()
                )
                return len(result.data) if result.data else 0
            except Exception as exc:
                logger.warning(f"[Delete attempt {attempt}/{MAX_RETRIES}] Failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
        
        logger.error(f"Failed to delete {len(product_urls)} stale products after {MAX_RETRIES} attempts")
        return 0
