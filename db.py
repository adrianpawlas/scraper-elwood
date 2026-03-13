"""
db.py — Supabase client with upsert support for the `products` table.

pgvector columns (image_embedding, info_embedding) are serialised as the
bracket-string format expected by PostgREST: "[0.1,0.2,...]"
"""

import logging

from supabase import Client, create_client

from config import SUPABASE_KEY, SUPABASE_URL, TABLE_NAME

logger = logging.getLogger(__name__)


def _format_vector(v: list[float]) -> str:
    """Convert a Python float list to the pgvector wire format '[x,y,z,…]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class SupabaseClient:
    """Thin wrapper around supabase-py with upsert helpers."""

    def __init__(self) -> None:
        logger.info(f"Connecting to Supabase: {SUPABASE_URL}")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client ready.")

    def upsert(self, record: dict) -> None:
        """
        Insert or update a product row.
        Conflict resolution: primary key `id`.

        Vector columns are serialised to string before sending.
        """
        row = dict(record)  # shallow copy so we don't mutate caller's dict

        # Serialise vector columns for PostgREST / pgvector
        for col in ("image_embedding", "info_embedding"):
            val = row.get(col)
            if isinstance(val, list):
                row[col] = _format_vector(val)

        try:
            result = (
                self.client.table(TABLE_NAME)
                .upsert(row, on_conflict="id")
                .execute()
            )
            if result.data:
                logger.debug(f"  DB upserted id={row.get('id')}")
            else:
                logger.warning(
                    f"  DB upsert returned no data for id={row.get('id')}"
                )
        except Exception as exc:
            logger.error(
                f"  DB upsert FAILED for id={row.get('id')} | title={row.get('title')}: {exc}"
            )
            raise
