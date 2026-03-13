"""
config.py — Central configuration for the Elwood scraper.
All secrets are loaded from .env (or environment variables).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY: str = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ."
    "XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
TABLE_NAME: str = "products"

# ── Shopify / Elwood ──────────────────────────────────────────────────────────
COLLECTION_HANDLE: str = "all-fashion-capsules"
COLLECTION_URL: str = (
    f"https://elwoodclothing.com/collections/{COLLECTION_HANDLE}/products.json"
)
BASE_URL: str = "https://elwoodclothing.com"
PRODUCTS_PER_PAGE: int = 250  # Shopify max

# ── Brand metadata ────────────────────────────────────────────────────────────
SOURCE: str = "scraper-elwood"
BRAND: str = "Elwood"
COUNTRY: str = "US"

# ── SigLIP embedding model ────────────────────────────────────────────────────
MODEL_NAME: str = "google/siglip-base-patch16-384"
EMBEDDING_DIM: int = 768
SIGLIP_MAX_TEXT_LENGTH: int = 64   # SigLIP tokenizer hard limit

# ── HTTP ──────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT: int = 30
REQUEST_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
