"""
processor.py — Transforms a raw Shopify product dict into a Supabase-ready record.

Returns a tuple: (record_dict, info_text)
  • record_dict  – all columns for the `products` table
  • info_text    – concatenated text used to build info_embedding
"""

import json
import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from config import BASE_URL, BRAND, COUNTRY, SOURCE

logger = logging.getLogger(__name__)

# ── Shopify product_type → clean category name ────────────────────────────────
# Split on '&', '/', 'and', then title-case each token.
_SPLIT_RE = re.compile(r"\s*[&/]\s*|\s+and\s+", re.IGNORECASE)

# Internal Shopify / Elwood tags we don't want to expose in the DB
_SKIP_TAG_FRAGMENTS = {
    "trynow",
    "smarttag",
    "protected_",
    "automated",
    "active_products",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_category(product_type: str) -> str:
    """'TEES' → 'Tees', 'Sweaters & Hoodies' → 'Sweaters, Hoodies'."""
    if not product_type:
        return ""
    parts = _SPLIT_RE.split(product_type)
    return ", ".join(p.strip().title() for p in parts if p.strip())


def _extract_gender(tags: list[str]) -> str:
    """
    Determine gender from Shopify tags.
      FEMALE only          → 'woman'
      MALE only            → 'man'
      UNISEX / both / none → 'unisex'
    """
    upper = {t.upper() for t in tags}
    is_female = "FEMALE" in upper
    is_male = "MALE" in upper
    is_unisex = "UNISEX" in upper

    if is_female and not is_male:
        return "woman"
    if is_male and not is_female and not is_unisex:
        return "man"
    return "unisex"


def _extract_price_sale(variants: list[dict]) -> tuple[str, str | None]:
    """
    Elwood is USD-only on their Shopify JSON endpoint.

    Shopify convention:
      price            = current (possibly discounted) price
      compare_at_price = original price (set only when on sale)

    We map to:
      price column = original price (no-sale reference)
      sale  column = discounted price (only set when there is actually a sale)
    """
    if not variants:
        return "", None

    v = variants[0]
    current = str(v.get("price", "") or "").strip()
    compare = str(v.get("compare_at_price", "") or "").strip()

    if compare and compare not in ("0.00", "0", ""):
        # On sale: compare_at_price is the "original", price is the sale price
        price_str = f"{compare}USD"
        sale_str = f"{current}USD"
    else:
        price_str = f"{current}USD"
        sale_str = None

    return price_str, sale_str


def _extract_sizes(product: dict) -> str:
    """Return a comma-joined string of available sizes (e.g. 'XS, S, M, L, XL')."""
    for opt in product.get("options", []):
        if "size" in opt.get("name", "").lower():
            return ", ".join(str(v) for v in opt.get("values", []))

    # Fallback: infer from variant titles "COLOR / SIZE"
    sizes: list[str] = []
    for v in product.get("variants", []):
        parts = v.get("title", "").split("/")
        if len(parts) > 1:
            size = parts[-1].strip()
            if size and size not in sizes:
                sizes.append(size)
    return ", ".join(sizes)


def _extract_colors(product: dict) -> list[str]:
    """Return a list of unique color values."""
    for opt in product.get("options", []):
        if "color" in opt.get("name", "").lower():
            return [str(v) for v in opt.get("values", [])]

    # Fallback: first option values (Shopify color is usually option1)
    if product.get("options"):
        return [str(v) for v in product["options"][0].get("values", [])]
    return []


def _clean_tags(raw_tags) -> list[str]:
    """
    Accept either a list or a comma-separated tag string (Shopify sends both).
    Filters internal operational tags.
    """
    if isinstance(raw_tags, str):
        tag_list = [t.strip() for t in raw_tags.split(",")]
    elif isinstance(raw_tags, list):
        tag_list = list(raw_tags)
    else:
        return []

    result = []
    for tag in tag_list:
        if not tag:
            continue
        lower = tag.lower()
        if any(frag in lower for frag in _SKIP_TAG_FRAGMENTS):
            continue
        result.append(tag)
    return result


def _build_info_text(
    title: str,
    description: str,
    category: str,
    gender: str,
    colors: list[str],
    sizes: str,
    price: str,
    sale: str | None,
    tags: list[str],
) -> str:
    """Build a rich text string used for text embedding."""
    parts = [
        f"Brand: {BRAND}",
        f"Title: {title}",
        f"Category: {category}" if category else "",
        f"Gender: {gender}",
        f"Description: {description}" if description else "",
        f"Colors: {', '.join(colors)}" if colors else "",
        f"Sizes: {sizes}" if sizes else "",
        f"Price: {price}" if price else "",
        f"Sale Price: {sale}" if sale else "",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    return " | ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Main transform
# ─────────────────────────────────────────────────────────────────────────────

def transform_product(product: dict) -> tuple[dict, str]:
    """
    Convert a Shopify product dict to a Supabase `products` row.

    Returns:
        record   – dict ready for upsert (no embedding vectors yet)
        info_text – full-text string for info_embedding generation
    """
    shopify_id: int = product["id"]
    handle: str = product["handle"]
    title: str = product.get("title", "").strip()
    product_url: str = f"{BASE_URL}/products/{handle}"

    # ── Description ──────────────────────────────────────────────────────────
    description = _clean_html(product.get("body_html", ""))

    # ── Images ───────────────────────────────────────────────────────────────
    images: list[dict] = product.get("images", [])
    image_url: str = images[0]["src"] if images else ""
    additional_images: str = (
        " , ".join(img["src"] for img in images[1:]) if len(images) > 1 else ""
    )

    # ── Variants / options ───────────────────────────────────────────────────
    variants: list[dict] = product.get("variants", [])
    price, sale = _extract_price_sale(variants)
    sizes = _extract_sizes(product)
    colors = _extract_colors(product)

    # ── Taxonomy ─────────────────────────────────────────────────────────────
    raw_tags = product.get("tags", [])
    tags = _clean_tags(raw_tags)
    category = _extract_category(product.get("product_type", ""))
    gender = _extract_gender(raw_tags if isinstance(raw_tags, list) else raw_tags.split(","))

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    skus = list({v.get("sku", "") for v in variants if v.get("sku")})
    available = any(v.get("available", False) for v in variants)
    metadata = json.dumps(
        {
            "shopify_id": shopify_id,
            "description": description,
            "colors": colors,
            "sizes": sizes.split(", ") if sizes else [],
            "price": price,
            "sale_price": sale,
            "product_type": product.get("product_type", ""),
            "tags": tags,
            "vendor": product.get("vendor", BRAND),
            "skus": skus,
            "available": available,
            "options": [
                {"name": o["name"], "values": o["values"]}
                for o in product.get("options", [])
            ],
            "total_variants": len(variants),
            "shopify_created_at": product.get("created_at", ""),
            "shopify_updated_at": product.get("updated_at", ""),
        },
        ensure_ascii=False,
    )

    # ── Info text (for embedding) ─────────────────────────────────────────────
    info_text = _build_info_text(
        title=title,
        description=description,
        category=category,
        gender=gender,
        colors=colors,
        sizes=sizes,
        price=price,
        sale=sale,
        tags=tags,
    )

    # ── Record ────────────────────────────────────────────────────────────────
    record: dict = {
        "id": f"elwood-{shopify_id}",
        "source": SOURCE,
        "brand": BRAND,
        "title": title,
        "description": description or None,
        "category": category or None,
        "gender": gender,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "additional_images": additional_images or None,
        "price": price or None,
        "sale": sale,
        "size": sizes or None,
        "tags": tags if tags else None,
        "metadata": metadata,
        "second_hand": False,
        "country": COUNTRY,
        "other": None,
        "compressed_image_url": None,
        # image_embedding and info_embedding added in main.py
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return record, info_text
