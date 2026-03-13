"""
scraper.py — Fetches all products from Elwood's Shopify JSON API.

Elwood runs on Shopify, which exposes a public products.json endpoint.
We paginate using ?limit=250&page=N until we receive an empty result set.
No browser automation needed.
"""

import logging
import time

import requests

from config import (
    COLLECTION_URL,
    PRODUCTS_PER_PAGE,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def fetch_all_products(
    collection_url: str = COLLECTION_URL,
    limit: int = PRODUCTS_PER_PAGE,
    delay: float = 0.5,
) -> list[dict]:
    """
    Return every product in the given Shopify collection.

    Pagination: Shopify accepts ?limit=250&page=N.
    Stops when an empty products array is returned.
    """
    all_products: list[dict] = []
    page = 1

    logger.info(f"Starting paginated fetch from: {collection_url}")

    while True:
        url = f"{collection_url}?limit={limit}&page={page}"
        logger.info(f"  Fetching page {page}: {url}")

        try:
            resp = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error(f"  HTTP error on page {page}: {exc}")
            break
        except ValueError as exc:
            logger.error(f"  JSON decode error on page {page}: {exc}")
            break

        products = data.get("products", [])
        if not products:
            logger.info(f"  No products on page {page}. Pagination complete.")
            break

        all_products.extend(products)
        logger.info(f"  Got {len(products)} products (total so far: {len(all_products)})")

        if len(products) < limit:
            # Last page — fewer results than limit means we're done
            break

        page += 1
        time.sleep(delay)  # polite delay between pages

    logger.info(f"Total products fetched: {len(all_products)}")
    return all_products
