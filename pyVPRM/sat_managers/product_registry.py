# product_registry.py
# Description: Registry of supported MODIS/VIIRS products for the LAADS downloader.
#              Add new products here without touching the downloader logic.
#
# cadence options:
#   "daily"  → one file per day      (e.g. MCD19A1, MOD09GA)
#   "8day"   → 8-day composites      (e.g. MOD09A1, MYD09A1)
#   "16day"  → 16-day composites     (e.g. MOD13A1)
#
# file_ext: extension to look for in directory listing
# notes:    human-readable description (optional, for documentation)

PRODUCT_REGISTRY = {
    # ── MAIAC (daily) ────────────────────────────────────────────────
    "MCD19A1": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "MAIAC Land Surface BRF and Albedo",
    },
    "MCD19A2": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "MAIAC Aerosol Optical Depth",
    },
    "MCD43A4": {
        "cadence":  "daily",        #  cambiar de 16day a daily
        "file_ext": ".hdf",
        "notes":    "MODIS BRDF-Adjusted Reflectance Daily 500m (nadir + SZA=45°) Collection 061",
    },
    "MCD43C4": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "MODIS BRDF-Adjusted Reflectance Daily 0.05deg CMG global (nadir + SZA=45°)",
        "global":   True,   # ← sin tiles
    },

    # ── Surface Reflectance daily ─────────────────────────────────────
    "MOD09GA": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "Terra Surface Reflectance Daily L2G",
    },
    "MYD09GA": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "Aqua Surface Reflectance Daily L2G",
    },

    # ── Aerosol daily ─────────────────────────────────────────────────
    "MOD04_L2": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "Terra Aerosol 5-Min L2 Swath",
    },
    "MYD04_L2": {
        "cadence":  "daily",
        "file_ext": ".hdf",
        "notes":    "Aqua Aerosol 5-Min L2 Swath",
    },

    # ── Surface Reflectance 8-day ─────────────────────────────────────
    "MOD09A1": {
        "cadence":  "8day",
        "file_ext": ".hdf",
        "notes":    "Terra Surface Reflectance 8-Day L3 500m",
    },
    "MYD09A1": {
        "cadence":  "8day",
        "file_ext": ".hdf",
        "notes":    "Aqua Surface Reflectance 8-Day L3 500m",
    },

    # ── Vegetation Index 16-day ───────────────────────────────────────
    "MOD13A1": {
        "cadence":  "16day",
        "file_ext": ".hdf",
        "notes":    "Terra Vegetation Indices 16-Day L3 500m",
    },
    "MYD13A1": {
        "cadence":  "16day",
        "file_ext": ".hdf",
        "notes":    "Aqua Vegetation Indices 16-Day L3 500m",
    },
    "MOD13A2": {
        "cadence":  "16day",
        "file_ext": ".hdf",
        "notes":    "Terra Vegetation Indices 16-Day L3 1km",
    },

    # ── Land Cover yearly ─────────────────────────────────────────────
    "MCD12Q1": {
        "cadence":  "yearly",
        "file_ext": ".hdf",
        "notes":    "MODIS Land Cover Type Yearly L3",
    },
}


def get_product_info(product_name: str) -> dict:
    """
    Return registry entry for product_name, or a safe default.

    Parameters
    ----------
    product_name : str
        Short product name, e.g. 'MCD19A1'.

    Returns
    -------
    dict with keys: cadence, file_ext, notes
    """
    if product_name in PRODUCT_REGISTRY:
        return PRODUCT_REGISTRY[product_name]

    # Unknown product → warn and assume 8-day (conservative default)
    import warnings
    warnings.warn(
        f"Product '{product_name}' not found in PRODUCT_REGISTRY. "
        f"Assuming 8-day cadence. Add it to product_registry.py if needed.",
        UserWarning,
        stacklevel=2,
    )
    return {"cadence": "8day", "file_ext": ".hdf", "notes": "unknown"}


def list_products() -> None:
    """Print a human-readable table of all registered products."""
    print(f"{'Product':<12} {'Cadence':<8} {'Description'}")
    print("-" * 55)
    for name, info in PRODUCT_REGISTRY.items():
        print(f"{name:<12} {info['cadence']:<8} {info['notes']}")