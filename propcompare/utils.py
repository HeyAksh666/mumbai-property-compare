def fmt_sale(lakhs: float) -> str:
    return f"₹{lakhs / 100:.2f} Cr" if lakhs >= 100 else f"₹{lakhs:.1f} L"


def fmt_rent(rs: float) -> str:
    return f"₹{rs:,.0f}/mo"


def fmt(listing_type: str, v: float) -> str:
    return fmt_sale(v) if listing_type == "Sale" else fmt_rent(v)


def per_sqft(listing_type: str, v: float, area: float) -> float:
    """Sale: Rs per sq ft (v in lakhs). Rent: Rs per sq ft per month."""
    return v * 100000 / area if listing_type == "Sale" else v / area
