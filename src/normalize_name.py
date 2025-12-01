import re


def normalize_name(name):
    """
    Normalize county/city names from CSV or VTK.
    Converts names to a comparable form.
    """
    name = name.lower().strip()

    # Remove common suffixes
    name = re.sub(r"\scounty$", "", name)
    name = re.sub(r"\scity$", "", name)

    # Handle known Virginia special cases
    special_cases = {
        "charles city": "charles city",
        "king and queen": "king and queen",
        "james city": "james city",
        "manassas park": "manassas park",
    }

    for key, val in special_cases.items():
        if name.startswith(key):
            return val

    return name
