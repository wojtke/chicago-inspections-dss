"""Polish wording for the app: plural forms, the city's facility types, readable names."""
import re

FACILITY_PL = {
    "Restaurant": "restauracja", "Grocery Store": "sklep spożywczy", "School": "szkoła",
    "Children's Services Facility": "placówka opieki nad dziećmi", "Bakery": "piekarnia",
    "Daycare Above And Under 2 Years": "żłobek i przedszkole", "Long Term Care": "dom opieki długoterminowej",
    "Mobile Food Preparer": "food truck", "Daycare (2 - 6 Years)": "przedszkole", "Catering": "catering",
    "Unknown": "nieznany", "Liquor": "sklep monopolowy", "Hospital": "szpital",
    "Shared Kitchen User (Long Term)": "kuchnia współdzielona", "Golden Diner": "stołówka dla seniorów",
    "Shared Kitchen": "kuchnia współdzielona", "Tavern": "bar", "Charter School": "szkoła (charter)",
    "Wholesale": "hurtownia", "Mobile Food Dispenser": "food truck", "Special Event": "wydarzenie",
    "Daycare (Under 2 Years)": "żłobek", "Banquet Hall": "sala bankietowa", "Mobile Prepared Food Vendor": "food truck",
    "Banquet": "sala bankietowa", "Private School": "szkoła prywatna", "Daycare Combo 1586": "żłobek i przedszkole",
    "Gas Station": "stacja benzynowa", "Rooftops": "taras na dachu", "Cooking School": "szkoła gotowania",
}


def facility_pl(facility_type) -> str:
    """The city's facility type in Polish; rare types keep the city's English name."""
    t = str(facility_type)
    return FACILITY_PL.get(t, t)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Polish plural: plural(1, 'wizyta', 'wizyty', 'wizyt') -> 'wizyta'; 3 -> 'wizyty'; 5 or 12 -> 'wizyt'."""
    n = abs(int(n))
    if n == 1:
        return one
    return few if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else many


def count(n: int, one: str, few: str, many: str) -> str:
    return f"{int(n)} {plural(n, one, few, many)}"


def title_case(text) -> str:
    """'3445 W 51ST ST' -> '3445 W 51st St', "MIN'S NOODLE HOUSE" -> "Min's Noodle House"."""
    t = str(text or "").strip().title()
    t = re.sub(r"'S\b", "'s", t)
    return re.sub(r"(\d)(St|Nd|Rd|Th)\b", lambda m: m.group(1) + m.group(2).lower(), t)


def every(days: int) -> str:
    """How often a risk category is inspected, in words."""
    return {182: "co 6 miesięcy", 365: "raz w roku", 730: "co 2 lata"}.get(int(days), f"co {int(days)} dni")
