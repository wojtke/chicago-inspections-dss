"""Registered cities. To add one, write a CityConfig module next to chicago.py and list it here."""
from .chicago import CHICAGO

CITIES = {c.slug: c for c in (CHICAGO,)}
