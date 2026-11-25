"""Polish wording and the city's violation notes as the place card shows them."""
from inspections.data import violations
from inspections.pl import count, title_case


def test_polish_plurals():
    assert [count(n, "wizyta", "wizyty", "wizyt") for n in (1, 3, 5, 12, 22)] == \
        ["1 wizyta", "3 wizyty", "5 wizyt", "12 wizyt", "22 wizyty"]


def test_names_read_well():
    assert title_case("MIN'S NOODLE HOUSE") == "Min's Noodle House"
    assert title_case("3445 W 51ST ST") == "3445 W 51st St"


def test_violation_notes_are_split_into_numbered_items():
    text = ("10. ADEQUATE HANDWASHING SINKS - Comments: OBSERVED NO SOAP. INSTRUCTED TO PROVIDE. | "
            "38. INSECTS, RODENTS, & ANIMALS NOT PRESENT")
    v = violations(text)
    assert [x["no"] for x in v] == ["10", "38"]
    assert v[0]["title"] == "Adequate handwashing sinks"
    assert v[0]["comment"] == "Observed no soap. Instructed to provide."
    assert v[1]["comment"] == ""
    assert violations(float("nan")) == [] and violations(None) == []
