"""data/readouts/catalog.json and config/readouts profiles stay consistent."""
from tools import readouts


def test_catalog_and_profiles_are_valid():
    assert readouts.check() == []


def test_every_tracked_readout_is_in_the_current_profile():
    tracked = {r["id"] for r in readouts.load_catalog()["readouts"] if r["status"] == "tracked"}
    assert tracked == set(readouts.load_profile("current")["track"])


def test_docs_are_up_to_date():
    catalog = readouts.load_catalog()
    for lang in ("ru", "en"):
        page = (readouts.project.ROOT / "docs" / lang / "game" / "readouts.md").read_text(encoding="utf-8")
        assert page == readouts.render(lang, catalog), f"run: python -m tools.readouts docs ({lang})"
