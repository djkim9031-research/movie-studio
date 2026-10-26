import json

import pytest
from PIL import Image

from studio import blind


def image(path, color):
    Image.new("RGB", (32, 18), color).save(path)
    return path


def groups(tmp_path):
    a1, b1 = image(tmp_path / "a1.png", "red"), image(tmp_path / "b1.png", "blue")
    a2, b2 = image(tmp_path / "a2.png", "red"), image(tmp_path / "b2.png", "blue")
    return [[{"id": "runA/p1-still", "run": "runA", "images": [a1], "score": 0.9},
             {"id": "runB/p1-still", "run": "runB", "images": [b1], "score": 0.1}],
            [{"id": "runA/p2-still", "run": "runA", "images": [a2], "score": 0.2},
             {"id": "runB/p2-still", "run": "runB", "images": [b2], "score": 0.3}]]


def test_the_page_has_no_labels_names_or_scores_and_the_key_is_private(tmp_path):
    out = tmp_path / "blind"
    blind.make(groups(tmp_path), out, seed=3)
    page = (out / "page.html").read_text()
    assert "runA" not in page and "runB" not in page and "0.9" not in page and "a1.png" not in page
    assert "Group 1" in page and ">A<" in page and ">B<" in page
    assert (out / "key.json").stat().st_mode & 0o777 == 0o600
    assert len(list((out / "img").iterdir())) == 4


def test_picks_tally_preferences_and_agreement_with_clear_score_differences(tmp_path):
    out = tmp_path / "blind"
    key = blind.make(groups(tmp_path), out, seed=3)
    letter_of = {o["id"]: letter for g in key["groups"] for letter, o in g["options"].items()}
    picks = [f"1{letter_of['runA/p1-still']}", f"2{letter_of['runA/p2-still']}"]
    report = blind.pick(out, picks, delta=0.25)
    assert report["wins"] == {"runA": 2, "runB": 0}
    assert (report["agree"], report["disagree"]) == (1, 0)
    assert json.loads((out / "picks.json").read_text())["picks"] == picks


def test_picks_must_name_a_group_and_an_option(tmp_path):
    out = tmp_path / "blind"
    blind.make(groups(tmp_path), out, seed=3)
    with pytest.raises(ValueError):
        blind.pick(out, ["3A"], delta=0.25)
    with pytest.raises(ValueError):
        blind.pick(out, ["1C"], delta=0.25)
    assert blind.pick(out, ["1="], delta=0.25)["ties"] == 1
