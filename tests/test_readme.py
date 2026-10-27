import re
from pathlib import Path

from studio import film, sound

README = Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_fetches_every_model_set_a_film_with_sound_needs():
    fetched = set(re.findall(r"uv run studio models fetch ([\w-]+)", README.read_text(encoding="utf-8")))
    assert set(film.MODEL_SETS) | set(sound.SETS) <= fetched
