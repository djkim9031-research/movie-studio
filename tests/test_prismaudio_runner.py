import importlib.util
import sys

import pytest

from studio.sound import RUNNER


@pytest.fixture(scope="module")
def runner():
    """The worker's run.py; its top-level imports are all standard library."""
    keep = sys.dont_write_bytecode
    spec = importlib.util.spec_from_file_location("prismaudio_run", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.dont_write_bytecode = keep
    return module


def test_the_latent_covers_every_whole_second_length(runner):
    # Upstream's round(44100 / 64 / 32 * seconds) gives 215 frames for 10 s: 15 ms short of the clip.
    for seconds in range(1, runner.MAX_SECONDS + 1):
        samples = runner.latent_length(seconds) * runner.LATENT_SAMPLES
        assert seconds * runner.SAMPLE_RATE <= samples < seconds * runner.SAMPLE_RATE + runner.LATENT_SAMPLES


def test_the_latent_is_upstreams_wherever_that_covers_the_clip(runner):
    for seconds in range(1, runner.MAX_SECONDS + 1):
        upstream = round(44100 / 64 / 32 * seconds)
        if upstream * 64 * 32 >= seconds * 44100:
            assert runner.latent_length(seconds) == upstream
