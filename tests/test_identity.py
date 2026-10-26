import numpy as np
import pytest

from studio import identity
from studio.config import Config, load_config
from studio.identity import Frame
from studio.media import ffmpeg
from tests.test_refs import ME, ME_TOO, OTHER, FakeTools, face, unit, write

def frames(sims, fps=24, px=300, yaw=0.0):
    return [Frame(t=i / fps, sim=s, yaw=None if s is None else yaw, px=None if s is None else px)
            for i, s in enumerate(sims)]


def test_pinned_frames_and_two_on_each_side_are_left_out():
    mask = identity.pinned_mask(20, [0, -1])
    assert [i for i, pinned in enumerate(mask) if pinned] == [0, 1, 2, 17, 18, 19]


def test_rolling_median_ignores_a_one_frame_blink():
    times = [i / 24 for i in range(24)]
    values = [0.7] * 24
    values[12] = 0.1
    assert identity.rolling_median(times, values, 0.3)[12] == pytest.approx(0.7)


def test_summary_reports_median_mid_clip_minimum_dips_and_coverage():
    sims = [0.7] * 120
    sims[50:62] = [0.3] * 12
    sims[100:105] = [None] * 5
    clip = frames(sims)
    clip[110] = Frame(t=110 / 24, sim=0.7, yaw=0.0, px=60)
    s = identity.summarize(clip, pinned=[0], genuine_p5=0.45)
    assert s.median == pytest.approx(0.7) and s.mid_min == pytest.approx(0.3)
    assert s.dips == 1 and s.dip_seconds == pytest.approx(12 / 24, abs=0.05)
    assert s.coverage == pytest.approx((117 - 6) / 117)


def test_a_short_drop_is_not_a_dip():
    sims = [0.7] * 120
    sims[50:53] = [0.3] * 3
    assert identity.summarize(frames(sims), pinned=[0], genuine_p5=0.45).dips == 0


def test_faces_turned_past_the_scorable_range_are_unscored_not_failed():
    clip = frames([0.7] * 48) + [Frame(t=(48 + i) / 24, sim=0.1, yaw=0.9, px=300) for i in range(24)]
    s = identity.summarize(clip, pinned=[], genuine_p5=0.45, max_yaw=0.5)
    assert s.dips == 0 and s.coverage == pytest.approx(48 / 72) and s.yaw_range == pytest.approx(0.0)


def test_percentile_is_the_share_of_genuine_pairs_at_or_below():
    profile = identity.Profile(alias="me", scorer="sface", embeddings=np.zeros((0, 3)), sources=[],
                               genuine=np.array([0.4, 0.5, 0.6, 0.7]), yaws=[])
    assert identity.percentile(profile, 0.55) == pytest.approx(50.0)
    assert identity.percentile(profile, 0.9) == pytest.approx(100.0)


def make_photos(cfg, n_me=4):
    folder = cfg.data_root / "cast" / "me" / "photos"
    folder.mkdir(parents=True)
    sizes = []
    for i in range(n_me):
        sizes.append((1500 + 10 * i, 1200))
        write(folder / f"me{i}.jpg", sizes[-1])
    write(folder / "friend.jpg", (1400, 1200))
    vectors = [ME, ME_TOO, unit(1, 0.3, -0.4), unit(1, -0.2, 0.1)]
    tools = FakeTools({(h, w): [face(400, 300, 400, vectors[i % 4])] for i, (w, h) in enumerate(sizes)}
                      | {(1200, 1400): [face(400, 300, 400, OTHER)]})
    return folder, tools


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({"data_root": str(tmp_path)})


def test_profile_uses_the_person_in_most_photos_and_leaves_out_bench_photos(cfg):
    _, tools = make_photos(cfg)
    profile = identity.build_profile(cfg, "me", tools, leave_out=["me3.jpg"])
    assert profile.sources == ["me0.jpg", "me1.jpg", "me2.jpg"]
    assert profile.embeddings.shape == (3, 3) and len(profile.genuine) == 3
    assert np.linalg.norm(profile.centroid) == pytest.approx(1.0)


def test_profile_is_saved_owner_only_and_loads_back_with_the_same_digest(cfg):
    _, tools = make_photos(cfg)
    profile = identity.build_profile(cfg, "me", tools)
    path = identity.save_profile(cfg, profile)
    assert (path.stat().st_mode & 0o777) == 0o600
    again = identity.load_profile(cfg, "me")
    assert again.digest == profile.digest and again.sources == profile.sources


needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(load_config()).exists(), reason="run: uv run studio fetch-ffmpeg")


@needs_ffmpeg
def test_score_clip_reads_every_frame_with_its_time(tmp_path):
    real = load_config()
    clip = tmp_path / "clip.mp4"
    ffmpeg.run(real, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=1",
               "-pix_fmt", "yuv420p", str(clip)).check_returncode()
    profile = identity.Profile(alias="me", scorer="sface", embeddings=np.stack([ME]), sources=["a"],
                               genuine=np.array([0.5, 0.6, 0.7]), yaws=[0.0])
    tools = FakeTools({(180, 320): [face(100, 40, 120, ME)]})
    result = identity.score_clip({"sface": profile}, tools, clip, pinned=[0])
    assert len(result.frames) == 24 and result.frames[12].t == pytest.approx(0.5)
    assert result.frames[12].sims["sface"] == pytest.approx(1.0, abs=1e-5)
    assert result.summary.median == pytest.approx(profile.z(1.0), abs=1e-4)
    assert result.summary.by_scorer["sface"] == pytest.approx(1.0, abs=1e-5)


def test_a_profile_without_one_photo_scores_that_photo_as_unseen(cfg):
    _, tools = make_photos(cfg)
    profile = identity.build_profile(cfg, "me", tools)
    rest = profile.without("me1.jpg")
    assert rest.sources == ["me0.jpg", "me2.jpg", "me3.jpg"] and len(rest.genuine) == 3
    assert not np.allclose(rest.centroid, profile.centroid)


class TwoScorerTools(FakeTools):
    """Each face carries an AuraFace-like embedding equal to its SFace one, rotated."""

    def faces(self, bgr):
        found = super().faces(bgr)
        for f in found:
            f.extra["auraface"] = np.roll(f.embedding, 1)
        return found


def test_profiles_are_built_per_scorer_and_saved_side_by_side(cfg):
    _, tools = make_photos(cfg)
    tools = TwoScorerTools(tools.by_size)
    profiles = identity.build_profiles(cfg, "me", tools, scorers=("sface", "auraface"))
    assert set(profiles) == {"sface", "auraface"}
    assert np.allclose(profiles["auraface"].embeddings, np.roll(profiles["sface"].embeddings, 1, axis=1))
    for p in profiles.values():
        identity.save_profile(cfg, p)
    again = identity.load_profiles(cfg, "me")
    assert set(again) == {"sface", "auraface"} and identity.digest(again) == identity.digest(profiles)


def test_z_scores_put_both_scorers_on_the_genuine_scale():
    profile = identity.Profile(alias="me", scorer="sface", embeddings=np.zeros((0, 3)), sources=[],
                               genuine=np.array([0.5, 0.6, 0.7]), yaws=[])
    assert profile.z(0.6) == pytest.approx(0.0)
    assert profile.z(0.7) == pytest.approx(0.1 / np.std([0.5, 0.6, 0.7]))


def test_a_frame_score_is_the_mean_z_of_the_scorers_and_keeps_each_cosine(cfg):
    _, tools = make_photos(cfg)
    tools = TwoScorerTools(tools.by_size)
    profiles = identity.build_profiles(cfg, "me", tools, scorers=("sface", "auraface"))
    picture = cfg.data_root / "cast" / "me" / "photos" / "me0.jpg"
    frame = identity.score_image(profiles, tools, picture)
    expected = np.mean([profiles[s].z(frame.sims[s]) for s in ("sface", "auraface")])
    assert set(frame.sims) == {"sface", "auraface"} and frame.sim == pytest.approx(expected)


def test_the_dip_line_of_a_combined_score_is_the_mean_z_of_each_genuine_fifth_percentile(cfg):
    _, tools = make_photos(cfg)
    profiles = identity.build_profiles(cfg, "me", TwoScorerTools(tools.by_size), scorers=("sface", "auraface"))
    line = identity.dip_line(profiles)
    assert line == pytest.approx(np.mean([p.z(p.genuine_p5) for p in profiles.values()]))
