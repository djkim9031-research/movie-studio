import cv2
import numpy as np
import pytest
from PIL import Image

from studio import refs
from studio.refs import Face


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def face(x, y, h, embedding, sharpness=200.0, yaw=0.0):
    landmarks = np.array([[x + 0.3 * h, y + 0.4 * h], [x + 0.7 * h, y + 0.4 * h], [x + 0.5 * h + yaw * 0.4 * h,
                          y + 0.6 * h], [x + 0.35 * h, y + 0.8 * h], [x + 0.65 * h, y + 0.8 * h]], dtype=np.float32)
    return Face(box=(x, y, h, h), landmarks=landmarks, score=0.9, embedding=embedding, sharpness=sharpness)


# Same person in two photos: similarity 0.68, as between the real reference photos. Others stay below 0.363.
ME = unit(1, 0.3, 0.2)
ME_TOO = unit(1, -0.6, 0.5)
OTHER = unit(0, 1, 0)


def test_the_subject_is_the_identity_in_the_most_pictures():
    faces = {"a": [face(0, 0, 400, ME)], "b": [face(0, 0, 400, ME_TOO), face(500, 0, 300, OTHER)],
             "c": [face(0, 0, 400, OTHER)], "d": [face(0, 0, 400, ME)]}
    subject = refs.find_subject(faces)
    assert sorted(name for name, _ in subject) == ["a", "b", "d"]
    assert dict(subject)["b"] is faces["b"][0]


def test_two_faces_in_one_picture_are_never_the_same_person():
    faces = {"a": [face(0, 0, 400, ME), face(500, 0, 400, ME_TOO)]}
    assert len(refs.find_subject(faces)) == 1


def test_a_face_that_matches_the_subject_only_weakly_is_left_out():
    weak = unit(0, 0.2, 1)
    faces = {"a": [face(0, 0, 400, ME)], "b": [face(0, 0, 400, ME_TOO)], "c": [face(0, 0, 400, ME)],
             "d": [face(0, 0, 400, weak)]}
    assert "d" not in dict(refs.find_subject(faces))


def test_best_faces_prefers_sharp_large_frontal_faces_and_skips_near_duplicates():
    chosen = refs.best_faces([("small", face(0, 0, 150, unit(1, 0.3, -0.4), sharpness=100)),
                              ("sharp", face(0, 0, 900, ME, sharpness=400)),
                              ("copy", face(0, 0, 900, unit(1, 0.3, 0.21), sharpness=390)),
                              ("turned", face(0, 0, 900, ME_TOO, sharpness=300, yaw=0.45))], limit=3)
    assert [name for name, _ in chosen] == ["sharp", "turned", "small"]


def test_square_crop_blurs_another_face_that_overlaps():
    picture = np.zeros((2000, 2000, 3), dtype=np.uint8)
    yy, xx = np.mgrid[700:1000, 1250:1550]
    picture[700:1000, 1250:1550] = (((yy // 8 + xx // 8) % 2) * 255).astype(np.uint8)[..., None]
    me, other = face(800, 700, 400, ME), face(1250, 700, 300, OTHER)
    x0, y0, x1, y1 = refs.crop_window(picture.shape, me, [other])
    raw = cv2.resize(picture[y0:y1, x0:x1], (1024, 1024))
    crop = refs.square_crop(picture, me, [other])
    assert crop.shape == (1024, 1024, 3)
    assert cv2.Laplacian(crop, cv2.CV_64F).var() < 0.1 * cv2.Laplacian(raw, cv2.CV_64F).var()


def test_square_crop_shrinks_the_window_to_leave_out_a_face_nearby():
    picture = np.zeros((3000, 3000, 3), dtype=np.uint8)
    me, other = face(1300, 1300, 400, ME), face(1850, 1300, 400, OTHER)
    window = refs.crop_window(picture.shape, me, [other])
    assert window[2] <= 1850 and window[2] - window[0] >= 1.6 * 400


def test_extra_pictures_keep_their_aspect_and_get_a_padded_square_for_the_upsampler():
    wide = np.full((1000, 2000, 3), 200, dtype=np.uint8)
    full, square = refs.extra_versions(wide)
    assert full.shape == (1000, 2000, 3)
    assert square.shape == (1024, 1024, 3) and square[0, 0].tolist() == [0, 0, 0] and square[512, 512, 0] == 200


class FakeTools:
    def __init__(self, faces):
        self.by_size = faces

    def faces(self, bgr):
        return self.by_size.get(bgr.shape[:2], [])


def write(path, size, exif=True):
    image = Image.new("RGB", size, (90, 120, 150))
    if exif:
        meta = Image.Exif()
        meta[0x010F] = "PhoneMaker"
        image.save(path, exif=meta.tobytes())
    else:
        image.save(path)
    return path


def test_prepare_writes_person_crops_and_extras_and_reports_what_it_skipped(tmp_path):
    pics = [write(tmp_path / "p1.jpg", (2000, 1500)), write(tmp_path / "p2.jpg", (1800, 1500)),
            write(tmp_path / "other.jpg", (1600, 1500)), write(tmp_path / "saturn.png", (1400, 700), exif=False)]
    tools = FakeTools({(1500, 2000): [face(800, 400, 500, ME)], (1500, 1800): [face(700, 400, 500, ME_TOO)],
                       (1500, 1600): [face(600, 400, 500, OTHER)]})
    out = tmp_path / "refs"
    report = refs.prepare(pics, out, tools)
    assert [(r.kind, r.file.name, r.source) for r in report.used] == [
        ("person", "person1.png", "p1.jpg"), ("person", "person2.png", "p2.jpg"), ("extra", "extra1.png", "saturn.png")]
    assert report.used[-1].square.name == "extra1_square.png"
    assert report.skipped == [("other.jpg", "only other people")]
    with Image.open(out / "person1.png") as crop:
        assert crop.size == (1024, 1024) and not crop.getexif() and not crop.info.get("exif")
    assert (out.stat().st_mode & 0o777) == 0o700


def test_prepare_caps_the_references_at_ten(tmp_path):
    pics = [write(tmp_path / f"place{i}.png", (800 + i, 600), exif=False) for i in range(12)]
    report = refs.prepare(pics, tmp_path / "refs", FakeTools({}))
    assert len(report.used) == 10 and len(report.skipped) == 2
    assert report.skipped[0][1] == "over the 10-reference limit"


def test_prepare_refuses_no_pictures(tmp_path):
    with pytest.raises(ValueError):
        refs.prepare([], tmp_path / "refs", FakeTools({}))
