from studio.fs import private_dir


def test_private_dir_creates_every_missing_level_owner_only(tmp_path):
    leaf = private_dir(tmp_path / "a" / "b")
    assert leaf.is_dir()
    assert (tmp_path / "a").stat().st_mode & 0o777 == 0o700
    assert leaf.stat().st_mode & 0o777 == 0o700


def test_private_dir_leaves_an_existing_folder_alone(tmp_path):
    (tmp_path / "a").mkdir(mode=0o755)
    private_dir(tmp_path / "a")
    assert (tmp_path / "a").stat().st_mode & 0o777 == 0o755


def test_copy_into_replaces_a_planted_link_instead_of_writing_through_it(tmp_path):
    from studio.fs import copy_into
    src = tmp_path / "still.png"
    src.write_bytes(b"still")
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "shot.png").symlink_to(victim)
    copy_into(src, folder, "shot.png")
    assert victim.read_bytes() == b"keep"
    assert not (folder / "shot.png").is_symlink() and (folder / "shot.png").read_bytes() == b"still"


def test_copy_into_refuses_a_folder_that_is_a_link(tmp_path):
    import pytest
    from studio.fs import copy_into
    src = tmp_path / "still.png"
    src.write_bytes(b"still")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "input").symlink_to(elsewhere)
    with pytest.raises(OSError):
        copy_into(src, tmp_path / "input", "shot.png")
    assert list(elsewhere.iterdir()) == []


def make_output(tmp_path):
    out = tmp_path / "io" / "output" / "studio"
    out.mkdir(parents=True)
    (out / "clip.mp4").write_bytes(b"clip")
    return tmp_path / "io"


def test_copy_out_copies_a_regular_file(tmp_path):
    from studio.fs import copy_out
    base = make_output(tmp_path)
    dest = copy_out(base, ["output", "studio", "clip.mp4"], tmp_path / "saved.mp4")
    assert dest.read_bytes() == b"clip"


def test_copy_out_refuses_absolute_and_parent_parts(tmp_path):
    import pytest
    from studio.fs import copy_out
    base = make_output(tmp_path)
    for parts in (["output", "/etc", "passwd"], ["output", "..", "secret"], ["output", "", "clip.mp4"]):
        with pytest.raises(ValueError):
            copy_out(base, parts, tmp_path / "saved.mp4")


def test_copy_out_refuses_a_linked_file_or_folder(tmp_path):
    import pytest
    from studio.fs import copy_out
    base = make_output(tmp_path)
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "family.mp4").write_bytes(b"private")
    (base / "output" / "studio" / "linked.mp4").symlink_to(outside / "family.mp4")
    (base / "output" / "linked-folder").symlink_to(outside)
    with pytest.raises(OSError):
        copy_out(base, ["output", "studio", "linked.mp4"], tmp_path / "saved.mp4")
    with pytest.raises(OSError):
        copy_out(base, ["output", "linked-folder", "family.mp4"], tmp_path / "saved.mp4")
    assert not (tmp_path / "saved.mp4").exists()
