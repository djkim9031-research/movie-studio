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
