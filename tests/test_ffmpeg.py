import io
import tarfile

import pytest

from studio.media.ffmpeg import (ChecksumMismatch, extract_binaries, parse_ebur128_summary, parse_encoders,
                                 parse_filters, verify_sha256)

ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""

FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
 ... xfade             VV->V      Cross fade one video with another video.
 T.. loudnorm          A->A       EBU R128 loudness normalization
 ..C sidechaincompress AA->A      Sidechain compressor.
"""

EBUR128 = """[Parsed_ebur128_0 @ 0x55d0] Summary:

  Integrated loudness:
    I:         -16.1 LUFS
    Threshold: -26.1 LUFS

  Loudness range:
    LRA:         0.0 LU

  True peak:
    Peak:      -13.4 dBFS
"""


def test_parse_encoders_reads_names_after_the_rule():
    assert parse_encoders(ENCODERS) == {"libx264", "h264_nvenc", "aac"}


def test_parse_filters_reads_names_and_skips_the_legend():
    assert parse_filters(FILTERS) == {"xfade", "loudnorm", "sidechaincompress"}


def test_parse_ebur128_summary_reads_integrated_loudness_and_true_peak():
    assert parse_ebur128_summary(EBUR128) == (-16.1, -13.4)


def test_verify_sha256_refuses_a_mismatch(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"not the release")
    with pytest.raises(ChecksumMismatch):
        verify_sha256(path, "0" * 64)


def add_member(tar, name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_extract_binaries_takes_only_ffmpeg_and_ffprobe(tmp_path):
    tarball = tmp_path / "ffmpeg.tar.xz"
    with tarfile.open(tarball, "w:xz") as tar:
        add_member(tar, "ffmpeg-n8.1/bin/ffmpeg", b"ffmpeg")
        add_member(tar, "ffmpeg-n8.1/bin/ffprobe", b"ffprobe")
        add_member(tar, "ffmpeg-n8.1/doc/README.txt")
        add_member(tar, "../bin/ffmpeg", b"evil")
    dest = tmp_path / "bin"
    extract_binaries(tarball, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["ffmpeg", "ffprobe"]
    assert (dest / "ffmpeg").read_bytes() == b"ffmpeg"
