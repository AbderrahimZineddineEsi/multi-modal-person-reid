"""Dataset fetching: resumable download with size check, extraction, reuse of existing files."""

import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reid.datasets import par_fetch as pf

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/f.bin"
    srv.shutdown()


def test_download_resumes_and_skips_complete(tmp_path, server):
    dest = tmp_path / "d" / "f.bin"
    dest.parent.mkdir()
    (tmp_path / "d" / "f.bin.part").write_bytes(PAYLOAD[:1000])  # an interrupted download
    pf.download(server, dest, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD and not (tmp_path / "d" / "f.bin.part").exists()
    mtime = dest.stat().st_mtime_ns
    pf.download("http://127.0.0.1:1/never", dest, len(PAYLOAD))  # complete: no request at all
    assert dest.stat().st_mtime_ns == mtime


def test_download_size_mismatch_raises(tmp_path, server, monkeypatch):
    monkeypatch.setattr(pf.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="could not download"):
        pf.download(server, tmp_path / "f.bin", len(PAYLOAD) + 5, retries=2)
    assert not (tmp_path / "f.bin").exists()


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def test_extract_once_and_from_dir(tmp_path):
    src, out = tmp_path / "src" / "nested", tmp_path / "out"
    src.mkdir(parents=True)
    _zip(src / "annotation.zip", {"ann/annotation.mat": "m"})
    _zip(src / "data.zip", {"data/000001.jpg": "i"})
    _zip(src / "PETA_dropbox.zip", {"PETA dataset/A/archive/Label.txt": "l"})
    pf.fetch(out, tmp_path / "src")
    assert (out / "pa100k/ann/annotation.mat").is_file() and (
        out / "pa100k/data/000001.jpg"
    ).is_file()
    assert (out / "peta/PETA dataset/A/archive/Label.txt").is_file()
    (out / "pa100k/ann/annotation.mat").write_text("changed")
    pf.fetch(out, tmp_path / "src")  # already extracted: left alone
    assert (out / "pa100k/ann/annotation.mat").read_text() == "changed"


def test_from_dir_with_extracted_folders(tmp_path):
    src = tmp_path / "in"
    (src / "pa" / "ann").mkdir(parents=True)
    (src / "pa" / "ann" / "annotation.mat").write_text("m")
    (src / "peta" / "S" / "archive").mkdir(parents=True)
    (src / "peta" / "S" / "archive" / "Label.txt").write_text("l")
    got = pf.use_existing(src, tmp_path / "out")
    assert got == {"pa100k": "linked", "peta": "linked"}
    assert (tmp_path / "out" / "pa100k" / "pa" / "ann" / "annotation.mat").is_file()
