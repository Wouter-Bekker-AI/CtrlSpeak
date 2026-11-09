"""Build-time only: stage the vendored, hash-pinned Xdelta tool and license."""
from pathlib import Path
import argparse
import hashlib
import io
import os
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.delta_tools import TOOLS, LICENSE_SHA256, tool_path


VENDOR = ROOT / "packaging" / "vendor" / "xdelta3"


def _read_pinned_input(name, expected):
    data = (VENDOR / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Patch tool input failed pinned SHA-256 verification: {name}")
    return data


def ensure_tool(platform_name=None):
    platform_name = platform_name or ("windows" if sys.platform.startswith("win") else "linux")
    spec, path = TOOLS[platform_name], tool_path(platform_name)
    license_path = path.parent / "LICENSE"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec["binary_sha256"]:
        data = _read_pinned_input(spec["archive"], spec["archive_sha256"])
        member = spec["archive"].removesuffix(".zip").removesuffix(".tar.gz")+"/"+spec["binary"]
        if spec["archive"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                binary = archive.read(member)
        else:
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                binary = archive.extractfile(member).read()
        if hashlib.sha256(binary).hexdigest() != spec["binary_sha256"]:
            raise RuntimeError("Patch tool executable failed pinned SHA-256 verification")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        temporary.write_bytes(binary)
        os.replace(temporary, path)
        if platform_name == "linux":
            path.chmod(0o755)
    if not license_path.is_file() or hashlib.sha256(license_path.read_bytes()).hexdigest() != LICENSE_SHA256:
        license_path.write_bytes(_read_pinned_input("LICENSE", LICENSE_SHA256))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=TOOLS)
    print(ensure_tool(parser.parse_args().platform))
