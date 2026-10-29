"""Build-time only: fetch a hash-pinned upstream tool and Apache-2.0 license."""
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
from utils.delta_tools import TOOLS, TOOL_ROOT, LICENSE_URL, LICENSE_SHA256, tool_path


def fetch(url, expected, maximum=2*1024*1024):
    import requests
    with requests.get(url, timeout=(8, 30), stream=True) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > maximum:
                raise RuntimeError("Patch tool download exceeds its build-time limit")
    data = bytes(data)
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Patch tool input failed pinned SHA-256 verification")
    return data


def ensure_tool(platform_name=None):
    platform_name = platform_name or ("windows" if sys.platform.startswith("win") else "linux")
    spec, path = TOOLS[platform_name], tool_path(platform_name)
    license_path = path.parent / "LICENSE"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec["binary_sha256"]:
        data = fetch(TOOL_ROOT+spec["archive"], spec["archive_sha256"])
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
        license_path.write_bytes(fetch(LICENSE_URL, LICENSE_SHA256))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=TOOLS)
    print(ensure_tool(parser.parse_args().platform))
