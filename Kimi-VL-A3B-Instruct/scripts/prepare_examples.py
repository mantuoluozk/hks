"""Validate the publisher's 525 KB demo image; download only with --download."""
import argparse
import hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SHA256 = "95de8765da89c41a2421f1c1fa3986e4d3c83793d92c8ade460a142b329d04c1"
URL = "https://hf-mirror.com/moonshotai/Kimi-VL-A3B-Instruct/resolve/398eede0903cd983a2bfa0cc634e9ac1d843f375/figures/demo.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    path = ROOT/"data/examples/demo.png"
    if not path.exists() and args.download:
        response = requests.get(URL, timeout=(15, 60))
        response.raise_for_status()
        assert len(response.content) == 525038
        assert hashlib.sha256(response.content).hexdigest() == SHA256
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    assert path.stat().st_size == 525038
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
    print("Publisher demo image SHA256 verified")


if __name__ == "__main__":
    main()
