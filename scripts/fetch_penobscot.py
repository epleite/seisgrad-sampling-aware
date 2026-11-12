"""Fetch only the Penobscot amplitude array required by the field experiment.

The source archive is the public AdaSemSeg dataset at Zenodo record 21764042.
HTTP range requests avoid downloading the unrelated labels and 2.3 GB HDF5
member.  Install the optional field-data dependency before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


URL = "https://zenodo.org/api/records/21764042/files/Penobscot_data.zip/content"
MEMBER = "Penobscot/seismic.npy"
EXPECTED_MD5 = "6afe5db0ffd28cdb0ffae98ee5d1cbdb"
ROOT = Path(__file__).resolve().parents[1]


class HashingWriter:
    """File wrapper that updates a checksum while bytes are written."""

    def __init__(self, stream, digest: hashlib._Hash) -> None:
        self.stream = stream
        self.digest = digest

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.stream.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "Penobscot" / "seismic.npy",
        help="Destination path (default: data/Penobscot/seismic.npy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination only after a verified download.",
    )
    args = parser.parse_args()

    try:
        from remotezip import RemoteZip
    except ImportError as error:
        raise SystemExit(
            "Install the optional dependency first: "
            "python -m pip install -e '.[field-data]'"
        ) from error

    destination = args.output.resolve()
    if destination.exists() and not args.force:
        raise SystemExit(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()

    try:
        with RemoteZip(URL) as archive, archive.open(MEMBER) as source:
            with temporary.open("wb") as raw_output:
                output = HashingWriter(raw_output, digest)
                shutil.copyfileobj(source, output, length=1024 * 1024)
        actual = digest.hexdigest()
        if actual != EXPECTED_MD5:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"checksum mismatch for {MEMBER}: expected {EXPECTED_MD5}, got {actual}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Saved verified Penobscot amplitude to {destination}")


if __name__ == "__main__":
    main()
