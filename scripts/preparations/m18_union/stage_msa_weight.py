"""Resume an official model download locally, then verify its HPC3 transfer."""

import hashlib
import json
from pathlib import Path
import subprocess
import time


URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt"
EXPECTED_BYTES = 1387625262
PATH = Path("/tmp/esm_msa1b_t12_100M_UR50S.pt")
DESTINATION = "/data/user/shuang886/PPC_M18/sequence_union_v2_20261030/msa_factorial_20261104/weights"


def main():
    subprocess.run(["aria2c", "--continue=true", "--file-allocation=none",
                    "--auto-file-renaming=false", "--max-connection-per-server=16",
                    "--split=16", "--min-split-size=4M", "--summary-interval=60",
                    "--show-console-readout=false", "--dir=/tmp", "--out=" + PATH.name, URL], check=True)
    if PATH.stat().st_size != EXPECTED_BYTES or Path(str(PATH) + ".aria2").exists():
        raise ValueError("Official asset download is incomplete")
    with PATH.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    subprocess.run(["ssh", "-o", "BatchMode=yes", "hpc3", "mkdir -p " + DESTINATION], check=True)
    subprocess.run(["rsync", "-a", "--partial", "--append-verify", str(PATH),
                    "hpc3:" + DESTINATION + "/" + PATH.name], check=True)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "hpc3",
                             "sha256sum " + DESTINATION + "/" + PATH.name],
                            check=True, text=True, capture_output=True)
    if result.stdout.split()[0] != checksum:
        raise ValueError("Local/HPC3 official checkpoint hash mismatch")
    receipt = PATH.with_suffix(".ready.json")
    receipt.write_text(json.dumps({"url": URL, "bytes": EXPECTED_BYTES, "sha256": checksum,
                                   "transfer_verified": True, "completed_unix": time.time()}, indent=2) + "\n")
    subprocess.run(["rsync", "-a", str(receipt), "hpc3:" + DESTINATION + "/ready.json"], check=True)
    print(json.dumps({"status": "verified_on_hpc3", "sha256": checksum}), flush=True)


if __name__ == "__main__":
    main()
