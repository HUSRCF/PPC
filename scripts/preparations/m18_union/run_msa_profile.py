"""Run CPU profile shards inside one scheduler allocation, then audit all files."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("build_msa_features.py")
    base = [sys.executable, "-u", str(script), "--index", str(args.index),
            "--fasta", str(args.fasta), "--output-root", str(args.output_root)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def shard(i):
        with (args.output_root / f"shard_{i}.log").open("w") as log:
            subprocess.run([*base, "--mode", "profile", "--shard", str(i), "--shards", "16"],
                           stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        print(f"Profile shard {i} passed", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(shard, range(16)))
    subprocess.run([*base, "--mode", "audit-profile"], env=environment, check=True)


if __name__ == "__main__":
    main()
