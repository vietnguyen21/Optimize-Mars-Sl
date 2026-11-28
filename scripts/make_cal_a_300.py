"""Create cal_a_300_ids.txt and cal_a_300_report.json once (QUC s3.3). S5 only; other streams read the file.

  python scripts/make_cal_a_300.py --data-dir <Drive>/data --out-dir <Drive>/members/m5/work/cal_a_300

Stops (non-zero exit) when the sample differs from the Leader's reference values; it never overwrites.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.generator_only import (cal_a_300_ids_text, cal_a_300_indices,  # noqa: E402
                                           check_cal_a_300_reference, read_jsonl_lf, sha256_file, CAL_A_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder holding retrieval_dev.jsonl")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.data_dir) / "retrieval_dev.jsonl"
    rows = read_jsonl_lf(source)
    if len(rows) != CAL_A_SIZE:
        raise SystemExit(f"{source} has {len(rows)} rows, expected {CAL_A_SIZE}; stop and report to the Leader")
    indices = cal_a_300_indices()
    text = cal_a_300_ids_text(indices)
    check_cal_a_300_reference(indices, text)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"n": len(indices),
              "per_db": dict(sorted(Counter(rows[i]["db_id"] for i in indices).items())),
              "sha256_ids": hashlib.sha256(text.encode("utf-8")).hexdigest(),
              "sha256_retrieval_dev": sha256_file(source),
              "first5": indices[:5], "last5": indices[-5:]}
    # 'x': created once; an existing list is never regenerated or changed.
    with open(out / "cal_a_300_ids.txt", "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(out / "cal_a_300_report.json", "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
