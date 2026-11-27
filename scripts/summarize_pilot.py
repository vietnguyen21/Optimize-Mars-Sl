"""Summarize one SFT metrics.jsonl session using successful optimizer updates."""
import argparse
import json
import math
from pathlib import Path


def summarize_pilot(metrics_path, summary_path=None) -> dict:
    config = None
    previous_step = None
    updates = {}
    with Path(metrics_path).open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                raise ValueError(f"Blank metrics row at {metrics_path}, 0-based line {index}")
            record = json.loads(line)
            if record.get("event") == "config":
                if config is not None:
                    raise ValueError("Multiple config events in pilot log (resumed session)")
                config = record["config"]
                if not isinstance(config, dict):
                    raise ValueError("Invalid config event")
            if "step" not in record:
                continue
            step = record["step"]
            if type(step) is not int:
                raise ValueError("Invalid optimizer step")
            if previous_step is not None and step < previous_step:
                raise ValueError("Decreasing steps in pilot log (resumed session)")
            previous_step = step
            if step >= 1 and record.get("optimizer_updated") is True and step not in updates:
                updates[step] = record

    if config is None:
        raise ValueError("Missing config event")
    if len(updates) < 2:
        raise ValueError("Timing requires two distinct successful optimizer steps")
    first_step, last_step = next(iter(updates)), next(reversed(updates))
    elapsed = updates[last_step]["time_unix"] - updates[first_step]["time_unix"]
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid elapsed time between successful optimizer updates")
    peaks = [update.get("peak_vram_bytes") for update in updates.values()]
    result = {
        "s_per_step": elapsed / (last_step - first_step),
        "peak_vram_gb": None if None in peaks else max(peaks) / (2**30),
        "max_length": config["max_length"],
        "n_dropped_overlength": config["dropped_overlength"],
    }
    if summary_path is not None:
        Path(summary_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_pilot(args.metrics_path, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
