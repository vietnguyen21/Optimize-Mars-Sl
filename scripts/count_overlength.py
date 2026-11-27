"""Count full-schema SFT examples with the trainer's prompt-v2 encoding."""
import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_sft_generator import CompletionOnlyDataCollator


def count_overlength(data_path, tokenizer, max_length=2048) -> dict:
    collator = CompletionOnlyDataCollator(tokenizer, max_length=max_length, prompt_format="v2")
    kept = dropped = 0
    with Path(data_path).open("rb") as stream:
        # Binary iteration splits only at LF and omits the empty tail after a final LF.
        for index, line in enumerate(stream):
            if not line.strip():
                raise ValueError(f"Blank SFT row at {data_path}, 0-based line {index}")
            row = json.loads(line)
            try:
                collator.encode_example(row)
            except ValueError as error:
                if "max_length=" not in str(error):
                    raise
                dropped += 1
            else:
                kept += 1
    return {
        "n": kept + dropped,
        "n_kept": kept,
        "n_dropped_overlength": dropped,
        "max_length": max_length,
        "epoch_steps": (kept + 7) // 8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct", revision=os.environ.get("MODEL_REVISION") or "main"
    )
    print(json.dumps(count_overlength(args.data_path, tokenizer, args.max_length)))


if __name__ == "__main__":
    main()
