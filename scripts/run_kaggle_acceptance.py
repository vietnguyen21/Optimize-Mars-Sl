"""Entry point: no default Hub writes, large model downloads or full training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.acceptance import parser,run


def main():
    arguments=parser().parse_args()
    try:
        report=run(arguments)
    except (ValueError, OSError) as exc:
        # These errors are authored by argument/output validation, before access
        # to credentials. Do not print arbitrary provider exception messages.
        print(type(exc).__name__+': invalid options or nonempty acceptance output; run --help',file=sys.stderr)
        return 2
    return 1 if report['result']=='FAIL' else 0


if __name__=='__main__':
    raise SystemExit(main())
