"""B2.7 read-only SQLite audit; no training or remote operations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.grounding_audit import audit_grounding,EXPECTED_SQLGLOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--db-root',required=True)
    parser.add_argument('--cards',required=True)
    parser.add_argument('--output',required=True,help='New output directory; existing artifacts are never replaced')
    parser.add_argument('--split',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-fraction',type=float,default=.1)
    parser.add_argument('--sqlglot-version',default=EXPECTED_SQLGLOT)
    args=parser.parse_args()
    summary=audit_grounding(args.data,args.db_root,args.cards,args.output,split=args.split,
        seed=args.seed,validation_fraction=args.validation_fraction,expected_sqlglot=args.sqlglot_version)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 2 if summary['split_status']=='FAIL' else 0


if __name__=='__main__':raise SystemExit(main())
