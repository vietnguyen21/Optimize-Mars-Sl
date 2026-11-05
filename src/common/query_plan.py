"""
Minimal QueryPlan: the structural operations of a SQL query, extracted with sqlglot.
Fields describe the outermost SELECT; `subquery` marks any nested SELECT. Labels come from gold SQL
offline, so a plan built this way is an oracle: never feed it to the system being evaluated
except in an explicitly labelled oracle experiment.
"""

import json
from typing import Dict, Any

import sqlglot
from sqlglot import exp

AGGREGATIONS = ((exp.Count, "COUNT"), (exp.Sum, "SUM"), (exp.Avg, "AVG"), (exp.Min, "MIN"), (exp.Max, "MAX"))


def extract_plan(sql: str) -> Dict[str, Any]:
    """Raises sqlglot.errors.ParseError on unparsable SQL so callers can count failures."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        raise ValueError(f"No SELECT in query: {sql[:80]}")

    aggregation = "NONE"
    for projection in select.expressions:
        for cls, name in AGGREGATIONS:
            if projection.find(cls):
                aggregation = name
                break
        if aggregation != "NONE":
            break

    order = select.args.get("order")
    direction = "NONE"
    if order:
        direction = "DESC" if order.expressions[0].args.get("desc") else "ASC"

    nested = sum(1 for _ in tree.find_all(exp.Select)) > 1
    return {
        "aggregation": aggregation,
        "group_by": bool(select.args.get("group")),
        "having": bool(select.args.get("having")),
        "order_by": bool(order),
        "direction": direction,
        "limit": bool(select.args.get("limit")),
        "distinct": bool(select.args.get("distinct") or select.find(exp.Distinct)),
        "subquery": nested
    }


def format_plan(plan: Dict[str, Any]) -> str:
    return json.dumps(plan, separators=(",", ":"))
