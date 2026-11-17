"""Conservative, schema-backed SQL labels. Never executes reference queries.

scope_v2 is opt-in for training: existing pilot checkpoints retain legacy labels.
SQLGlot qualification operates on a copy, never rewrites source gold SQL.
"""
from contextlib import closing
from pathlib import Path
import sqlite3

RESOLVER_VERSION = 'scope_v2'


class GroundingReview(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def open_readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA trusted_schema=OFF')
    return conn


def sqlite_schema(path):
    with closing(open_readonly(path)) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")]
        result = {}
        for table in tables:
            escaped = table.replace('"', '""')
            columns = {}
            for row in conn.execute(f'PRAGMA table_xinfo("{escaped}")'):
                name = row[1].lower()
                if name in columns:
                    raise GroundingReview('schema_collision', 'Case-normalized column collision')
                columns[name] = row[2] or 'UNKNOWN'
            if table.lower() in result:
                raise GroundingReview('schema_collision', 'Case-normalized table collision')
            result[table.lower()] = columns
        return result


def compile_sqlite(path, sql):
    """Compile only, with read-only URI, query_only, and deny-write authorizer."""
    import re
    head = re.sub(r'\A(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*', '', sql, flags=re.S)
    if not re.match(r'(SELECT|WITH)\b', head, re.I):
        return dict(status='FAIL', code='unsupported_statement', message='Only SELECT/WITH references are audited')
    try:
        with closing(open_readonly(path)) as conn:
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                       getattr(sqlite3, 'SQLITE_RECURSIVE', 33)}
            conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
            conn.execute('EXPLAIN QUERY PLAN ' + sql)
        return dict(status='PASS', code=None)
    except sqlite3.Error as exc:
        message = str(exc)
        lower = message.lower()
        code = ('missing_table' if 'no such table:' in lower else
                'missing_column' if 'no such column:' in lower else
                'ambiguous_column' if 'ambiguous column name:' in lower else
                'syntax_error' if any(s in lower for s in ('syntax error', 'incomplete input', 'one statement at a time')) else
                'other_sqlite_error')
        return dict(status='FAIL', code=code, message=message)


def card_names(cards, indices):
    names = {}
    for index in indices:
        card = cards[index]
        name = f"{card['table_name']}.{card['column_name']}".lower()
        if name in names:
            raise GroundingReview('duplicate_card', f'Duplicate/ambiguous card identity: {name}')
        names[name] = index
    return names


def cards_for_db(cards, db_id, *, strict=False, schema=None, single_db=False):
    indices = [i for i,c in enumerate(cards) if c.get('db_id') == db_id]
    if not indices:
        # Legacy untagged fixtures can be verified against one DB. Never assign
        # one untagged card to multiple databases just because table names match.
        if strict or not single_db or schema is None:
            raise GroundingReview('card_db_membership', 'Cards require explicit db_id for strict/multi-database labels')
        indices = [i for i,c in enumerate(cards) if not c.get('db_id') and
                   c['table_name'].lower() in schema and c['column_name'].lower() in schema[c['table_name'].lower()]]
    names = card_names(cards, indices)
    if not names:
        raise GroundingReview('card_db_membership', f'No cards for database {db_id}')
    if schema is not None:
        for i in indices:
            c=cards[i]; table=c['table_name'].lower(); column=c['column_name'].lower()
            if table not in schema or column not in schema[table]:
                raise GroundingReview('card_schema_mismatch', f'Card absent from SQLite schema: {table}.{column}')
    return indices, names


def schema_from_cards(cards, indices):
    schema = {}
    for i in indices:
        c=cards[i]
        schema.setdefault(c['table_name'].lower(), {})[c['column_name'].lower()] = 'UNKNOWN'
    return schema


def resolve_sql_columns(sql, schema):
    """Return physical table.column evidence; fail the whole record on uncertainty.

    SQLGlot qualifies using only selected sources, expands projections (not
    COUNT(*)), validates CTE output names, and handles correlated references.
    Walk each scope locally, never process a child's columns with parent aliases.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError, OptimizeError, SqlglotError
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.optimizer.scope import traverse_scope, Scope
    try:
        statements = sqlglot.parse(sql, read='sqlite')
        if len(statements)!=1 or not isinstance(statements[0], exp.Query):
            raise GroundingReview('unsupported_statement', 'Expected one SELECT query')
        tree = statements[0]
        if any(join.args.get('method') == 'NATURAL' for join in tree.find_all(exp.Join)):
            raise GroundingReview('unsupported_join', 'NATURAL JOIN implicit column evidence requires review')
        # Validate physical tables before optimizer inference can hide missing sources.
        for scope in traverse_scope(tree):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    if source.db or source.catalog or source.name.lower() not in schema:
                        raise GroundingReview('missing_table', f'Unknown physical table: {source.sql()}')
        tree = qualify(tree.copy(), dialect='sqlite',
                       schema={t:{c:'UNKNOWN' for c in cols} for t,cols in schema.items()}, infer_schema=False,
                       allow_partial_qualification=False, identify=False, quote_identifiers=False)
        positives = set()
        for scope in traverse_scope(tree):
            for column in (n for n in scope.walk() if isinstance(n, exp.Column)):
                if column.is_star:
                    raise GroundingReview('unresolved_wildcard', 'Projection wildcard could not be expanded')
                # Qualification intentionally retains SELECT output aliases in
                # top-level ORDER BY. They refer to an already visited projection,
                # not to an additional physical column or a database-wide match.
                if (not column.table and scope.expression.args.get('order') is not None
                        and column.find_ancestor(exp.Order) is scope.expression.args['order']):
                    outputs=[p for p in scope.expression.selects if p.alias_or_name.lower()==column.name.lower()]
                    if len(outputs)==1:
                        continue
                owner=scope; source=None
                while owner is not None:
                    selected={name.lower(): pair[1] for name,pair in owner.selected_sources.items()}
                    if column.table.lower() in selected:
                        source=selected[column.table.lower()]
                        break
                    if not owner.can_be_correlated:
                        break
                    owner=owner.parent
                if source is None:
                    raise GroundingReview('unresolved_column', f'Unresolved column: {column.sql()}')
                if isinstance(source, exp.Table):
                    table=source.name.lower(); name=column.name.lower()
                    if table not in schema or name not in schema[table]:
                        raise GroundingReview('missing_column', f'Unknown column: {table}.{name}')
                    positives.add(f'{table}.{name}')
                elif isinstance(source, Scope):
                    # Its output was validated by qualify. Physical evidence is
                    # collected inside its own scope, including expression inputs.
                    continue
                else:
                    raise GroundingReview('unsupported_source', 'Unverified table-valued source')
        if not positives:
            raise GroundingReview('no_positive_columns', 'No physical column evidence (COUNT(*)/literals alone are not labels)')
        return sorted(positives)
    except GroundingReview:
        raise
    except ParseError as exc:
        raise GroundingReview('parse_error', str(exc)) from exc
    except OptimizeError as exc:
        message=str(exc)
        code='unresolved_column'
        if 'ambiguous' in message.lower():
            code='ambiguous_column'
        elif 'could not be resolved' in message:
            # Only label ambiguity when multiple *local selected* sources have
            # evidence for the same unqualified name. Unknown != ambiguous.
            for scope in traverse_scope(tree):
                for column in (n for n in scope.walk() if isinstance(n,exp.Column) and not n.table):
                    matches=0
                    for _,source in scope.selected_sources.values():
                        if isinstance(source,exp.Table):
                            matches += column.name.lower() in schema.get(source.name.lower(),{})
                        elif isinstance(source,Scope):
                            matches += column.name.lower() in {n.lower() for n in source.expression.named_selects}
                    if matches>1:
                        code='ambiguous_column'
                        break
        raise GroundingReview(code, message) from exc
    except SqlglotError as exc:
        raise GroundingReview('unsupported_sql', str(exc)) from exc


def resolve_record(cards, row, indices, names, schema, *, strict=False):
    explicit=row.get('positive_columns')
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit or any(not isinstance(n,str) or n.lower() not in names for n in explicit):
            raise GroundingReview('invalid_positive_columns', 'Every explicit positive must exist in the selected database cards')
        positives=sorted({n.lower() for n in explicit})
        if strict and positives != resolve_sql_columns(row.get('SQL',row.get('answer','')),schema):
            raise GroundingReview('positive_sql_mismatch', 'Explicit positives differ from schema-backed SQL evidence')
    else:
        positives=resolve_sql_columns(row.get('SQL',row.get('answer','')),schema)
    missing=set(positives)-names.keys()
    if missing:
        raise GroundingReview('missing_column_cards', f'No cards for SQL evidence: {sorted(missing)}')
    pos=[names[n] for n in positives]
    return dict(query=f"Query: {row['question']} Evidence: {row.get('evidence','')}".strip(),
                positive=pos, negative=[i for i in indices if i not in pos], db_id=row['db_id'])


def build_scope_examples(cards, records, db_dir=None, *, strict=False):
    from src.training.runtime import resolve_db
    cache={}; examples=[]; single_db=len({r['db_id'] for r in records})==1
    for row in records:
        db_id=row['db_id']
        if db_id not in cache:
            path=resolve_db(db_dir,db_id) if db_dir else None
            if strict and path is None:
                raise GroundingReview('missing_database', 'Strict mode requires SQLite databases')
            schema=sqlite_schema(path) if path else None
            indices,names=cards_for_db(cards,db_id,strict=strict,schema=schema,single_db=single_db)
            cache[db_id]=(path,indices,names,schema if schema is not None else schema_from_cards(cards,indices))
        path,indices,names,schema=cache[db_id]
        if strict:
            check=compile_sqlite(path,row.get('SQL',row.get('answer','')))
            if check['status']!='PASS':
                raise GroundingReview(check['code'], 'SQLite compile failed: '+check.get('message',''))
        item=resolve_record(cards,row,indices,names,schema,strict=strict)
        if item['negative']:
            examples.append(item)
    if not examples:
        raise GroundingReview('no_training_examples', 'No examples with both positive and negative columns')
    return examples
