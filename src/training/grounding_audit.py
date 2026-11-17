"""B2.7 offline SQLite/grounding audit and internal BIRD Train group split.

No training, query result execution, input mutation, checkpoint or Hub operations.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import unicodedata

from src.training.sql_grounding import (GroundingReview, cards_for_db, sqlite_schema,
    compile_sqlite, resolve_record, RESOLVER_VERSION)

NORMALIZATION = 'NFKC + casefold + Unicode whitespace collapse/strip; preserve punctuation (v1)'
EXPECTED_SQLGLOT = '26.16.4'


def normalized_question(question):
    if not isinstance(question,str) or not question.strip():
        raise GroundingReview('invalid_question', 'Question must be a nonempty string')
    return ' '.join(unicodedata.normalize('NFKC',question).casefold().split())


def question_key(row):
    if not isinstance(row.get('db_id'),str) or not row['db_id']:
        raise GroundingReview('invalid_record','db_id must be a nonempty string')
    return row['db_id'], normalized_question(row['question'])


def group_id(key):
    return hashlib.sha256(json.dumps(key,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def duplicate_groups(rows):
    groups=defaultdict(list)
    for row in rows:
        try:groups[question_key(row)].append(row)
        except (GroundingReview,KeyError,TypeError):continue
    result=[]
    for key,group in sorted(groups.items()):
        if len(group)<2:continue
        # Exact stripped SQL: never normalize inside SQL literals or choose a gold.
        variants=defaultdict(list)
        for row in group:
            sql=row.get('SQL',row.get('answer',''))
            variants[sql.strip() if isinstance(sql,str) else json.dumps(sql,sort_keys=True)].append(row['source_index'])
        result.append(dict(group_id=group_id(key),db_id=key[0],normalized_question=key[1],
            source_indices=[r['source_index'] for r in group],conflicting_sql=len(variants)>1,
            sql_variants=[dict(sql=sql,source_indices=indices) for sql,indices in sorted(variants.items())]))
    return result


def assert_no_conflicting_gold(records):
    indexed=[dict(row,source_index=i) for i,row in enumerate(records)]
    if any(group['conflicting_sql'] for group in duplicate_groups(indexed)):
        raise ValueError('Conflicting SQL for duplicate questions; review/exclude the whole group before strict training')


def split_groups(rows, *, seed=42, validation_fraction=0.1, excluded_group_ids=()):
    if not 0<validation_fraction<1:
        raise ValueError('validation_fraction must be between 0 and 1')
    excluded=set(excluded_group_ids) | {d['group_id'] for d in duplicate_groups(rows) if d['conflicting_sql']}
    groups=defaultdict(list)
    for row in rows:
        key=question_key(row)
        if group_id(key) not in excluded:groups[key].append(row)
    keys=sorted(groups,key=lambda k:hashlib.sha256(f'{seed}:{group_id(k)}'.encode()).hexdigest())
    if len(keys)<2:
        raise ValueError('Need at least two non-conflicting groups for internal train/validation')
    count=min(len(keys)-1,max(1,math.ceil(len(keys)*validation_fraction)))
    val=set(keys[:count]);train=[];validation=[];manifest=[]
    for key in keys:
        part='internal_validation' if key in val else 'train'
        group=sorted(groups[key],key=lambda r:r['source_index'])
        (validation if key in val else train).extend(group)
        manifest.append(dict(group_id=group_id(key),db_id=key[0],normalized_question=key[1],
                             split=part,source_indices=[r['source_index'] for r in group]))
    assert not {question_key(r) for r in train}&{question_key(r) for r in validation}
    return train,validation,dict(seed=seed,validation_fraction=validation_fraction,
        normalization=NORMALIZATION,split_kind='internal_from_BIRD_Train_NOT_official_dev_test',
        excluded_conflicting_group_ids=sorted(excluded),groups=manifest)


def dump_jsonl(path, rows):
    with Path(path).open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def audit_grounding(data_path, db_root, cards_path, output_dir, *, split=False,
                    seed=42, validation_fraction=0.1, expected_sqlglot=EXPECTED_SQLGLOT):
    import sqlglot
    from src.training.runtime import load_records, resolve_db, dataset_fingerprint
    from src.training.train_embedding import _build_legacy_embedding_examples
    if sqlglot.__version__ != expected_sqlglot:
        raise ValueError(f'Audit requires sqlglot {expected_sqlglot}, installed {sqlglot.__version__}')
    records=load_records(data_path);cards=load_records(cards_path)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)  # Never replace earlier audit/research artifacts.
    inputs=dict(data_sha256=dataset_fingerprint(data_path),cards_sha256=dataset_fingerprint(cards_path))
    cache={}; db_metadata={}; all_question_keys=set(); audited=[];valid=[];recovered=[];quarantine=[];manual=[]
    errors=Counter();breakdown=defaultdict(Counter)
    for index,original in enumerate(records):
        row=dict(original);row['source_index']=index;row['source_record']=original
        db_id=row.get('db_id');sql=row.get('SQL',row.get('answer',''))
        result=dict(source_index=index,db_id=db_id,sqlite=dict(status='FAIL',code='invalid_record'),
                    grounding=dict(status='FAIL',code='invalid_record'),legacy_grounding=dict(status='NOT_EXECUTED'))
        try:
            all_question_keys.add(question_key(row))
            if not isinstance(db_id,str) or not db_id or not isinstance(sql,str) or not sql.strip():
                raise GroundingReview('invalid_record','Need db_id, question and gold SQL')
            if db_id not in cache:
                try:
                    path=resolve_db(db_root,db_id)
                    schema=sqlite_schema(path)
                    db_metadata[db_id]=dict(sha256=dataset_fingerprint(path),
                        schema_sha256=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest())
                    indices,names=cards_for_db(cards,db_id,strict=True,schema=schema)
                    cache[db_id]=(path,schema,indices,names,None)
                except (FileNotFoundError,GroundingReview,OSError,sqlite3.Error) as exc:
                    # SQLite compile can still be evaluated if only cards are bad.
                    path=resolve_db(db_root,db_id) if not isinstance(exc,FileNotFoundError) else None
                    cache[db_id]=(path,None,None,None,exc)
            path,schema,indices,names,problem=cache[db_id]
            result['sqlite']=compile_sqlite(path,sql) if path else dict(status='FAIL',code='missing_database')
            try:
                old=_build_legacy_embedding_examples(cards,[original],db_root)
                result['legacy_grounding']=dict(status='PASS',positive_indices=old[0]['positive'])
            except (ValueError,KeyError,IndexError,FileNotFoundError,sqlglot.errors.SqlglotError) as exc:
                result['legacy_grounding']=dict(status='FAIL',code=type(exc).__name__)
            if problem is not None:
                raise problem
            item=resolve_record(cards,row,indices,names,schema,strict=True)
            positives=sorted(f"{cards[i]['table_name']}.{cards[i]['column_name']}".lower() for i in item['positive'])
            result['grounding']=dict(status='PASS',positive_columns=positives,
                                     negative_count=len(item['negative']))
            row['positive_columns']=positives
            result['training_eligible']=bool(item['negative'])
        except (GroundingReview,FileNotFoundError,KeyError,TypeError,OSError,sqlite3.Error) as exc:
            result['grounding']=dict(status='FAIL',code=getattr(exc,'code','invalid_record'),message=str(exc))
            result['training_eligible']=False
        row['_grounding_audit']=result;audited.append(row)
        db=breakdown[str(db_id)];db['records']+=1
        for key in ('sqlite','grounding'):
            db[key+'_'+result[key]['status'].lower()]+=1
            if result[key]['status']=='FAIL':errors[key+':'+result[key]['code']]+=1
        if result['sqlite']['status']!='PASS':quarantine.append(row)
        elif result['grounding']['status']=='PASS' and result['training_eligible']:
            valid.append(row)
            if result['legacy_grounding']['status']=='FAIL':recovered.append(row)
        if result['grounding']['status']!='PASS' or not result['training_eligible']:
            result['manual_review_reason']=result['grounding'].get('code') or 'no_negative_columns'
            manual.append(row)
    duplicates=duplicate_groups(audited)
    conflicting={d['group_id'] for d in duplicates if d['conflicting_sql']}
    for row in audited:
        try: conflict=group_id(question_key(row)) in conflicting
        except (GroundingReview,KeyError,TypeError):conflict=False
        if conflict:
            row['_grounding_audit']['conflicting_sql']=True
            if row not in manual:
                row['_grounding_audit']['manual_review_reason']='duplicate_conflicting_sql'
                manual.append(row)
    for rows,field in ((valid,'valid_candidates'),(recovered,'recovered'),(quarantine,'quarantined'),(manual,'manual_review')):
        for row in rows:breakdown[str(row.get('db_id'))][field]+=1
    valid_duplicates=duplicate_groups(valid)
    matrix=Counter(r['_grounding_audit']['legacy_grounding']['status']+'__'+r['_grounding_audit']['sqlite']['status'] for r in audited)
    summary=dict(total_records=len(records),sqlglot_version=sqlglot.__version__,sqlite_version=sqlite3.sqlite_version,
        resolver=RESOLVER_VERSION,normalization=NORMALIZATION,inputs=inputs,databases=db_metadata,
        database_breakdown={k:dict(v) for k,v in sorted(breakdown.items())},error_groups=dict(errors),
        sqlite_pass=sum(r['_grounding_audit']['sqlite']['status']=='PASS' for r in audited),
        grounding_pass=sum(r['_grounding_audit']['grounding']['status']=='PASS' for r in audited),
        valid_candidates=len(valid),recovered_count=len(recovered),
        recovery_definition='SQLite PASS + scope_v2 usable labels PASS + current retained legacy builder FAIL; not a claim about the 36 baseline records',
        quarantined_count=len(quarantine),manual_review_count=len(manual),duplicate_groups=len(duplicates),
        duplicate_extra_records=sum(len(d['source_indices'])-1 for d in duplicates),
        legacy_label_sqlite_matrix=dict(matrix),duplicate_scope='all input records; valid_candidate_* is the strict-valid subset',
        valid_candidate_duplicate_groups=len(valid_duplicates),
        valid_candidate_duplicate_extra_records=sum(len(d['source_indices'])-1 for d in valid_duplicates),
        valid_candidate_unique_question_groups=len({question_key(r) for r in valid}),
        valid_candidate_conflicting_sql_groups=sum(d['conflicting_sql'] for d in valid_duplicates),
        conflicting_sql_groups=len(conflicting),unique_question_groups=len(all_question_keys),
        sqlite_fail=sum(r['_grounding_audit']['sqlite']['status']!='PASS' for r in audited),
        grounding_fail=sum(r['_grounding_audit']['grounding']['status']!='PASS' for r in audited))
    summary['source_sha256']={name:dataset_fingerprint(Path(__file__).parent/name)
                             for name in ('grounding_audit.py','sql_grounding.py','train_embedding.py')}
    summary['split_status']='NOT_REQUESTED'
    if split:
        try:
            train,validation,manifest=split_groups(valid,seed=seed,validation_fraction=validation_fraction,excluded_group_ids=conflicting)
            manifest.update(inputs=inputs,resolver=RESOLVER_VERSION,sqlglot_version=sqlglot.__version__)
            dump_jsonl(output/'train.jsonl',train);dump_jsonl(output/'internal_validation.jsonl',validation)
            (output/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            summary.update(train_records=len(train),internal_validation_records=len(validation),split_status='PASS')
        except ValueError as exc:
            summary.update(split_status='FAIL',split_failure=str(exc))
    for name,rows in [('valid_candidates',valid),('recoverable_candidates',recovered),('quarantined_records',quarantine),('manual_review',manual),('duplicate_groups',duplicates)]:
        dump_jsonl(output/(name+'.jsonl'),rows)
    (output/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
