"""B2.7 fixtures: exact SQL labels and read-only SQLite, no external dataset."""
from contextlib import closing
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch, Mock
from pathlib import Path
import sqlite3
import tempfile
import unittest

try:
    import sqlglot
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

from src.training.sql_grounding import GroundingReview,sqlite_schema,compile_sqlite,resolve_sql_columns
from src.training.grounding_audit import (audit_grounding,normalized_question,question_key,
                                         duplicate_groups,split_groups)

A="""SELECT T2.movie_release_year, T1.user_id FROM ratings AS T1
INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_id = (
SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino'
ORDER BY movie_release_year ASC LIMIT 2, 1) AND T1.rating_score = 4"""
B="""SELECT T3.stor_id, T2.title FROM sales AS T1 INNER JOIN titles AS T2
ON T1.title_id = T2.title_id INNER JOIN stores AS T3 ON T3.stor_id = T1.stor_id
WHERE T3.stor_id = (SELECT stor_id FROM sales GROUP BY stor_id
ORDER BY SUM(qty) DESC LIMIT 1) GROUP BY T3.stor_id, T2.title
ORDER BY SUM(T1.qty) ASC LIMIT 1"""
C="SELECT 'voice-actor' FROM `voice-actors` GROUP BY 'voice-actor' HAVING COUNT(movie) > 5"


@unittest.skipUnless(AVAILABLE,'Pinned SQLGlot required for scope audit fixtures')
class TestSQLGroundingAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'db.sqlite'
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.executescript('''CREATE TABLE t(id INTEGER, name TEXT, flag INTEGER);
                CREATE TABLE u(id INTEGER, value TEXT);
                CREATE TABLE ratings(movie_id INT,user_id INT,rating_score INT);
                CREATE TABLE movies(movie_id INT,director_name TEXT,movie_release_year INT);
                CREATE TABLE sales(title_id INT,stor_id INT,qty INT);
                CREATE TABLE titles(title_id INT,title TEXT);
                CREATE TABLE stores(stor_id INT,address TEXT);
                CREATE TABLE "voice-actors"("voice-actor" TEXT,movie TEXT);
                INSERT INTO t VALUES (1,'same',1);''')
        self.schema=sqlite_schema(self.path)
        self.cards=[dict(db_id='db',table_name=t,column_name=c) for t,cols in self.schema.items() for c in cols]
        self.cards.append(dict(db_id='other',table_name='t',column_name='id'))

    def assertLabels(self,sql,expected):
        self.assertEqual(compile_sqlite(self.path,sql)['status'],'PASS',sql)
        self.assertEqual(resolve_sql_columns(sql,self.schema),sorted(expected))

    def test_schema_normalization_collision_is_not_guessed(self):
        path=self.root/'collision.sqlite'
        with closing(sqlite3.connect(path)) as conn,conn:
            conn.execute('CREATE TABLE t("Ü" TEXT, "ü" TEXT)')
        with self.assertRaises(GroundingReview) as raised:sqlite_schema(path)
        self.assertEqual(raised.exception.code,'schema_collision')

    def test_simple_and_single_table(self):
        self.assertLabels('SELECT id, name FROM t WHERE flag=1',['t.id','t.name','t.flag'])
        self.assertLabels('SELECT t.id FROM t',['t.id'])

    def test_join_qualified_aliases(self):
        self.assertLabels('SELECT a.name,b.value FROM t a JOIN u b ON a.id=b.id',
                          ['t.name','u.value','t.id','u.id'])

    def test_movie_regression_original_index_25(self):
        self.assertLabels(A,['movies.movie_release_year','movies.movie_id','movies.director_name',
                             'ratings.movie_id','ratings.user_id','ratings.rating_score'])

    def test_sales_regression_original_index_177(self):
        self.assertLabels(B,['stores.stor_id','titles.title','sales.title_id','titles.title_id','sales.stor_id','sales.qty'])

    def test_disney_regression_original_index_4733_literal(self):
        self.assertLabels(C,['voice-actors.movie'])

    def test_subquery_alias_shadowing(self):
        self.assertLabels('SELECT a.name FROM t a WHERE EXISTS(SELECT a.id FROM u a WHERE a.value=\'x\')',
                          ['t.name','u.id','u.value'])

    def test_correlated_qualified_and_unqualified(self):
        self.assertLabels('SELECT t.name FROM t WHERE EXISTS(SELECT 1 FROM u WHERE u.id=t.id)',
                          ['t.name','t.id','u.id'])
        self.assertLabels('SELECT t.id FROM t WHERE EXISTS(SELECT 1 FROM u WHERE name=\'x\' AND u.id=t.id)',
                          ['t.id','t.name','u.id'])

    def test_cte_and_renamed_output(self):
        self.assertLabels('WITH q(x) AS (SELECT id FROM t) SELECT q.x FROM q',['t.id'])
        self.assertLabels('WITH q AS (SELECT id, COUNT(flag) AS n FROM t GROUP BY id) SELECT q.n FROM q',
                          ['t.id','t.flag'])

    def test_derived_alias_never_becomes_physical_table(self):
        self.assertLabels('SELECT t.x FROM (SELECT u.id AS x FROM u) t',['u.id'])
        with self.assertRaises(GroundingReview):
            resolve_sql_columns('SELECT d.missing FROM (SELECT id FROM t) d',self.schema)

    def test_union_and_nested_derived(self):
        self.assertLabels('SELECT d.id FROM (SELECT id FROM t UNION SELECT id FROM u) d', ['t.id','u.id'])

    def test_true_ambiguous_not_guessed(self):
        sql='SELECT id FROM t JOIN u ON t.id=u.id'
        self.assertEqual(compile_sqlite(self.path,sql)['code'],'ambiguous_column')
        with self.assertRaises(GroundingReview):resolve_sql_columns(sql,self.schema)

    def test_unqualified_does_not_search_entire_database(self):
        with self.assertRaises(GroundingReview):resolve_sql_columns('SELECT movie_id FROM t',self.schema)

    def test_projection_stars_and_count(self):
        self.assertLabels('SELECT * FROM t',['t.id','t.name','t.flag'])
        self.assertLabels('SELECT a.* FROM t a JOIN u b ON a.id=b.id',['t.id','t.name','t.flag','u.id'])
        self.assertLabels('SELECT COUNT(*) FROM t WHERE flag=1',['t.flag'])
        for sql in ['SELECT COUNT(*) FROM t',"SELECT 'name' FROM t",'SELECT d.n FROM (SELECT COUNT(*) AS n FROM t) d']:
            with self.subTest(sql=sql),self.assertRaisesRegex(GroundingReview,'No physical column'):
                resolve_sql_columns(sql,self.schema)

    def test_string_literal_and_projection_alias(self):
        self.assertLabels("SELECT 'id', name AS alias FROM t ORDER BY alias",['t.name'])
        self.assertLabels('SELECT "name" FROM t',['t.name'])

    def test_missing_tables_columns_and_syntax(self):
        for sql,code in [('SELECT id FROM missing','missing_table'),
                         ('SELECT missing FROM t','missing_column'),('SELECT FROM t','syntax_error')]:
            with self.subTest(sql=sql):
                self.assertEqual(compile_sqlite(self.path,sql)['code'],code)
                with self.assertRaises(GroundingReview):resolve_sql_columns(sql,self.schema)

    def test_explicit_annotations_validate_all_not_partial(self):
        from src.training.train_embedding import build_embedding_examples
        row=dict(db_id='db',question='Get IDs',SQL='SELECT id FROM t')
        for positive in [['t.id','t.missing'],['other.id'],[],['u.id']]:
            with self.subTest(positive=positive),self.assertRaises(GroundingReview):
                build_embedding_examples(self.cards,[dict(row,positive_columns=positive)],self.root,label_mode='strict')
        item=build_embedding_examples(self.cards,[dict(row,positive_columns=['t.id'])],self.root,label_mode='strict')[0]
        self.assertEqual([self.cards[i]['column_name'] for i in item['positive']],['id'])
        self.assertTrue(all(self.cards[i]['db_id']=='db' for i in item['positive']+item['negative']))

    def test_strict_requires_tagged_cards_and_real_database(self):
        from src.training.train_embedding import build_embedding_examples
        row=dict(db_id='db',question='Get IDs',SQL='SELECT id FROM t')
        cards=[{k:v for k,v in c.items() if k!='db_id'} for c in self.cards[:-1]]
        with self.assertRaises(GroundingReview):build_embedding_examples(cards,[row],self.root,label_mode='strict')
        with self.assertRaises(GroundingReview):build_embedding_examples(self.cards,[row],label_mode='strict')
        stale=self.cards+[dict(db_id='db',table_name='t',column_name='missing')]
        with self.assertRaises(GroundingReview):build_embedding_examples(stale,[row],self.root,label_mode='strict')

    def test_read_only_compile_no_query_results_or_mutation(self):
        before=self.path.read_bytes()
        self.assertEqual(compile_sqlite(self.path,'SELECT randomblob(1000000000) FROM t')['status'],'PASS')
        self.assertEqual(compile_sqlite(self.path,'DELETE FROM t')['status'],'FAIL')
        self.assertEqual(compile_sqlite(self.path,'WITH q AS (SELECT id FROM t) DELETE FROM t')['status'],'FAIL')
        self.assertEqual(compile_sqlite(self.path,'SELECT id FROM t; DROP TABLE t')['status'],'FAIL')
        self.assertEqual(self.path.read_bytes(),before)

    def write_inputs(self,rows):
        data=self.root/'train.json';cards=self.root/'cards.jsonl'
        data.write_text(json.dumps(rows))
        cards.write_text(''.join(json.dumps(c)+'\n' for c in self.cards))
        return data,cards

    def test_audit_quarantines_sqlite_invalid_even_with_labels(self):
        rows=[dict(db_id='db',question=q,SQL=sql) for q,sql in [('ok','SELECT id FROM t'),
             ('bad aggregate','SELECT id FROM t WHERE SUM(id)>0'),('missing','SELECT id FROM missing'),
             ('constant','SELECT COUNT(*) FROM t'),('ambiguous','SELECT id FROM t JOIN u ON t.id=u.id')]]
        data,cards=self.write_inputs(rows);before=self.path.read_bytes();out=self.root/'audit'
        result=audit_grounding(data,self.root,cards,out)
        self.assertEqual(result['total_records'],5)
        self.assertEqual(result['sqlite_pass'],2)
        self.assertEqual(result['valid_candidates'],1)
        self.assertEqual(result['quarantined_count'],3)
        valid=[json.loads(s) for s in (out/'valid_candidates.jsonl').read_text().splitlines()]
        self.assertEqual([r['source_index'] for r in valid],[0])
        self.assertEqual(valid[0]['positive_columns'],['t.id'])
        quarantined=[json.loads(s) for s in (out/'quarantined_records.jsonl').read_text().splitlines()]
        bad=next(r for r in quarantined if r['source_index']==1)
        self.assertEqual(bad['_grounding_audit']['grounding']['status'],'PASS')
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(json.loads(data.read_text()),rows)
        self.assertEqual(result['inputs']['data_sha256'],hashlib.sha256(data.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):audit_grounding(data,self.root,cards,out)

    def test_recovered_counts_actual_legacy_comparison(self):
        data,cards=self.write_inputs([dict(db_id='db',question='movie',SQL=A),dict(db_id='db',question='sales',SQL=B)])
        result=audit_grounding(data,self.root,cards,self.root/'recovery')
        self.assertEqual(result['recovered_count'],2)
        self.assertEqual(result['valid_candidates'],2)

    def test_full_audit_conflicts_preserved_and_excluded_from_split(self):
        rows=[dict(db_id='db',question=q,SQL=sql) for q,sql in [('  Same  Question ','SELECT id FROM t'),
            ('same question','SELECT name FROM t'),('other','SELECT flag FROM t'),('third','SELECT value FROM u')]]
        data,cards=self.write_inputs(rows);out=self.root/'split'
        result=audit_grounding(data,self.root,cards,out,split=True)
        self.assertEqual(result['duplicate_groups'],1)
        self.assertEqual(result['conflicting_sql_groups'],1)
        self.assertEqual(result['manual_review_count'],2)
        manifest=json.loads((out/'split_manifest.json').read_text())
        self.assertEqual(sorted(i for g in manifest['groups'] for i in g['source_indices']),[2,3])
        dup=json.loads((out/'duplicate_groups.jsonl').read_text())
        self.assertEqual(dup['source_indices'],[0,1])
        self.assertEqual(len(dup['sql_variants']),2)

    def test_notebook_full_mode_prepares_without_training(self):
        rows=[dict(db_id='db',question=q,SQL='SELECT id FROM t') for q in ('first','second')]
        data,cards=self.write_inputs(rows)
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/'notebooks/01_data_preparation_and_embedding.ipynb').read_text())
        code=''.join(notebook['cells'][4]['source'])
        output=self.root/'notebook';output.mkdir()
        paths=SimpleNamespace(device='cpu',input=lambda name,relative:os.environ[name])
        with patch.dict(os.environ,GROUNDING_FULL_DATASET='1',GROUNDING_ALLOW_FULL_TRAINING='0',
                        TRAIN_DATA=str(data),DB_DIR=str(self.root),CARDS_PATH=str(cards),
                        GROUNDING_AUDIT_DIR=str(self.root/'nb-audit')), \
             patch('src.training.train_embedding.run_embedding_training') as train, \
             patch('src.training.train_embedding.run_dry_run') as smoke:
            namespace=dict(PROFILE='LEGACY',os=os,Path=Path,json=json,paths=paths,OUTPUT_DIR=str(output),SMOKE=False,
                           SEED=42,RESUME='none',MAX_STEPS=2,MAX_LENGTH=8,SAVE_STEPS=1)
            exec(compile(code,'notebook01-cell4','exec'),namespace)
            train.assert_not_called();smoke.assert_not_called()
            self.assertFalse(namespace['summary']['trained'])
            self.assertEqual(namespace['audit_summary']['split_status'],'PASS')
            self.assertEqual(namespace['label_options'],{'grounding_label_mode':'strict'})

    def test_notebook_prepare_skips_checkpoint_writer(self):
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/'notebooks/01_data_preparation_and_embedding.ipynb').read_text())
        paths=SimpleNamespace(device='cpu',output=lambda phase:str(self.root))
        with patch.dict(os.environ,GROUNDING_FULL_DATASET='1',GROUNDING_ALLOW_FULL_TRAINING='0'), \
             patch('src.training.kaggle.initialize_environment',return_value={}), \
             patch('src.training.kaggle.environment_report',return_value={'gpus':[]}) as report, \
             patch('src.training.kaggle.initialize_checkpoint_backend') as backend:
            namespace=dict(PROFILE='LEGACY',os=os,Path=Path,paths=paths,SMOKE=False,REPO_DIR=root,environment_report=report)
            exec(compile(''.join(notebook['cells'][2]['source']),'notebook01-cell2','exec'),namespace)
            backend.assert_not_called()
            self.assertIsNone(namespace['checkpoint_backend'])

    def test_split_fail_still_writes_audit_and_diagnostics(self):
        data,cards=self.write_inputs([dict(db_id='db',question='one',SQL='SELECT id FROM t')])
        out=self.root/'small'
        result=audit_grounding(data,self.root,cards,out,split=True)
        self.assertEqual(result['split_status'],'FAIL')
        self.assertEqual(result['valid_candidates'],1)
        self.assertTrue((out/'audit_summary.json').exists())
        self.assertFalse((out/'train.jsonl').exists())

    def test_compile_stale_cards_and_missing_db_report_separately(self):
        rows=[dict(db_id='db',question='q',SQL='SELECT id FROM t'),
              dict(db_id='absent',question='q',SQL='SELECT id FROM t')]
        data,cards=self.write_inputs(rows)
        with cards.open('a') as f:f.write(json.dumps(dict(db_id='db',table_name='t',column_name='absent'))+'\n')
        result=audit_grounding(data,self.root,cards,self.root/'bad-schema')
        self.assertEqual(result['sqlite_pass'],1)
        self.assertEqual(result['grounding_pass'],0)
        self.assertEqual(result['error_groups']['grounding:card_schema_mismatch'],1)
        self.assertEqual(result['error_groups']['sqlite:missing_database'],1)

    def test_natural_join_is_manual_not_inferred(self):
        with self.assertRaises(GroundingReview) as raised:
            resolve_sql_columns('SELECT t.id FROM t NATURAL JOIN u',self.schema)
        self.assertEqual(raised.exception.code,'unsupported_join')

    def test_runner_rejects_normalized_eval_overlap_before_model_load(self):
        from src.training import train_embedding as embedding
        rows=[dict(db_id='db',question='Same  question',SQL='SELECT id FROM t')]
        data,cards=self.write_inputs(rows)
        evaluation=self.root/'evaluation.json'
        evaluation.write_text(json.dumps([dict(rows[0],question=' SAME\tQUESTION ')]))
        model=Mock(side_effect=AssertionError('Model must not load'))
        with patch.dict(sys.modules,sentence_transformers=SimpleNamespace(SentenceTransformer=model)), \
             patch.object(embedding,'TORCH_AVAILABLE',True), \
             patch('src.training.runtime.seed_everything'):
            with self.assertRaisesRegex(ValueError,'overlap'):
                embedding.run_embedding_training.__wrapped__(cards_path=cards,bird_path=data,
                    eval_path=evaluation,db_dir=self.root,output_dir=self.root/'training',device='cpu')
        model.assert_not_called()

    def test_pin_mismatch_fails_before_output(self):
        data,cards=self.write_inputs([dict(db_id='db',question='q',SQL='SELECT id FROM t')])
        with self.assertRaises(ValueError):audit_grounding(data,self.root,cards,self.root/'never',expected_sqlglot='0.0')
        self.assertFalse((self.root/'never').exists())


class TestGroundingDuplicates(unittest.TestCase):
    def test_normalization_fixed(self):
        self.assertEqual(normalized_question('  Ｈｅｌｌｏ\tWORLD\n? '),'hello world ?')
        self.assertEqual(normalized_question('Straße'),'strasse')
        self.assertNotEqual(normalized_question('Q?'),normalized_question('Q'))

    def test_deterministic_group_split_no_overlap(self):
        rows=[dict(db_id='db',question=q,SQL='SELECT id FROM t',source_index=i)
              for i,q in enumerate([' SAME question ','same\tquestion','a','b','c','d'])]
        train,val,manifest=split_groups(rows,seed=42)
        self.assertFalse({question_key(r) for r in train}&{question_key(r) for r in val})
        self.assertEqual((train,val,manifest),split_groups(list(reversed(rows)),seed=42))
        dup=duplicate_groups(rows)
        self.assertEqual(len(dup),1);self.assertFalse(dup[0]['conflicting_sql'])
        self.assertEqual(dup[0]['source_indices'],[0,1])
        assigned=[g for g in manifest['groups'] if len(g['source_indices'])==2]
        self.assertEqual(len(assigned),1)

    def test_split_directly_excludes_conflicting_gold(self):
        rows=[dict(db_id='db',question=q,SQL=sql,source_index=i) for i,(q,sql) in enumerate(
            [('same','SELECT id FROM t'),('SAME','SELECT name FROM t'),('a','SELECT 1'),('b','SELECT 2')])]
        train,val,manifest=split_groups(rows)
        self.assertEqual(sorted(r['source_index'] for r in train+val),[2,3])
        self.assertEqual(len(manifest['excluded_conflicting_group_ids']),1)

    def test_strict_conflicting_gold_guard(self):
        from src.training.grounding_audit import assert_no_conflicting_gold
        rows=[dict(db_id='db',question=q,SQL=sql) for q,sql in
              [('same','SELECT id FROM t'),('SAME','SELECT name FROM t')]]
        with self.assertRaisesRegex(ValueError,'Conflicting SQL'):assert_no_conflicting_gold(rows)
        assert_no_conflicting_gold([dict(r,SQL='SELECT id FROM t') for r in rows])

    def test_same_question_different_db_not_duplicate(self):
        rows=[dict(db_id=d,question='q',SQL='SELECT 1',source_index=i) for i,d in enumerate(['a','b'])]
        self.assertEqual(duplicate_groups(rows),[])

