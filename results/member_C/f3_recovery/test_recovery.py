import importlib.util
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import config
import crawler
import main

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recover_2025', HERE / 'recover_2025.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
audit_spec = importlib.util.spec_from_file_location('audit_recovery', HERE / 'audit_recovery.py')
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


class Response:
    status_code = 200
    text = 'User-agent: *\nAllow: /\n'
    def raise_for_status(self):
        pass


class Session:
    def get(self, *args, **kwargs):
        return Response()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.original = {k: getattr(config, k) for k in ('DATASET_DIR', 'STATE_DIR', 'RAW_PARTS_DIR', 'REPORT_DIR', 'LOG_DIR')}
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        for key, val in self.original.items():
            setattr(config, key, val)
        self.tmp.cleanup()

    def test_paths_are_isolated_and_crawl_is_bounded_and_uncapped(self):
        def crawl(args):
            self.assertEqual((args.start, args.end), (date(2025, 1, 1), date(2025, 12, 31)))
            self.assertEqual(args.delay, config.DEFAULT_DELAY)
            self.assertEqual(args.max_per_class, 0)
            self.assertEqual(Path(args.outdir), self.root / 'data')
            main.write_part(date(2025, 1, 1), [], {'result': 'ok'})
            main.save_checkpoint({'completed': {'2025-01-01': {'result': 'ok'}}})
            return 0
        with patch.object(crawler, 'make_session', return_value=Session()), patch.object(
                main, 'setup_logging', side_effect=lambda _: config.ensure_dirs()), patch.object(
                main, 'cmd_crawl', side_effect=crawl), patch.object(main, 'cmd_clean', return_value=0) as clean:
            self.assertEqual(recovery.run(self.root), 0)
            clean.assert_called_once()
        self.assertTrue((self.root / 'state/checkpoint.json').is_file())
        self.assertTrue((self.root / 'state/raw_parts/2025-01-01.jsonl').is_file())
        self.assertTrue((self.root / 'robots.txt').is_file())
        self.assertNotEqual(config.STATE_DIR, self.original['STATE_DIR'])

    def test_existing_data_are_not_overwritten(self):
        (self.root / 'data').mkdir()
        target = self.root / 'data/news_raw.csv'
        target.write_text('preserve', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            recovery.run(self.root)
        self.assertEqual(target.read_text(encoding='utf-8'), 'preserve')

    def test_unregistered_cache_is_not_overwritten(self):
        part = self.root / 'state/raw_parts/2025-01-01.jsonl'
        part.parent.mkdir(parents=True)
        part.write_text('preserve', encoding='utf-8')
        with self.assertRaises(recovery.RecoveryStopped):
            recovery.run(self.root)
        self.assertEqual(part.read_text(encoding='utf-8'), 'preserve')

    def test_disallowed_robots_stops_before_crawl(self):
        response = Response()
        response.text = 'User-agent: *\nDisallow: /scroll-news/\n'
        session = Session()
        session.get = lambda *a, **k: response
        with patch.object(crawler, 'make_session', return_value=session), patch.object(
                main, 'setup_logging', side_effect=lambda _: config.ensure_dirs()), patch.object(main, 'cmd_crawl') as crawl:
            with self.assertRaises(recovery.RecoveryStopped):
                recovery.run(self.root)
            crawl.assert_not_called()

    def test_http_403_stops_without_retry_or_cleaning(self):
        class BlockedSession:
            def __init__(self):
                self.calls = []
            def get(self, url, *args, **kwargs):
                self.calls.append(url)
                response = Response()
                response.status_code = 403 if '/scroll-news/' in url else 200
                return response
        session = BlockedSession()
        def crawl(_args):
            crawler.make_session().get('https://www.chinanews.com/scroll-news/2025/0101/news.shtml')
        with patch.object(crawler, 'make_session', return_value=session), patch.object(
                main, 'setup_logging', side_effect=lambda _: config.ensure_dirs()), patch.object(
                main, 'cmd_crawl', side_effect=crawl), patch.object(main, 'cmd_clean') as clean:
            with self.assertRaises(recovery.RecoveryStopped):
                recovery.run(self.root)
            clean.assert_not_called()
        self.assertEqual(sum('/scroll-news/' in x for x in session.calls), 1)

    def test_refuses_paths_outside_recovery(self):
        with self.assertRaises(ValueError):
            recovery.configure_paths(recovery.PROJECT / 'dataset')

    def test_near_index_uses_normalized_trigram_jaccard(self):
        idx = audit.NearIndex()
        idx.add(audit.title_key('ＡＢＣ，defghij'))
        self.assertEqual(idx.matches(audit.title_key('abcdefghij')), [(0, 1.0)])
        self.assertEqual(idx.matches(audit.title_key('unrelated title')), [])

    def test_audit_keeps_base_and_flags_reference_and_candidate_near_duplicates(self):
        data, splits, out = self.root / 'data', self.root / 'dataset/splits', self.root / 'analysis'
        data.mkdir(parents=True)
        splits.mkdir(parents=True)
        label = config.TARGET_CATEGORIES[0]
        def row(i, title):
            return {'id': str(i), 'title': title, 'url': f'https://example.com/{i}',
                    'publish_time': '2025-01-01 12:00:00', 'archive_date': '2025-01-01',
                    'category': label, 'raw_category': label, 'source': 'test'}
        base = row(1, 'abcdefghijklmnop')
        base_rows = [base]
        for i, cat in enumerate(config.TARGET_CATEGORIES[1:], 101):
            base_rows.append(dict(row(i, 'reference unique category ' + str(i)), category=cat))
        values = [base, row(2, 'abcdefghijklmnoq'), row(3, 'qrstuvwxyzabcdef'),
                  row(4, 'qrstuvwxyzabcdeg'), row(5, 'extrauniquenewstitle')]
        for name, records in [('train', base_rows), ('val', [row(90, 'distinctvalidation')]),
                              ('test', [row(91, 'differentheldouttitle')])]:
            pd.DataFrame(records).to_csv(splits / f'{name}.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(values).to_csv(data / 'news_clean.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([dict(x, _time_issue='') for x in values]).to_csv(data / 'news_raw.csv', index=False, encoding='utf-8-sig')
        (data / 'crawl_report.json').write_text(json.dumps({'days_requested': 365, 'summary': {'days_success': 365},
            'date_start': '2025-01-01', 'date_end': '2025-12-31'}), encoding='utf-8')
        version = {'files': {f'{name}.csv': {'sha256': hashlib.sha256((splits / f'{name}.csv').read_bytes()).hexdigest()}
                             for name in ('train', 'val', 'test')}}
        (self.root / 'dataset/dataset_version.json').write_text(json.dumps(version), encoding='utf-8')
        with patch.object(audit, 'ROOT', self.root), patch.object(audit, 'DATA', data), patch.object(
                audit, 'SPLITS', splits), patch.object(audit, 'OUT', out):
            audit.main()
        result = json.loads((out / 'audit_results.json').read_text(encoding='utf-8'))
        self.assertEqual(result['exact_overlap_rows']['train']['any'], 1)
        self.assertEqual(result['near_reference_rows']['train'], 1)
        self.assertEqual(result['internal_near_pairs'], 1)
        self.assertEqual(result['conservative_candidate_count'], 1)
        self.assertEqual(pd.read_csv(out / 'candidate_manifest.csv')['id'].tolist(), [5])
        self.assertEqual(len(pd.read_csv(splits / 'train.csv')), 6)


if __name__ == '__main__':
    unittest.main()
