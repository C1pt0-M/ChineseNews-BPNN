"""Isolated, resumable 2025 recovery using the project's crawler and cleaner."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from uuid import uuid4
from urllib.robotparser import RobotFileParser

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import config
import crawler
import main

ALLOWED_ROOT = (PROJECT / 'results/member_C/f3_recovery').resolve()


def configure_paths(root: Path) -> None:
    root = root.resolve()
    if root != ALLOWED_ROOT and ALLOWED_ROOT not in root.parents:
        raise ValueError('Recovery output must stay inside results/member_C/f3_recovery')
    config.DATASET_DIR = root / 'data'
    config.STATE_DIR = root / 'state'
    config.RAW_PARTS_DIR = config.STATE_DIR / 'raw_parts'
    config.REPORT_DIR = root / 'reports'
    config.LOG_DIR = root / 'logs'


class RecoveryStopped(RuntimeError):
    pass


def run(root: Path) -> int:
    configure_paths(root)
    for name in ('news_raw.csv', 'news_clean.csv', 'crawl_report.json'):
        if (config.DATASET_DIR / name).exists():
            raise FileExistsError(f'Refusing to overwrite existing recovery output: {config.DATASET_DIR / name}')
    checkpoint = config.STATE_DIR / 'checkpoint.json'
    completed = {}
    if checkpoint.exists():
        completed = json.loads(checkpoint.read_text(encoding='utf-8'))['completed']
        for day, entry in completed.items():
            if entry.get('result') != 'ok' or not (config.RAW_PARTS_DIR / f'{day}.jsonl').is_file():
                raise RecoveryStopped(f'Unsafe or incomplete cached day: {day}; inspect before resuming')
    if config.RAW_PARTS_DIR.exists():
        for part in config.RAW_PARTS_DIR.glob('*.jsonl'):
            if part.stem not in completed:
                raise RecoveryStopped(f'Unregistered cached day: {part}; inspect before resuming')

    main.setup_logging(False)
    actual_session = crawler.make_session
    actual_parse = crawler.parse_archive_page

    def guarded_session():
        session = actual_session()
        original_get = session.get
        def checked_get(url, *args, **kwargs):
            response = original_get(url, *args, **kwargs)
            if response.status_code in (401, 403, 429):
                raise RecoveryStopped(f'Site returned HTTP {response.status_code}; stop without retrying or bypassing restrictions')
            return response
        session.get = checked_get
        return session

    def guarded_parse(*args, **kwargs):
        parsed = actual_parse(*args, **kwargs)
        if not parsed.container_found or parsed.has_pagination or not parsed.records:
            raise RecoveryStopped('Archive parser returned missing container, pagination, or no records; inspect before resuming')
        return parsed

    crawler.make_session = guarded_session
    crawler.parse_archive_page = guarded_parse
    original_fetch = crawler.fetch_archive_page
    def guarded_fetch(*args, **kwargs):
        result = original_fetch(*args, **kwargs)
        if not result.ok:
            raise RecoveryStopped(f'Archive request failed: {result.url}: status={result.status}, error={result.error}')
        return result
    crawler.fetch_archive_page = guarded_fetch

    try:
        session = guarded_session()
        robots_response = session.get('https://www.chinanews.com/robots.txt', timeout=config.TIMEOUT)
        robots_response.raise_for_status()
        parser = RobotFileParser()
        parser.parse(robots_response.text.splitlines())
        if not parser.can_fetch(config.REQUEST_HEADERS['User-Agent'], crawler.build_archive_url(date(2025, 1, 1))):
            raise RecoveryStopped('robots.txt disallows the archive path')
        root.mkdir(parents=True, exist_ok=True)
        robots_path = root / 'robots.txt'
        if robots_path.exists():
            if robots_path.read_text(encoding='utf-8') != robots_response.text:
                raise RecoveryStopped('robots.txt changed since the previous attempt; inspect before resuming')
        else:
            robots_path.write_text(robots_response.text, encoding='utf-8')
        args = argparse.Namespace(start=date(2025, 1, 1), end=date(2025, 12, 31),
                                  delay=config.DEFAULT_DELAY, no_resume=False,
                                  outdir=str(config.DATASET_DIR), reportdir=str(config.REPORT_DIR),
                                  max_per_class=0)
        code = main.cmd_crawl(args)
        if code:
            return code
        return main.cmd_clean(args)
    except Exception as exc:
        root.mkdir(parents=True, exist_ok=True)
        (root / f'interruption_{uuid4().hex}.json').write_text(
            json.dumps({'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        raise
    finally:
        crawler.make_session = actual_session
        crawler.parse_archive_page = actual_parse
        crawler.fetch_archive_page = original_fetch


if __name__ == '__main__':
    raise SystemExit(run(ALLOWED_ROOT))
