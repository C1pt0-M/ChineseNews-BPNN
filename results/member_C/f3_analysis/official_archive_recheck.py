"""Re-fetch and preserve official archive pages, with conservative interruption rules."""
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config
import crawler

OUT = ROOT / 'results/member_C/f3_analysis/official_archive_recheck'


def write_exclusive(path, data):
    with path.open('xb') as f:
        f.write(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = crawler.make_session()
    robots = session.get('https://www.chinanews.com/robots.txt', timeout=config.TIMEOUT)
    robots.raise_for_status()
    if urlsplit(robots.url).hostname != 'www.chinanews.com':
        raise RuntimeError('Unexpected robots.txt redirect')
    robot_data = robots.content
    robot_path = OUT / 'robots.txt'
    if not robot_path.exists():
        write_exclusive(robot_path, robot_data)
    elif robot_path.read_bytes() != robot_data:
        raise RuntimeError('robots.txt changed; inspect before resuming')
    parser = RobotFileParser()
    parser.parse(robots.text.splitlines())
    for i in range(365):
        day = date(2025, 1, 1) + timedelta(days=i)
        tag = day.isoformat()
        url = crawler.build_archive_url(day)
        if not parser.can_fetch(config.REQUEST_HEADERS['User-Agent'], url):
            raise RuntimeError(f'robots.txt disallows {url}')
        page, meta = OUT / (tag + '.html.gz'), OUT / (tag + '.json')
        if page.exists() and meta.exists():
            m = json.loads(meta.read_text(encoding='utf-8'))
            if sha256(gzip.decompress(page.read_bytes())).hexdigest() != m['html_sha256']:
                raise RuntimeError(f'Cached page hash mismatch: {tag}')
            continue
        if page.exists() or meta.exists():
            raise RuntimeError(f'Incomplete existing page pair: {tag}')
        response = session.get(url, timeout=config.TIMEOUT)
        if response.status_code in (401, 403, 429):
            raise RuntimeError(f'Access or rate limit: HTTP {response.status_code} for {url}')
        response.raise_for_status()
        if urlsplit(response.url).hostname != 'www.chinanews.com':
            raise RuntimeError(f'Unexpected archive redirect: {response.url}')
        content = response.content
        parsed = crawler.parse_archive_page(content.decode('utf-8'), day, url)
        if not parsed.container_found or parsed.has_pagination or not parsed.records:
            raise RuntimeError(f'Archive structure is incomplete or changed: {url}')
        payload = gzip.compress(content, mtime=0)
        record = {'date': tag, 'requested_url': url, 'final_url': response.url,
                  'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                  'http_status': response.status_code, 'html_sha256': sha256(content).hexdigest(),
                  'records': len(parsed.records), 'pagination': parsed.has_pagination}
        write_exclusive(page, payload)
        write_exclusive(meta, json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'))
        if (i + 1) % 20 == 0 or i == 0:
            print(f'official pages {i+1}/365: {tag}, records={len(parsed.records)}', flush=True)
        time.sleep(config.DEFAULT_DELAY + 0.15)
    print('Official archive retrieval complete.', flush=True)


if __name__ == '__main__':
    main()
