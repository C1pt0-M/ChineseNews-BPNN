import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('official_evidence', Path(__file__).with_name('compile_official_evidence.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OfficialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'cn123', 'url': 'https://www.chinanews.com/edu/2025/01-03/123.shtml',
                    'title': '某学校 开学', 'category': '教育', 'raw_category': '教育',
                    'publish_time': '2025-01-03 12:34:00', 'archive_date': '2025-01-03'}
        self.official = dict(self.row, _raw_time='1-3 12:34', _time_issue='')

    def test_matching_fields_and_format_whitespace(self):
        record = dict(self.official, title=' 某学校  开学 ')
        matches, conflicts = module.check_record(self.row, [record], '2025-01-03')
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(conflicts), 0)

    def test_conflicts_never_qualify(self):
        for update in ({'id': 'different'}, {'url': 'https://else.invalid/x'},
                       {'title': '不同标题'}, {'category': '社会'},
                       {'_raw_time': ''}, {'publish_time': '2025-01-03 12:35:00'},
                       {'archive_date': '2025-01-04'}):
            with self.subTest(update=update):
                matches, conflicts = module.check_record(self.row, [dict(self.official, **update)], '2025-01-03')
                self.assertEqual(matches, [])
                self.assertEqual(len(conflicts), 1)


if __name__ == '__main__':
    unittest.main()
