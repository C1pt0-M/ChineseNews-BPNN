"""Verify saved F6 vocabularies, IDF, chi2 scores and selector supports."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
PATHS = {
    'Baseline': ROOT / 'results/member_C/f3_formal_a_reviewed/F_word_char_s42/feature_builder.pkl',
    'F6-A': HERE / 'runs/F6_A_s42/feature_builder.pkl',
    'F6-B': HERE / 'runs/F6_B_s42/feature_builder.pkl',
}


def main():
    target = HERE / 'feature_equivalence.json'
    report = HERE / 'feature_equivalence.md'
    marker = '## 三组特征处理器等同性核验'
    for path in (target, report):
        if path.exists():
            raise FileExistsError(path)
    builders = {}
    for name, path in PATHS.items():
        with path.open('rb') as f:
            builders[name] = pickle.load(f)
    base = builders['Baseline']
    result = {}
    for name, b in builders.items():
        details = {}
        for branch, v0, v, s0, s in [
            ('word', base.word.vectorizer, b.word.vectorizer, base.word.selector, b.word.selector),
            ('char', base.char_vectorizer, b.char_vectorizer, base.char_selector, b.char_selector),
        ]:
            assert v0.vocabulary_ == v.vocabulary_, (name, branch, 'vocabulary')
            np.testing.assert_allclose(v0.idf_, v.idf_, rtol=0, atol=1e-14)
            np.testing.assert_allclose(s0.scores_, s.scores_, rtol=1e-12, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(s0.pvalues_, s.pvalues_, rtol=1e-12, atol=1e-12, equal_nan=True)
            selected, reference = s.get_support(), s0.get_support()
            details[branch] = {
                'vocabulary_size': len(v.vocabulary_), 'selected_size': int(selected.sum()),
                'vocabulary_equal': True, 'idf_equal_atol_1e-14': True,
                'chi2_scores_pvalues_equal_rtol_atol_1e-12': True,
                'support_overlap_with_baseline': int((selected & reference).sum()),
                'support_is_subset_of_baseline': bool((selected & ~reference).sum() == 0),
                'support_is_superset_of_baseline': bool((reference & ~selected).sum() == 0),
            }
        result[name] = {'processor_sha256': sha256(PATHS[name].read_bytes()).hexdigest(), 'branches': details}
    out = {'source': 'saved processors; no fitting or test data used', 'models': result}
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['', marker, '',
             '直接比较三组已保存处理器的词、字词表与索引，均完全一致；IDF 在绝对误差 1e-14 内一致，卡方得分和 p 值在相对/绝对误差 1e-12 内一致（允许对应 NaN）。选择器仅按各组配额改变支持集。三组使用同一训练数据且只拟合训练集的证据见处理器 fit_rows、训练数据哈希与训练代码；本核验不独立重拟合。', '',
             '| 方案 | 词支选择数 | 与基准词支重合 | 字支选择数 | 与基准字支重合 |',
             '|---|---:|---:|---:|---:|']
    for name, item in result.items():
        w, c = item['branches']['word'], item['branches']['char']
        lines.append(f'| {name} | {w["selected_size"]} | {w["support_overlap_with_baseline"]} | {c["selected_size"]} | {c["support_overlap_with_baseline"]} |')
    lines += ['', '完整哈希、词表规模和支持集包含关系见 `feature_equivalence.json`。', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
