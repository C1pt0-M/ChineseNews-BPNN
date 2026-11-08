"""Flag weaker lexical matches around the same date for manual event review."""
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('prior_audit', ROOT/'results/member_C/f3_recovery/audit_recovery.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main(dataset_name='f3_a_dataset', output_name='f3_a_posttrain_event_flags.csv'):
    selected = audit.enrich(audit.read(HERE/dataset_name/'new_sample_evidence.csv')).reset_index(drop=True)
    train = audit.enrich(audit.read(ROOT/'dataset/splits/train.csv')).reset_index(drop=True)
    hits = []
    def scan(ref, label, include_self):
        idx = audit.NearIndex()
        for title in ref._title_key:
            idx.add(title)
        ref_days = [datetime.fromisoformat(x[:10]).date().toordinal() for x in ref.publish_time]
        for _, row in selected.iterrows():
            g = audit.grams(row['_title_key'])
            counts = Counter(p for gram in g for p in idx.index.get(gram, ()))
            day = datetime.fromisoformat(row.publish_time[:10]).date().toordinal()
            for j, n in counts.items():
                if include_self and (row.id >= ref.iloc[j].id):
                    continue
                if abs(day-ref_days[j]) > 7:
                    continue
                jac = n/(len(g)+len(idx.values[j])-n)
                if jac < .25:
                    continue
                other = ref.iloc[j]
                ratio = SequenceMatcher(None,row['_title_key'],other['_title_key']).ratio()
                if ratio >= .65:
                    hits.append({'candidate_id':row.id,'candidate_title':row.title,'candidate_date':row.publish_time[:10],
                                 'reference_type':label,'reference_id':other.id,'reference_title':other.title,
                                 'reference_date':other.publish_time[:10],'trigram_jaccard':jac,'sequence_ratio':ratio})
    scan(train,'baseline',False)
    scan(selected,'new',True)
    columns=['candidate_id','candidate_title','candidate_date','reference_type','reference_id','reference_title','reference_date','trigram_jaccard','sequence_ratio']
    output = HERE/output_name
    if output.exists():
        raise FileExistsError(output)
    pd.DataFrame(hits,columns=columns).to_csv(output,index=False,encoding='utf-8-sig')
    print('weaker-lexical flags:',len(hits),flush=True)


if __name__=='__main__':
    main()
