"""Document title-level review and conservatively exclude questionable or related-event records."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
AMBIGUOUS = {
    # Education titles where subject is research, policy outside education, museum or ambiguous.
    'cn10352214','cn10360530','cn10373451','cn10374034','cn10375846','cn10383287','cn10389363',
    'cn10394442','cn10400914','cn10402187','cn10413455','cn10416910','cn10428061','cn10428262',
    'cn10431856','cn10442667','cn10450326','cn10451651','cn10458266','cn10462837','cn10470727',
    'cn10473512','cn10474318','cn10479891','cn10485460','cn10486873','cn10487545','cn10489101',
    'cn10490452','cn10491271','cn10500009','cn10504840','cn10511883','cn10513523','cn10523497',
    'cn10528426','cn10528690','cn10533156','cn10374773',
    # Other classes use the same conservative title-level standard.
    'cn10362972','cn10369563','cn10383807','cn10410220','cn10504290','cn10516751','cn10517279','cn10517314',
    'cn10377684','cn10414536','cn10451424','cn10514266','cn10518043','cn10519894',
    'cn10439143','cn10530861',
    'cn10377282','cn10378104','cn10387067','cn10388433','cn10390646','cn10391417','cn10391688',
    'cn10395342','cn10395391','cn10396106','cn10496047','cn10505428','cn10541560',
    'cn10364446','cn10366361','cn10367107','cn10401475','cn10444902','cn10510404','cn10525670','cn10530695',
}
RELATED_EVENT = {
    # Explicitly related reports are excluded, even when trigram similarity is below 0.5.
    'cn10414410','cn10414527','cn10428339','cn10428440','cn10428637','cn10437669',
    'cn10411503','cn10411527','cn10367308','cn10367318','cn10458641','cn10458659',
    'cn10348620','cn10348755','cn10348909','cn10349237','cn10349285','cn10349286','cn10350241',
    'cn10350247','cn10350778','cn10351057','cn10351237','cn10351339','cn10351787','cn10352780','cn10354303',
    'cn10365539','cn10365676','cn10391958','cn10420809','cn10421390','cn10421764','cn10423813',
    'cn10427019','cn10427087','cn10429643','cn10431083','cn10432120','cn10454589','cn10454793',
    'cn10455528','cn10455701','cn10457094','cn10461968','cn10464531','cn10508300','cn10508366',
    'cn10522062','cn10522532','cn10522678','cn10523633',
}


def main():
    inp = HERE / 'f3_title_review_with_event_flags.csv'
    dest = HERE / 'f3_title_review_decisions.csv'
    if dest.exists():
        raise FileExistsError(dest)
    df = pd.read_csv(inp, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    def decide(r):
        why = []
        if r.possible_event_conflict == 'True':
            why.append('文本近似筛查达到0.5阈值，保守排除')
        if r.id in AMBIGUOUS:
            why.append('标题主题与栏目交界或证据不明，待人工核对')
        if r.id in RELATED_EVENT:
            why.append('可能为同一事件的多篇报道，保守排除')
        return '；'.join(why)
    df['review_note'] = df.apply(decide, axis=1)
    df['semantic_review'] = df.review_note.map(lambda x: 'exclude_pending' if x else 'title_review_pass')
    df.to_csv(dest, index=False, encoding='utf-8-sig')
    print(df.groupby(['category','semantic_review']).size().to_string())


if __name__ == '__main__':
    main()
