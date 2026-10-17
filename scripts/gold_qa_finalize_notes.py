"""Safe Gold QA targeted editorial corrections. Preview by default; --apply saves a backup and updates only confirmed rows."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / 'data' / 'evaluation' / 'gold_qa_set.csv'
SOURCES = ROOT / 'data' / 'processed'

REVISIONS = {
    'Q016': {
        'reference_answer': 'على جهة التحكم، بعد تصحيح البيانات الشخصية، إشعار الجهات التي أُفصح لها سابقاً عن تلك البيانات دون تأخير، وفق المادة السابعة، الفقرة الثالثة، من اللائحة التنفيذية لنظام حماية البيانات الشخصية.'
    },
    'Q034': {
        'question': 'هل يمكن تحديد قيمة غرامة فُرضت فعليًا على شركة غير محددة بسبب واقعة تسرب بيانات في الرياض بتاريخ 1 أكتوبر 2026 باستخدام النصوص القانونية المتاحة وحدها؟',
        'reference_answer': 'لا يمكن تحديد وجود غرامة فعلية أو قيمتها من النصوص القانونية المتاحة وحدها؛ يلزم تحديد الشركة والرجوع إلى قرار رسمي موثق بشأن الواقعة، إن وجد.'
    },
    'Q039': {
        'question': 'هل تتضمن نصوص نظام حماية البيانات الشخصية ولوائحه تاريخ آخر تحديث لسياسة الخصوصية الخاصة بتطبيق «واو» حتى 17 أكتوبر 2026؟',
        'reference_answer': 'لا تتضمن النصوص القانونية المتاحة سياسة الخصوصية الحية لهذا التطبيق أو سجل تحديثاتها؛ يلزم الرجوع إلى سياسة التطبيق المنشورة ومصدر موثوق لتاريخ تحديثها.'
    }
}

def read_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp)
        return reader.fieldnames, list(reader)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply 3 editorial changes after reviewing preview')
    args = parser.parse_args()
    if not QA.exists():
        sys.exit(f'Missing file: {QA}')
    fields, rows = read_rows(QA)
    ids = [r['question_id'] for r in rows]
    if len(rows) != 60 or len(set(ids)) != 60:
        sys.exit('Stopped: expected 60 questions with unique IDs.')
    by_id = {r['question_id']: r for r in rows}
    for key in REVISIONS:
        if key not in by_id:
            sys.exit(f'Stopped: {key} is missing.')
    article, article_rows = read_rows(SOURCES / 'implementing_ar_articles.csv')
    ref = next((r for r in article_rows if r['article_number'] == '7'), None)
    if not ref or by_id['Q016']['source_document'] != 'implementing_ar' or by_id['Q016']['article_number'] != '7' or by_id['Q016']['supporting_text'].strip() != ref['article_text'].strip():
        sys.exit('Stopped: Q016 source differs from expected Implementing Regulation article 7.')
    q42 = by_id['Q042']
    print('Q042 LOCAL:',q42['source_document'], 'article',q42['article_number'])
    if q42['source_document'] != 'pdpl_ar' or q42['article_number'] != '11':
        print('WARNING: Q042 is NOT the edited article-11 version. Recheck the local data before final signoff.')
    print('\nOTHER REVIEW NOTES (no automatic edit):')
    print('Q029: verify exact scope of article 25; current answer already mentions large scale/recurrent processing.')
    print('Q059: compare conditions under PDPL article 20 and applicable regulation; current answer is substantially supported.')
    print('\nSUGGESTED EDITS:')
    for qid, changes in REVISIONS.items():
        print('\n'+qid)
        for column, new in changes.items():
            print(f'  {column}\n  OLD: {by_id[qid][column]}\n  NEW: {new}')
    if not args.apply:
        print('\nPREVIEW ONLY. No file was changed. Re-run with --apply only after reviewing edits.')
        return
    if q42['source_document'] != 'pdpl_ar' or q42['article_number'] != '11':
        sys.exit('Stopped: update Q042 to your approved version before applying.')
    for qid, changes in REVISIONS.items():
        by_id[qid].update(changes)
    backup = QA.with_name(f'gold_qa_pre_edits_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    shutil.copy2(QA, backup)
    tmp = QA.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(QA)
    print('\nUpdated Q016, Q034, Q039 only. Saved backup:',backup)
    print('Next: run python scripts/gold_qa_pipeline.py; verify output and conduct human review.')

if __name__ == '__main__':
    main()
