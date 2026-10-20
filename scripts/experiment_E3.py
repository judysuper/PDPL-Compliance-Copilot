"""PDPL E3: Arabic/English cross-lingual retrieval vs an English-focused baseline.
Run from repo root: python scripts/experiment_E3.py
Requires: pandas numpy sentence-transformers
No LLM API key used.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / 'data/evaluation/gold_qa_set.csv'
META = ROOT / 'data/indexes/pdpl_metadata.csv'
OUT = ROOT / 'reports/experiment_E3'
K = 5
MODELS = {
    'Multilingual E5': 'intfloat/multilingual-e5-base',
    'English MiniLM baseline': 'sentence-transformers/all-MiniLM-L6-v2',
}

def key(doc, number):
    doc = str(doc).strip().lower()
    for suffix in ('_ar', '_en'):
        if doc.endswith(suffix):
            doc = doc[:-len(suffix)]
    try:
        n = str(int(float(number)))
    except (TypeError, ValueError):
        n = str(number).strip()
    return f'{doc}:{n}'

def score(gold, hits):
    first = next((i for i, x in enumerate(hits[:K], 1) if x == gold), None)
    return int(first is not None), 1.0 / first if first else 0.0, first

def main():
    questions = pd.read_csv(GOLD)
    corpus = pd.read_csv(META)
    required_q = {'question_id', 'question', 'language', 'answerable_from_sources', 'source_document', 'article_number'}
    required_c = {'document_id', 'language', 'article_number', 'article_text'}
    if required_q - set(questions.columns) or required_c - set(corpus.columns):
        raise ValueError('Missing required columns in Gold QA or metadata')
    questions = questions[questions.answerable_from_sources.astype(str).str.lower().eq('true')].copy()
    questions['language'] = questions.language.astype(str).str.lower().str.strip()
    corpus['language'] = corpus.language.astype(str).str.lower().str.strip()
    questions = questions[questions.language.isin(['ar', 'en'])]
    corpus = corpus[corpus.language.isin(['ar', 'en'])].copy().reset_index(drop=True)
    questions['gold_key'] = [key(a, b) for a, b in zip(questions.source_document, questions.article_number)]
    corpus['article_key'] = [key(a, b) for a, b in zip(corpus.document_id, corpus.article_number)]
    if questions.empty or corpus.empty:
        raise ValueError('No valid questions or corpus rows')
    print('Answerable questions by language:', questions.language.value_counts().to_dict())
    print('Corpus articles by language:', corpus.language.value_counts().to_dict())

    rows = []
    qtext = questions.question.astype(str).tolist()
    for label, model_name in MODELS.items():
        print(f'\nLoading {label}: {model_name}')
        model = SentenceTransformer(model_name)
        is_e5 = label == 'Multilingual E5'
        qprefix = 'query: ' if is_e5 else ''
        pprefix = 'passage: ' if is_e5 else ''
        qvectors = model.encode([qprefix + x for x in qtext], normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=16, show_progress_bar=True)
        pvectors = model.encode([pprefix + x for x in corpus.article_text.fillna('').astype(str)],
                                normalize_embeddings=True, convert_to_numpy=True,
                                batch_size=16, show_progress_bar=True)
        # Equal-language and cross-language tests explicitly restrict target corpus language.
        for target in ('ar', 'en'):
            ids = np.flatnonzero(corpus.language.to_numpy() == target)
            keys = corpus.iloc[ids].article_key.tolist()
            if len(ids) == 0:
                continue
            similarities = qvectors @ pvectors[ids].T
            for qi, (_, q) in enumerate(questions.iterrows()):
                positions = np.argsort(-similarities[qi])[:K]
                found = [keys[j] for j in positions]
                recall, mrr, rank = score(q.gold_key, found)
                rows.append({
                    'question_id': q.question_id,
                    'question_language': q.language,
                    'corpus_language': target,
                    'mode': 'cross-language' if q.language != target else 'same-language',
                    'model': label,
                    'gold_article': q.gold_key,
                    'retrieved_top5': ' | '.join(found),
                    'recall_at_5': recall,
                    'mrr_at_5': mrr,
                    'first_correct_rank': rank,
                })
            print(f'Finished {label} -> {target} corpus')
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(['model', 'question_language', 'corpus_language', 'mode'], as_index=False).agg(
        questions=('question_id', 'count'), recall_at_5=('recall_at_5', 'mean'),
        mrr_at_5=('mrr_at_5', 'mean'))
    errors = detail[detail.first_correct_rank.isna()].copy()
    detail.to_csv(OUT / 'e3_per_question.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'e3_summary.csv', index=False, encoding='utf-8-sig')
    errors.to_csv(OUT / 'e3_errors.csv', index=False, encoding='utf-8-sig')
    print('\nE3 summary:')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nReports:', OUT)
    print('Caution: English MiniLM is an English-focused baseline, not a controlled monolingual-vs-multilingual architecture comparison.')
    print('Each gold question currently has one marked article; additional valid citations require manual review.')

if __name__ == '__main__':
    main()
