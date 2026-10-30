## Retrieval (all answerable questions)

| method   | language   |   n |   recall@5 |   mrr@5 |   recall@10 |
|:---------|:-----------|----:|-----------:|--------:|------------:|
| bm25     | all        |  50 |      0.78  |   0.552 |       0.8   |
| bm25     | ar         |  34 |      0.765 |   0.506 |       0.794 |
| bm25     | en         |  16 |      0.812 |   0.651 |       0.812 |
| dense    | all        |  50 |      0.92  |   0.682 |       0.96  |
| dense    | ar         |  34 |      0.912 |   0.658 |       0.941 |
| dense    | en         |  16 |      0.938 |   0.734 |       1     |
| rrf      | all        |  50 |      0.84  |   0.717 |       0.92  |
| rrf      | ar         |  34 |      0.824 |   0.722 |       0.912 |
| rrf      | en         |  16 |      0.875 |   0.708 |       0.938 |
| reranked | all        |  50 |      0.98  |   0.84  |     nan     |
| reranked | ar         |  34 |      0.971 |   0.828 |     nan     |
| reranked | en         |  16 |      1     |   0.865 |     nan     |

## Experiment E1 (evaluation half)

| system    |   n_questions |   abstention_accuracy |   false_refusal_rate |   citations_valid_first_attempt |   cites_gold_article |   gold_in_sources |   n_errors |   manual_correctness_0_2 |   manual_support_yes |
|:----------|--------------:|----------------------:|---------------------:|--------------------------------:|---------------------:|------------------:|-----------:|-------------------------:|---------------------:|
| llm_only  |            30 |                   0.6 |                  0.2 |                         nan     |                 0.05 |            nan    |          0 |                    1.4   |              nan     |
| naive_rag |            30 |                   0.6 |                  0   |                           0.963 |                 0.96 |              0.96 |          0 |                    1.783 |                0.883 |
| full      |            30 |                   0.8 |                  0   |                           0.962 |                 1    |              1    |          0 |                    1.8   |                0.944 |
