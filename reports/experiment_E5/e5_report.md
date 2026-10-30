# Experiment E5: Abstention Strategy

## Objective
Evaluate whether explicit abstention instructions and a retrieval-score threshold help the RAG system say `NOT_COVERED` when the legal corpus does not answer a question, while avoiding unnecessary refusals on answerable questions.

## Experimental Conditions
The experiment compares four settings:

1. `no_instruction_no_threshold`
2. `instruction_only`
3. `threshold_only`
4. `instruction_and_threshold`

## Results

| Setting | Abstention Accuracy | False Refusal Rate | Citations Valid (first attempt) |
|---|---:|---:|---:|
| no_instruction_no_threshold | 0.000 | 0.000 | 0.933 |
| instruction_only | 0.600 | 0.000 | 0.963 |
| threshold_only | 0.600 | 0.000 | 0.926 |
| instruction_and_threshold | **0.800** | **0.000** | **0.962** |

## Interpretation
The strongest configuration was the combination of an explicit abstention instruction and a retrieval-score threshold. It improved abstention accuracy from 0.0 with no controls to 0.8, while maintaining a false-refusal rate of 0.0 on answerable questions.

The instruction-only and threshold-only settings each reached 0.6 abstention accuracy, showing that both components help independently. However, combining them produced the best overall behavior.

Citation validity remained high across all settings, with the combined setting achieving 0.962 on first-attempt citation validity.

## Selected Final Configuration
- Prompt style: `system_v2`
- Minimum rerank score: `0.007795105688273907`
- Final strategy: explicit abstention instruction + retrieval-score threshold

## Conclusion
E5 supports using both an explicit abstention instruction and a tuned retrieval threshold in the final PDPL Compliance Copilot. This configuration improved detection of unsupported questions without increasing false refusals.
