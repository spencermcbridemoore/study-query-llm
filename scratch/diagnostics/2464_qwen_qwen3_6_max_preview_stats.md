# Aggregate stats — qwen/qwen3.6-max-preview

## Header

- model: `qwen/qwen3.6-max-preview`
- run_id: `1062`
- strategy: `latin_squares_25`
- parquet_row_count: `1125`
- error_rate: `100.00%`

## Distribution of correct_letter

| value | count | percent |
| --- | --- | --- |
| A | 225 | 20.00% |
| B | 225 | 20.00% |
| C | 225 | 20.00% |
| D | 225 | 20.00% |
| E | 225 | 20.00% |

## Distribution of predicted_letter

| value | count | percent |
| --- | --- | --- |
| A | 0 | 0.00% |
| B | 0 | 0.00% |
| C | 0 | 0.00% |
| D | 0 | 0.00% |
| E | 0 | 0.00% |
| null | 1125 | 100.00% |

## Cross-tab predicted_letter × correct_letter

| predicted_letter | A | B | C | D | E | null | All |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| null | 225 | 225 | 225 | 225 | 225 | 0 | 1125 |
| All | 225 | 225 | 225 | 225 | 225 | 0 | 1125 |

## Manual accuracy

- numerator `(predicted_letter == correct_letter).sum()`: `0`
- denominator `predicted_letter.notna().sum()`: `0`
- manual_accuracy: `n/a`

## Verifier-reported accuracy comparison

| manual_accuracy | verifier_reported_accuracy |
| --- | --- |
| n/a | n/a |

## Distribution of error values

| error | count | percent |
| --- | --- | --- |
| http_400 | 1123 | 99.82% |
| typeerror | 2 | 0.18% |

## Per-permutation correct_letter check (question_idx=0)

| permutation_idx | permutation_sigma | correct_letter |
| --- | --- | --- |
| 0 | [1, 2, 3, 4, 5] | A |
| 1 | [1, 2, 3, 5, 4] | A |
| 10 | [1, 3, 5, 2, 4] | A |
| 13 | [1, 4, 2, 5, 3] | A |
| 23 | [1, 5, 4, 3, 2] | A |
| 29 | [2, 1, 5, 4, 3] | B |
| 33 | [2, 3, 4, 5, 1] | E |
| 35 | [2, 3, 5, 4, 1] | E |
| 36 | [2, 4, 1, 3, 5] | C |
| 44 | [2, 5, 3, 1, 4] | D |
| 50 | [3, 1, 4, 2, 5] | B |
| 55 | [3, 2, 1, 5, 4] | C |
| 64 | [3, 4, 5, 1, 2] | D |
| 69 | [3, 5, 2, 4, 1] | E |
| 70 | [3, 5, 4, 1, 2] | D |
| 72 | [4, 1, 2, 3, 5] | B |
| 75 | [4, 1, 3, 5, 2] | B |
| 83 | [4, 2, 5, 3, 1] | E |
| 86 | [4, 3, 2, 1, 5] | D |
| 90 | [4, 5, 1, 2, 3] | C |
| 96 | [5, 1, 2, 3, 4] | B |
| 106 | [5, 2, 4, 1, 3] | D |
| 109 | [5, 3, 1, 4, 2] | C |
| 114 | [5, 4, 1, 2, 3] | C |
| 119 | [5, 4, 3, 2, 1] | E |
