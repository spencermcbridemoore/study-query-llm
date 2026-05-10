# Top-token distribution — 14 models

## alpindale/goliath-120b

- model: `alpindale/goliath-120b`, run_id: `1053`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `The` | 610 | 54.22% |
| `To` | 262 | 23.29% |
| `C` | 193 | 17.16% |
| `B` | 46 | 4.09% |
| `D` | 14 | 1.24% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 1125 | 100.00% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## anthracite-org/magnum-v4-72b

- model: `anthracite-org/magnum-v4-72b`, run_id: `1054`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `To` | 1111 | 98.76% |
| `The` | 14 | 1.24% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 1125 | 100.00% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 3 | 0.27% |

## gryphe/mythomax-l2-13b

- model: `gryphe/mythomax-l2-13b`, run_id: `1055`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=5078, empty_rows=322

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| ` Situ` | 2527 | 49.76% |
| ` The` | 2274 | 44.78% |
| `▁The` | 158 | 3.11% |
| `
` | 45 | 0.89% |
| `▁To` | 43 | 0.85% |
| ` A` | 12 | 0.24% |
| `▁Let` | 5 | 0.10% |
| `para` | 3 | 0.06% |
| `w` | 3 | 0.06% |
| ` To` | 2 | 0.04% |
| ` now` | 1 | 0.02% |
| `2` | 1 | 0.02% |
| `f` | 1 | 0.02% |
| `h` | 1 | 0.02% |
| `un` | 1 | 0.02% |
| `y` | 1 | 0.02% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 4868 | 90.15% |
| sentencepiece_prefix (`▁A`) | 206 | 3.81% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## mancer/weaver

- model: `mancer/weaver`, run_id: `1056`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `▁The` | 812 | 72.18% |
| `▁To` | 313 | 27.82% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 0 | 0.00% |
| sentencepiece_prefix (`▁A`) | 1125 | 100.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## mistralai/ministral-3b-2512

- model: `mistralai/ministral-3b-2512`, run_id: `1057`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=1631, empty_rows=3769
- most rows have empty payloads; skipping rank-1 token frequency and letter-token presence stats.

## openai/gpt-3.5-turbo

- model: `openai/gpt-3.5-turbo`, run_id: `1058`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `A` | 333 | 29.60% |
| `C` | 314 | 27.91% |
| `B` | 257 | 22.84% |
| `D` | 168 | 14.93% |
| `E` | 53 | 4.71% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 1125 | 100.00% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 255 | 22.67% |

## openai/gpt-4-turbo

- model: `openai/gpt-4-turbo`, run_id: `1059`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `To` | 1125 | 100.00% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 239 | 21.24% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## openai/gpt-4o

- model: `openai/gpt-4o`, run_id: `1060`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `To` | 1125 | 100.00% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 528 | 46.93% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## openai/gpt-4o-mini

- model: `openai/gpt-4o-mini`, run_id: `1061`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=5400, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `To` | 5400 | 100.00% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 20 | 0.37% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## qwen/qwen3.6-max-preview

- model: `qwen/qwen3.6-max-preview`, run_id: `1062`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=0, empty_rows=1125
- most rows have empty payloads; skipping rank-1 token frequency and letter-token presence stats.

## sao10k/l3.3-euryale-70b

- model: `sao10k/l3.3-euryale-70b`, run_id: `1063`, parquet_row_count: `1125`
- top_logprobs payload capture: captured_rows=1125, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `To` | 914 | 81.24% |
| `The` | 151 | 13.42% |
| `D` | 27 | 2.40% |
| `C` | 24 | 2.13% |
| `A` | 5 | 0.44% |
| `B` | 4 | 0.36% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 1125 | 100.00% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 625 | 55.56% |

## thedrummer/rocinante-12b

- model: `thedrummer/rocinante-12b`, run_id: `1064`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=5393, empty_rows=7

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `The` | 2940 | 54.52% |
| `To` | 2453 | 45.48% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 5393 | 99.87% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## thedrummer/unslopnemo-12b

- model: `thedrummer/unslopnemo-12b`, run_id: `1065`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=5360, empty_rows=40

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `The` | 2786 | 51.98% |
| `To` | 2574 | 48.02% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 5360 | 99.26% |
| sentencepiece_prefix (`▁A`) | 0 | 0.00% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |

## undi95/remm-slerp-l2-13b

- model: `undi95/remm-slerp-l2-13b`, run_id: `1066`, parquet_row_count: `5400`
- top_logprobs payload capture: captured_rows=5400, empty_rows=0

### Most common rank-1 token (top 20)

| token (verbatim) | count | percent_of_captured_rows |
|---|---:|---:|
| `Situ` | 1625 | 30.09% |
| `▁The` | 1403 | 25.98% |
| `The` | 1215 | 22.50% |
| `▁To` | 1009 | 18.69% |
| `A` | 83 | 1.54% |
| `▁Let` | 64 | 1.19% |
| `▁E` | 1 | 0.02% |

### Letter-token presence stats (top-20 per row)

| category | row_count | percent_of_all_rows |
|---|---:|---:|
| no_prefix (`A`, ` A`, etc.) | 2923 | 54.13% |
| sentencepiece_prefix (`▁A`) | 2477 | 45.87% |
| bpe_prefix (`ĠA`) | 0 | 0.00% |
| other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`) | 0 | 0.00% |
