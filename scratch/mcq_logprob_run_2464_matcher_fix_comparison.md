# MCQ Logprob matcher-fix comparison - request_group_id=2464

## Headline

- additional_non_null_predictions_across_all_models: `3808`
- compared_models: `14`
- artifact_suffix: `_v2_matcher_fix`

## Per-model side-by-side summary

| model | run_id | strategy | parquet_row_count (orig / repro) | non_null_rate (orig / repro) | manual_accuracy (orig / repro) | top_predicted_letter (orig / repro) | top_predicted_letter_pct (orig / repro) | delta_non_null |
|---|---:|---|---:|---:|---:|---|---:|---:|
| `alpindale/goliath-120b` | 1053 | `latin_squares_25` | 1125 / 1125 | 100.00% / 100.00% | 36.62% / 36.62% | `C` / `C` | 53.96% / 53.96% | +0 |
| `anthracite-org/magnum-v4-72b` | 1054 | `latin_squares_25` | 1125 / 1125 | 100.00% / 100.00% | 61.87% / 61.87% | `A` / `A` | 39.64% / 39.64% | +0 |
| `gryphe/mythomax-l2-13b` | 1055 | `full_120` | 5400 / 5400 | 90.15% / 93.96% | 20.40% / 21.99% | `A` / `A` | 99.94% / 96.75% | +206 |
| `mancer/weaver` | 1056 | `latin_squares_25` | 1125 / 1125 | 0.00% / 100.00% | n/a / 16.53% | `n/a` / `A` | n/a / 80.36% | +1125 |
| `mistralai/ministral-3b-2512` | 1057 | `full_120` | 5400 / 5400 | 1.11% / 1.11% | 21.67% / 21.67% | `A` / `A` | 100.00% / 100.00% | +0 |
| `openai/gpt-3.5-turbo` | 1058 | `latin_squares_25` | 1125 / 1125 | 100.00% / 100.00% | 30.93% / 30.93% | `A` / `A` | 29.60% / 29.60% | +0 |
| `openai/gpt-4-turbo` | 1059 | `latin_squares_25` | 1125 / 1125 | 21.24% / 21.24% | 94.98% / 94.98% | `E` / `E` | 23.85% / 23.85% | +0 |
| `openai/gpt-4o` | 1060 | `latin_squares_25` | 1125 / 1125 | 46.93% / 46.93% | 76.14% / 76.14% | `A` / `A` | 44.51% / 44.51% | +0 |
| `openai/gpt-4o-mini` | 1061 | `full_120` | 5400 / 5400 | 0.37% / 0.37% | 100.00% / 100.00% | `A` / `A` | 95.00% / 95.00% | +0 |
| `qwen/qwen3.6-max-preview` | 1062 | `latin_squares_25` | 1125 / 1125 | 0.00% / 0.00% | n/a / n/a | `n/a` / `n/a` | n/a / n/a | +0 |
| `sao10k/l3.3-euryale-70b` | 1063 | `latin_squares_25` | 1125 / 1125 | 100.00% / 100.00% | 52.18% / 52.18% | `A` / `A` | 61.42% / 61.42% | +0 |
| `thedrummer/rocinante-12b` | 1064 | `full_120` | 5400 / 5400 | 99.87% / 99.87% | 25.81% / 25.81% | `C` / `C` | 37.38% / 37.38% | +0 |
| `thedrummer/unslopnemo-12b` | 1065 | `full_120` | 5400 / 5400 | 99.26% / 99.26% | 26.36% / 26.36% | `D` / `D` | 39.57% / 39.57% | +0 |
| `undi95/remm-slerp-l2-13b` | 1066 | `full_120` | 5400 / 5400 | 54.13% / 100.00% | 19.47% / 29.28% | `A` / `A` | 100.00% / 68.22% | +2477 |

## Models with biggest shifts

- largest_non_null_recovery:
  - `undi95/remm-slerp-l2-13b`: `+2477`
  - `mancer/weaver`: `+1125`
  - `gryphe/mythomax-l2-13b`: `+206`
- largest_manual_accuracy_shift (absolute):
  - `undi95/remm-slerp-l2-13b`: `+9.81 pp` (abs `9.81 pp`)
  - `gryphe/mythomax-l2-13b`: `+1.60 pp` (abs `1.60 pp`)
  - `thedrummer/unslopnemo-12b`: `+0.00 pp` (abs `0.00 pp`)
- largest_top_letter_bias_shift (absolute):
  - `undi95/remm-slerp-l2-13b`: `-31.78 pp` (abs `31.78 pp`, `A` -> `A`)
  - `gryphe/mythomax-l2-13b`: `-3.19 pp` (abs `3.19 pp`, `A` -> `A`)
  - `thedrummer/unslopnemo-12b`: `+0.00 pp` (abs `0.00 pp`, `D` -> `D`)

Report generated at `2026-05-10T07:37:01.036335Z` with reprocessing wall-clock `8.111s` (write run).
