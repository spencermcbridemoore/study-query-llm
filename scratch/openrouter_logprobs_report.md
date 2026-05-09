# OpenRouter Logprobs Validation Report

- Run started: `2026-05-09T06:04:51Z`
- Run finished: `2026-05-09T06:07:17Z`

## Section 1: PASS

| Model | Family | Context Length | Prompt Cost | Completion Cost | Observed top_logprobs cap |
|---|---|---:|---:|---:|---:|
| qwen/qwen3.6-max-preview | qwen | 262144 | 0.00000104 | 0.00000624 | 5 |
| mistralai/ministral-3b-2512 | mistralai | 131072 | 0.0000001 | 0.0000001 | 5 |
| sao10k/l3.3-euryale-70b | sao10k | 131072 | 0.00000065 | 0.00000075 | 5 |
| openai/gpt-4o-2024-11-20 | openai | 128000 | 0.0000025 | 0.00001 | 5 |
| thedrummer/unslopnemo-12b | thedrummer | 32768 | 0.0000004 | 0.0000004 | 5 |
| anthracite-org/magnum-v4-72b | anthracite-org | 16384 | 0.000003 | 0.000005 | 5 |
| thedrummer/rocinante-12b | thedrummer | 32768 | 0.00000017 | 0.00000043 | 5 |
| openai/gpt-4o-2024-08-06 | openai | 128000 | 0.0000025 | 0.00001 | 5 |
| openai/gpt-4o-mini-2024-07-18 | openai | 128000 | 0.00000015 | 0.0000006 | 5 |
| openai/gpt-4o-mini | openai | 128000 | 0.00000015 | 0.0000006 | 5 |
| openai/gpt-4o-2024-05-13 | openai | 128000 | 0.000005 | 0.000015 | 5 |
| openai/gpt-4o | openai | 128000 | 0.0000025 | 0.00001 | 5 |
| openai/gpt-4-turbo | openai | 128000 | 0.00001 | 0.00003 | 5 |
| openai/gpt-3.5-turbo-0613 | openai | 4095 | 0.000001 | 0.000002 | 5 |
| alpindale/goliath-120b | alpindale | 6144 | 0.00000375 | 0.0000075 | 5 |
| openai/gpt-3.5-turbo-16k | openai | 16385 | 0.000003 | 0.000004 | 5 |
| mancer/weaver | mancer | 8000 | 0.00000075 | 0.000001 | 5 |
| undi95/remm-slerp-l2-13b | undi95 | 6144 | 0.00000045 | 0.00000065 | 5 |
| gryphe/mythomax-l2-13b | gryphe | 4096 | 0.00000006 | 0.00000006 | 5 |
| openai/gpt-3.5-turbo | openai | 16385 | 0.0000005 | 0.0000015 | 5 |

## Section 2: EMPTY/REJECTED

| Model | Classification | Response Excerpt |
|---|---|---|
| openai/gpt-chat-latest | OTHER_ERROR | Provider returned error |
| x-ai/grok-4.3 | OTHER_ERROR | Provider returned error |
| ~moonshotai/kimi-latest | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.6-27b | EMPTY | Successful response without logprobs object. |
| deepseek/deepseek-v4-pro | EMPTY | Successful response without logprobs object. |
| deepseek/deepseek-v4-flash | EMPTY | Successful response without logprobs object. |
| openai/gpt-5.4-image-2 | OTHER_ERROR | Provider returned error |
| moonshotai/kimi-k2.6 | EMPTY | Successful response without logprobs object. |
| z-ai/glm-5.1 | EMPTY | Successful response without logprobs object. |
| google/gemma-4-26b-a4b-it | EMPTY | Successful response without logprobs object. |
| google/gemma-4-31b-it | EMPTY | Successful response without logprobs object. |
| x-ai/grok-4.20-multi-agent | OTHER_ERROR | Provider returned error |
| x-ai/grok-4.20 | OTHER_ERROR | Provider returned error |
| minimax/minimax-m2.7 | EMPTY | Successful response without logprobs object. |
| nvidia/nemotron-3-super-120b-a12b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.5-9b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.5-35b-a3b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.5-27b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.5-122b-a10b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3.5-397b-a17b | EMPTY | Successful response without logprobs object. |
| minimax/minimax-m2.5 | EMPTY | Successful response without logprobs object. |
| z-ai/glm-5 | EMPTY | Successful response without logprobs object. |
| moonshotai/kimi-k2.5 | EMPTY | Successful response without logprobs object. |
| openai/gpt-audio | OTHER_ERROR | Provider returned error |
| openai/gpt-audio-mini | OTHER_ERROR | Provider returned error |
| z-ai/glm-4.7 | EMPTY | Successful response without logprobs object. |
| mistralai/ministral-14b-2512 | EMPTY | Successful response without logprobs object. |
| mistralai/ministral-8b-2512 | EMPTY | Successful response without logprobs object. |
| x-ai/grok-4.1-fast | OTHER_ERROR | Provider returned error |
| openai/gpt-5-image-mini | OTHER_ERROR | Provider returned error |
| openai/gpt-5-image | OTHER_ERROR | Provider returned error |
| openai/o3-deep-research | OTHER_ERROR | Provider returned error |
| openai/o4-mini-deep-research | OTHER_ERROR | Provider returned error |
| x-ai/grok-4-fast | OTHER_ERROR | Provider returned error |
| x-ai/grok-code-fast-1 | OTHER_ERROR | Provider returned error |
| deepseek/deepseek-chat-v3.1 | EMPTY | Successful response without logprobs object. |
| openai/gpt-4o-audio-preview | OTHER_ERROR | Provider returned error |
| openai/gpt-oss-120b | EMPTY | Successful response without logprobs object. |
| openai/gpt-oss-20b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3-235b-a22b-2507 | EMPTY | Successful response without logprobs object. |
| x-ai/grok-4 | OTHER_ERROR | Provider returned error |
| x-ai/grok-3-mini | OTHER_ERROR | Provider returned error |
| x-ai/grok-3 | OTHER_ERROR | Provider returned error |
| qwen/qwen3-30b-a3b | EMPTY | Successful response without logprobs object. |
| qwen/qwen3-14b | EMPTY | Successful response without logprobs object. |
| x-ai/grok-3-mini-beta | OTHER_ERROR | Provider returned error |
| x-ai/grok-3-beta | OTHER_ERROR | Provider returned error |
| deepseek/deepseek-r1-distill-qwen-32b | EMPTY | Successful response without logprobs object. |
| microsoft/phi-4 | EMPTY | Successful response without logprobs object. |
| meta-llama/llama-3.1-8b-instruct | EMPTY | Successful response without logprobs object. |
| mistralai/mistral-nemo | EMPTY | Successful response without logprobs object. |
| openai/gpt-4-turbo-preview | OTHER_ERROR | Provider returned error |
| openrouter/auto | EMPTY | Successful response without logprobs object. |
| openai/gpt-4-1106-preview | OTHER_ERROR | Provider returned error |
| openai/gpt-3.5-turbo-instruct | OTHER_ERROR | Provider returned error |

## Section 3: SKIPPED-DUE-TO-COST

| Model | Family | Prompt Cost | Completion Cost | Reason |
|---|---|---:|---:|---|
| openai/gpt-4-0314 | openai | 0.00003 | 0.00006 | prompt_price_above_threshold |
| openai/gpt-4 | openai | 0.00003 | 0.00006 | prompt_price_above_threshold |

## Section 4: Catalog Summary Stats

| Family | PASS | EMPTY | REJECTED | OTHER_ERROR | SKIPPED-DUE-TO-COST |
|---|---:|---:|---:|---:|---:|
| alpindale | 1 | 0 | 0 | 0 | 0 |
| anthracite-org | 1 | 0 | 0 | 0 | 0 |
| deepseek | 0 | 4 | 0 | 0 | 0 |
| google | 0 | 2 | 0 | 0 | 0 |
| gryphe | 1 | 0 | 0 | 0 | 0 |
| mancer | 1 | 0 | 0 | 0 | 0 |
| meta-llama | 0 | 1 | 0 | 0 | 0 |
| microsoft | 0 | 1 | 0 | 0 | 0 |
| minimax | 0 | 2 | 0 | 0 | 0 |
| mistralai | 1 | 3 | 0 | 0 | 0 |
| moonshotai | 0 | 2 | 0 | 0 | 0 |
| nvidia | 0 | 1 | 0 | 0 | 0 |
| openai | 10 | 2 | 0 | 12 | 2 |
| openrouter | 0 | 1 | 0 | 0 | 0 |
| qwen | 1 | 9 | 0 | 0 | 0 |
| sao10k | 1 | 0 | 0 | 0 | 0 |
| thedrummer | 2 | 0 | 0 | 0 | 0 |
| undi95 | 1 | 0 | 0 | 0 | 0 |
| x-ai | 0 | 0 | 0 | 11 | 0 |
| z-ai | 0 | 3 | 0 | 0 | 0 |
| ~moonshotai | 0 | 1 | 0 | 0 | 0 |

Validation executed at `2026-05-09T06:07:17Z` UTC; OpenRouter catalog and behavior may change over time.
