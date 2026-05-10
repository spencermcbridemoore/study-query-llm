# Follow-up brief: OpenRouter TypeError in MCQ logprob run

## Scope
- Capability classification: `lower-capability-feasible`.
- Target issue from Phase 2 run `request_group_id=2464`.
- Goal: remove provider-side `TypeError: 'NoneType' object is not subscriptable` while preserving existing runner-level skip-and-record behavior.

## Symptom observed
- During Phase 2 execution, some calls logged:
  - `TypeError: 'NoneType' object is not subscriptable`
  - stack points to `src/study_query_llm/providers/openai_compatible_chat_provider.py` at `choice = response.choices[0]`.
- Error is caught upstream by `InferenceService` as non-retryable and recorded into MCQ parquet rows (error class `typeerror`), so run completes but quality degrades.

## Likely root cause
- Provider parsing path assumes a fully-populated OpenAI-style response object:
  - `response` exists
  - `response.choices` is list-like and non-empty
  - `choice.message.content` exists
- Some OpenRouter responses for specific model/call combinations appear to return null/empty payload shape (or `choices=None`), violating those assumptions.
- Current provider implementation subscripts directly without defensive guards, causing `TypeError` before a normalized `ProviderResponse` can be returned.

## Proposed fix (provider-only)
1. In `OpenAICompatibleChatProvider.complete(...)`, replace direct subscript chain with defensive extraction:
   - check `response is not None`
   - check `choices` is a non-empty list/sequence
   - handle missing `message`, missing `content`, missing `finish_reason`, missing `usage`.
2. If response envelope is syntactically valid but has no usable choices/logprobs content:
   - return a normalized `ProviderResponse` (no raise),
   - set `text=""`,
   - preserve `raw_response`,
   - include metadata flag(s), e.g. `logprobs_returned_count=0` and `empty_choices=True`.
3. Only raise for truly invalid transport/protocol failures where no response object can be interpreted at all.

## Test plan
- Add provider unit test in `tests/test_providers/test_openai_compatible_chat.py`:
  - mock OpenAI client response with `choices=None` (or empty list),
  - call `OpenAICompatibleChatProvider.complete(...)`,
  - assert no exception,
  - assert returned `ProviderResponse` is well-formed and includes normalization metadata.
- Add a second variant where `choices=[...]` but `message.content=None` to ensure empty-string normalization is stable.

## Out of scope
- Any changes to `src/study_query_llm/services/method_runners/mcq_logprob_basic.py`.
- Any changes to `src/study_query_llm/services/inference_service.py` retry/exception policy.
- Any change to Phase 2 orchestration guardrails/reporting semantics.

## Acceptance criteria
- No `TypeError` emitted from `openai_compatible_chat_provider.py` for null/empty `choices` payloads.
- Provider tests cover null/empty-choice normalization path.
- Existing MCQ run path continues to classify true upstream HTTP/protocol failures as errors without behavior drift.
