# Investigation: `embedding_cache_leases` `UniqueViolation` in per-pair phase0 (Cohere v4)

Date: 2026-05-17  
Scope: Step 1 only (investigation + evidence), per operator request.

## 1) Matrix-level lease call path in `pipeline/embed.py`

I inspected `src/study_query_llm/pipeline/embed.py` (current `HEAD`).

The matrix-level path **does call** `RawCallRepository.try_acquire_embedding_cache_lease(...)`; it is not doing a raw SQL insert in `embed.py`.

Relevant excerpt:

```python
lease_logical_key, lease_storage_key = _matrix_lease_key(...)
owner = f"embed-matrix:{os.getpid()}:{id(dataset_key)}:{time.time_ns()}"
with db_conn.session_scope() as session:
    repo = RawCallRepository(session)
    lease_holder = repo.try_acquire_embedding_cache_lease(
        cache_key=lease_storage_key,
        owner=owner,
        lease_seconds=max(1, int(singleflight_lease_seconds)),
    )
```

Also relevant: `_matrix_lease_key(...)` stores a **hashed/truncated** key, not the full logical key:

```python
logical_key = (
    f"embed_matrix:{dataset_key}:{embedding_engine}:{provider}:"
    f"{int(entry_max)}:{key_version}"
)
digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()
storage_key = f"embed_matrix:{digest}"[:64]
```

## 2) Stale-lease takeover logic in `RawCallRepository.try_acquire_embedding_cache_lease`

I inspected `src/study_query_llm/db/raw_call_repository.py::try_acquire_embedding_cache_lease`.

Observed behavior:

1. `SELECT` lease row by `cache_key`.
2. If none found: create row + `flush()` + return `True`.
3. Else, if any of these are true:
   - `lease_expires_at` is null after UTC coercion, OR
   - `lease_owner == owner`, OR
   - `lease_expires_at <= now` (expired),
   then overwrite owner/expiry + `flush()` + return `True`.
4. Otherwise return `False`.

Relevant excerpt:

```python
lease = (
    self.session.query(EmbeddingCacheLease)
    .filter(EmbeddingCacheLease.cache_key == cache_key)
    .first()
)
if lease is None:
    lease = EmbeddingCacheLease(...)
    self.session.add(lease)
    self.session.flush()
    return True

lease_expiry_utc = self._coerce_utc_aware(lease.lease_expires_at)
if lease_expiry_utc is None or lease.lease_owner == owner or lease_expiry_utc <= now:
    lease.lease_owner = owner
    lease.lease_expires_at = expires
    lease.updated_at = now
    self.session.flush()
    return True
return False
```

## 3) Evidence from failing pair artifacts (`20260515T055216Z`)

`snap18__embed-v-4-0` failed with:

- `psycopg2.errors.UniqueViolation`
- constraint: `embedding_cache_leases_pkey`
- key: `embed_matrix:c8c1f9773f978ce3baa9e17a70ede5dc8691b5cb5b4cae5a870`
- failing SQL: `INSERT INTO embedding_cache_leases (...) VALUES (...)`
- stack frame: `raw_call_repository.py`, inside `try_acquire_embedding_cache_lease`, at `self.session.flush()`.

This indicates the exception originates from the repository method’s insert path, not from bypass in `embed.py`.

## 4) Cache key for the colliding pair lineage

Pair specs:

- `snap17__embed-v-4-0`: `source_dataframe_group_id=16`, `source_dataframe_row_count=13069`
- `snap18__embed-v-4-0`: same lineage key `[16, 13069]`

Matrix logical lease key resolves to:

`embed_matrix:dataframe:16:full:embed-v-4-0:azure:13069:raw_v1`

SHA-256 storage lease key (truncated to 64 chars):

`embed_matrix:c8c1f9773f978ce3baa9e17a70ede5dc8691b5cb5b4cae5a870`

This exactly matches the failing `UniqueViolation` key in `snap18` logs.

## 5) Required DB query output (verbatim query requested)

Executed:

```sql
SELECT cache_key, lease_owner, lease_expires_at
FROM embedding_cache_leases
WHERE cache_key LIKE 'embed_matrix:%embed-v-4-0%';
```

Result at investigation time:

- `row_count=0`

Note: because matrix lease storage keys are hashed (`embed_matrix:<sha256...>`), they do not include the literal substring `embed-v-4-0`, so this `LIKE` pattern does not match current hashed rows by construction.

## 6) Additional direct key probe

Checked exact collision key:

```sql
SELECT cache_key, lease_owner, lease_expires_at
FROM embedding_cache_leases
WHERE cache_key = 'embed_matrix:c8c1f9773f978ce3baa9e17a70ede5dc8691b5cb5b4cae5a870';
```

Result at investigation time:

- `row_count=0`

No currently present stale row for that key on the investigated DB target.

## 7) Root-cause conclusion

The observed blocker is a **repository-level race condition** in `try_acquire_embedding_cache_lease`:

- two contenders can both observe `lease is None`;
- both attempt insert;
- one wins, the other gets `UniqueViolation`;
- exception bubbles as a permanent phase0 failure.

This is independent of stale-lease takeover logic and independent of any raw insert bypass in `embed.py` (none found).

---

## STOP (per operator stop condition)

A code bug was discovered during step 1 (repository race), so implementation is paused pending operator review/approval of the fix design.
