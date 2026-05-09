# Midterm 2 — clean single-answer MCQ subset

Self-contained summary of the filtered subset of `midterm2_questions@v1` for use as input to MCQ inference experiments. Share this with a fresh agent that has no prior context.

## Dataset reference

- Dataset name: `midterm2_questions@v1`
- `imported_run_id = 1049` (source: `1048`)
- Stored as parquet artifact in the `provenanced_runs` execution row; schema in `metadata_json.pipeline_stage_context`.
- Original CSV: `midterm2_hashed_question_info_enhanced.csv` — 204 rows × 21 columns total before filtering.

## Filter to apply at runner time

A runner consuming this subset must apply these filters inside the runner so the subset definition is captured in the runner's parameter schema (boundary validation; not implicit in the caller):

1. `ItemType == "choice"` (single-answer MCQ; excludes `multi-answer`, `numeric`, `categorization`, `formula`).
2. No picture references in `ProblemBody`, `ProblemBodyTemplate`, or `OptionA`-`OptionE`. Specifically, exclude any row where the regex `\b(figure|diagram|graph|plot|sketch|illustrat|depicted)\b|includegraphics` matches case-insensitively in any of those text columns.
3. No HTML `<img>` tags, markdown image syntax, or external image URLs (`https?://\S+\.(png|jpg|jpeg|gif|svg)`) in any text column.

The filter is reproducible against the parquet artifact at `imported_run_id=1049` with no external lookup.

## Resulting subset

- **45 rows** survive the filter.
- **5-option MCQ format**, single-answer. All five `OptionA`-`OptionE` columns are populated for every row.
- Correctness is stored **only** in the `CorrectLetter` column. Critical caveat: `IsCorrectA`, `IsCorrectB`, `IsCorrectC`, `IsCorrectD`, `IsCorrectE` are all `FALSE` for every row in this dataset and must NOT be used as the correctness signal — that would silently report zero correct answers everywhere. Use `CorrectLetter` only.

## Template structure (significant for experimental design)

The 45 rows are NOT 45 distinct problems. They cluster into **2 parameterized templates** by `temp_bank_id`:

- `temp_bank_id = 6445` → **25 instances**
- `temp_bank_id = 6438` → **20 instances**

Each cluster is the same underlying physics problem with different surface labels (object names, materials) and different numeric values (masses, angles, applied forces). All 45 are variants of two underlying problem types in classical mechanics.

The `ProblemBodyTemplate`, `Placeholders`, and `Values` columns are empty in this dataset — only the realized `ProblemBody` text is available per variant. The template structure is recoverable only via `temp_bank_id` grouping.

## Subject area

Introductory physics — classical mechanics. Forces, kinetic friction, Newton's second law, objects pulled or pushed at angles on surfaces.

## Schema reference (post-filter)

Relevant columns for an experiment runner:

- `ItemID` (int) — unique row id within the source CSV
- `ItemType` (str) — always `"choice"` after filter
- `ProblemBody` (str) — the realized question text
- `temp_bank_id` (int) — template cluster id (one of 6445, 6438)
- `OptionA`-`OptionE` (str) — the five answer choices, all non-null
- `CorrectLetter` (str) — single character `A`-`E`, the correct answer

Other columns in the source schema (`ProblemBodyTemplate`, `Placeholders`, `Values`, `Formula`, `Answer`, `MatchScore`, `IsCorrectA`-`IsCorrectE`) are either empty, redundant, or unreliable for this subset and should not be relied on by the experiment runner.
