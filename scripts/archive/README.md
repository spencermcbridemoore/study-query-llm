# Archive Directory

This directory contains scripts that were used for one-time tasks or are no longer maintained.

## Archived Scripts

### Migration Scripts (One-time, Completed)

- **`migrate_v1_to_v2.py`** - V1 SQLite to V2 Postgres migration
  - Purpose: One-time migration from legacy v1 database to v2 database
  - Status: Completed - migration was successful
  - Kept for: Historical reference and documentation

- **`verify_migration.py`** - Migration verification
  - Purpose: Verified data integrity after v1 → v2 migration
  - Status: Completed - verification passed
  - Kept for: Historical reference

### Legacy Testing Scripts

- **`test_deployment_completion.py`** - Legacy deployment testing
  - Purpose: Tested text completion with specific Azure OpenAI deployments
  - Status: May be obsolete (superseded by other testing methods)
  - Kept for: Reference if needed for debugging

## Notes

- These scripts are kept for historical reference and documentation
- They may not work with the current codebase
- Do not use these scripts without reviewing and updating them first
- See `../README.md` for active scripts
