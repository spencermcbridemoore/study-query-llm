#!/usr/bin/env python3
"""Quick check: Is the database empty?"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

db = DatabaseConnectionV2(db_url, enable_pgvector=False)
db.init_db()

with db.session_scope() as session:
    repo = RawCallRepository(session)
    
    # Count total calls
    total = repo.session.query(repo.model).count()
    
    # Count by modality
    embeddings = repo.session.query(repo.model).filter_by(modality="embedding").count()
    text = repo.session.query(repo.model).filter_by(modality="text").count()
    
    # Count groups
    from study_query_llm.db.models_v2 import Group
    groups = repo.session.query(Group).count()

print(f"Total RawCalls: {total}")
print(f"  - Embeddings: {embeddings}")
print(f"  - Text: {text}")
print(f"Total Groups: {groups}")

if total == 0:
    print("\n[EMPTY] Database is empty")
    sys.exit(1)
else:
    print(f"\n[OK] Database has {total} records")
    sys.exit(0)
