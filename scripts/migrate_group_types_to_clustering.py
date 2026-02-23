"""
One-time migration: rename generic group types to clustering-specific names.

Before:
    groups.group_type  = 'run'   → 'clustering_run'
    groups.group_type  = 'step'  → 'clustering_step'
    group_links.link_type = 'step' → 'clustering_step'

'run' groups created by inference.py (panel UI) are identifiable by
description = 'Run group created from Panel UI' and are renamed to
'inference_run' so the generic 'run' name stays free for the future.

The script is idempotent — re-running is safe.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text as sa_text
from study_query_llm.db.connection_v2 import DatabaseConnectionV2


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)

    with db.engine.connect() as conn:
        # ------------------------------------------------------------------ #
        # Count before                                                        #
        # ------------------------------------------------------------------ #
        before = {
            "run":   conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'run'")).scalar(),
            "step":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'step'")).scalar(),
            "link_step": conn.execute(sa_text("SELECT COUNT(*) FROM group_links WHERE link_type = 'step'")).scalar(),
            "c_run":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'clustering_run'")).scalar(),
            "c_step": conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'clustering_step'")).scalar(),
            "i_run":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'inference_run'")).scalar(),
        }

        print("=== Before migration ===")
        print(f"  groups  group_type='run'            : {before['run']}")
        print(f"  groups  group_type='step'           : {before['step']}")
        print(f"  group_links link_type='step'        : {before['link_step']}")
        print(f"  groups  group_type='clustering_run' : {before['c_run']}  (already migrated)")
        print(f"  groups  group_type='clustering_step': {before['c_step']}  (already migrated)")
        print(f"  groups  group_type='inference_run'  : {before['i_run']}  (already migrated)")

        if before["run"] == 0 and before["step"] == 0 and before["link_step"] == 0:
            print("\nNothing to migrate — all rows already use new type names.")
            return

        # ------------------------------------------------------------------ #
        # Migrate inference runs first (subset of 'run' rows)                #
        # ------------------------------------------------------------------ #
        r_inference = conn.execute(
            sa_text(
                "UPDATE groups SET group_type = 'inference_run' "
                "WHERE group_type = 'run' "
                "AND description = 'Run group created from Panel UI'"
            )
        )
        print(f"\n  Renamed {r_inference.rowcount} panel-UI run -> inference_run")

        # ------------------------------------------------------------------ #
        # Migrate remaining 'run' rows to clustering_run                     #
        # ------------------------------------------------------------------ #
        r_crun = conn.execute(
            sa_text(
                "UPDATE groups SET group_type = 'clustering_run' "
                "WHERE group_type = 'run'"
            )
        )
        print(f"  Renamed {r_crun.rowcount} run -> clustering_run")

        # ------------------------------------------------------------------ #
        # Migrate 'step' group_type                                          #
        # ------------------------------------------------------------------ #
        r_cstep = conn.execute(
            sa_text(
                "UPDATE groups SET group_type = 'clustering_step' "
                "WHERE group_type = 'step'"
            )
        )
        print(f"  Renamed {r_cstep.rowcount} step -> clustering_step (group_type)")

        # ------------------------------------------------------------------ #
        # Migrate 'step' link_type in group_links                            #
        # ------------------------------------------------------------------ #
        r_lstep = conn.execute(
            sa_text(
                "UPDATE group_links SET link_type = 'clustering_step' "
                "WHERE link_type = 'step'"
            )
        )
        print(f"  Renamed {r_lstep.rowcount} step -> clustering_step (link_type)")

        conn.commit()

        # ------------------------------------------------------------------ #
        # Count after                                                         #
        # ------------------------------------------------------------------ #
        after = {
            "run":   conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'run'")).scalar(),
            "step":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'step'")).scalar(),
            "link_step": conn.execute(sa_text("SELECT COUNT(*) FROM group_links WHERE link_type = 'step'")).scalar(),
            "c_run":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'clustering_run'")).scalar(),
            "c_step": conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'clustering_step'")).scalar(),
            "i_run":  conn.execute(sa_text("SELECT COUNT(*) FROM groups WHERE group_type = 'inference_run'")).scalar(),
        }

        print("\n=== After migration ===")
        print(f"  groups  group_type='run'            : {after['run']}  (should be 0)")
        print(f"  groups  group_type='step'           : {after['step']}  (should be 0)")
        print(f"  group_links link_type='step'        : {after['link_step']}  (should be 0)")
        print(f"  groups  group_type='clustering_run' : {after['c_run']}")
        print(f"  groups  group_type='clustering_step': {after['c_step']}")
        print(f"  groups  group_type='inference_run'  : {after['i_run']}")

        if after["run"] == 0 and after["step"] == 0 and after["link_step"] == 0:
            print("\n[OK] Migration complete.")
        else:
            print("\n[WARN] Some rows still have old type names — investigate manually.")


if __name__ == "__main__":
    main()
