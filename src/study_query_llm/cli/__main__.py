"""python -m study_query_llm.cli [sweep-worker|analyze] ..."""

from __future__ import annotations

import sys


def _main() -> None:
    argv = list(sys.argv[1:])
    if not argv:
        print(
            "Usage: python -m study_query_llm.cli sweep-worker --request-id ID ...\n"
            "       python -m study_query_llm.cli analyze --request-id ID [--dry-run]",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = argv[0]
    rest = argv[1:]
    if cmd in ("sweep-worker", "worker"):
        from study_query_llm.experiments.sweep_worker_main import main as worker_main

        worker_main(rest)
        return
    if cmd in ("analyze", "sweep-analyze"):
        from study_query_llm.analysis.mcq_analyze_request import main as analyze_main

        analyze_main(rest)
        return
    # Back-compat: treat first token as worker flag if it looks like --request-id
    if cmd.startswith("--"):
        from study_query_llm.experiments.sweep_worker_main import main as worker_main

        worker_main(argv)
        return
    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    _main()
