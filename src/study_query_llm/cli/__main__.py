"""python -m study_query_llm.cli [COMMAND] ..."""

from __future__ import annotations

import sys


def _print_usage() -> None:
    print(
        "Usage:\n"
        "  python -m study_query_llm.cli sweep-worker --request-id ID ...\n"
        "  python -m study_query_llm.cli analyze --request-id ID [--dry-run]\n"
        "  python -m study_query_llm.cli jobs langgraph-worker --request-id ID ...\n"
        "  python -m study_query_llm.cli jobs cached-supervisor --request-id ID --engine MODEL ...\n"
        "  python -m study_query_llm.cli sweep engine-supervisor --request-id ID ...\n"
        "  python -m study_query_llm.cli sweep run-bigrun [--create-request] ...\n",
        file=sys.stderr,
        end="",
    )


def _main() -> None:
    argv = list(sys.argv[1:])
    if not argv:
        _print_usage()
        sys.exit(2)

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "jobs" and rest:
        sub = rest[0]
        sub_rest = rest[1:]
        if sub == "langgraph-worker":
            from study_query_llm.services.jobs.runtime_workers import main_langgraph_worker

            raise SystemExit(main_langgraph_worker(sub_rest))
        if sub == "cached-supervisor":
            from study_query_llm.services.jobs.runtime_supervisors import main_cached_job_supervisor

            raise SystemExit(main_cached_job_supervisor(sub_rest))
        print(f"Unknown jobs subcommand: {sub!r}", file=sys.stderr)
        sys.exit(2)

    if cmd == "sweep" and rest:
        sub = rest[0]
        sub_rest = rest[1:]
        if sub == "engine-supervisor":
            from study_query_llm.services.jobs.runtime_supervisors import main_engine_supervisor

            raise SystemExit(main_engine_supervisor(sub_rest))
        if sub == "run-bigrun":
            from study_query_llm.experiments.runtime_sweeps import main_bigrun_sync

            main_bigrun_sync(sub_rest)
            return
        print(f"Unknown sweep subcommand: {sub!r}", file=sys.stderr)
        sys.exit(2)

    if cmd in ("sweep-worker", "worker"):
        from study_query_llm.experiments.sweep_worker_main import main as worker_main

        worker_main(rest)
        return
    if cmd in ("analyze", "sweep-analyze"):
        from study_query_llm.analysis.mcq_analyze_request import main as analyze_main

        analyze_main(rest)
        return
    if cmd.startswith("--"):
        from study_query_llm.experiments.sweep_worker_main import main as worker_main

        worker_main(argv)
        return
    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    _main()
