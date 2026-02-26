from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lang import Executor, parse_file, project_to_json


def _cmd_lint(args: argparse.Namespace) -> int:
    project = parse_file(args.file)
    print(f"OK: {project.name}")
    return 0


def _cmd_dump(args: argparse.Namespace) -> int:
    project = parse_file(args.file)
    print(json.dumps(project_to_json(project), indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    src = Path(args.file).resolve()
    project = parse_file(src)
    executor = Executor(project, source_path=src, out_dir=Path(args.out_dir) if args.out_dir else None)

    pipeline_info = None
    if args.pipeline or project.pipelines:
        pipeline_info = executor.run_pipeline(args.pipeline)

    if args.export_json:
        executor.export_json(args.export_json)
    if args.export_html:
        executor.export_html(args.export_html)

    snapshot = executor.snapshot()
    print(json.dumps({"project": project.name, "pipeline": pipeline_info, "snapshot": snapshot}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nexusflow",
        description="NexusFlow DSL for rapid simulation/web/AI prototype definitions.",
    )
    sp = p.add_subparsers(dest="command", required=True)

    lint_p = sp.add_parser("lint", help="Parse and validate a NexusFlow file")
    lint_p.add_argument("file")
    lint_p.set_defaults(func=_cmd_lint)

    dump_p = sp.add_parser("dump", help="Parse and print JSON IR")
    dump_p.add_argument("file")
    dump_p.set_defaults(func=_cmd_dump)

    run_p = sp.add_parser("run", help="Run a program and optional pipeline")
    run_p.add_argument("file")
    run_p.add_argument("--pipeline", help="Pipeline name (defaults to first)")
    run_p.add_argument("--out-dir", help="Base directory for relative exports")
    run_p.add_argument("--export-json", help="Export snapshot JSON after run")
    run_p.add_argument("--export-html", help="Export preview HTML after run")
    run_p.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
