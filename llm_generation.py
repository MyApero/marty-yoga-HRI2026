#!/usr/bin/env python3
"""Generate CSV files to inspect LLM responses for all Marty Yoga prompts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import tomli

from src.speak_prompts import (
    CORRECTIVE_FEEDBACK_MODEL,
    INTRO_MODEL,
    build_corrective_feedback_messages,
    build_end_pose_feedback_messages,
    build_intro_messages,
    build_load_pose_messages,
    build_show_pose_messages,
)

PROMPT_NAMES = (
    "intro",
    "show_pose",
    "load_pose",
    "corrective_feedback",
    "end_pose_feedback",
)

CSV_COLUMNS = (
    "timestamp_utc",
    "prompt_name",
    "pose_name",
    "model",
    "latency_ms",
    "messages_json",
    "response_text",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate debug CSV outputs for all prompt families using local Ollama models."
        )
    )
    parser.add_argument("--runs", type=int, default=1, help="How many runs per prompt.")
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=Path("poses"),
        help="Directory containing pose subfolders with pose.toml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm-response/generated"),
        help="Destination for generated CSV files.",
    )
    parser.add_argument(
        "--general-model",
        default=INTRO_MODEL,
        help="Model used for intro/show/load/end-pose prompts.",
    )
    parser.add_argument(
        "--corrective-model",
        default=CORRECTIVE_FEEDBACK_MODEL,
        help="Model used for corrective-feedback prompts.",
    )
    parser.add_argument(
        "--only-prompt",
        action="append",
        choices=PROMPT_NAMES,
        help="Run only selected prompt families. Repeat to select multiple.",
    )
    parser.add_argument(
        "--limit-poses",
        type=int,
        default=None,
        help="Optional limit on number of poses to process.",
    )
    parser.add_argument(
        "--pose-name",
        default=None,
        help="Filter jobs to a single pose name (e.g. chair).",
    )
    parser.add_argument(
        "--single-job",
        action="store_true",
        help=(
            "Run only one generation job after filters and repeat it --runs times. "
            "Useful for prompt iteration."
        ),
    )
    parser.add_argument(
        "--single-output-file",
        type=Path,
        default=None,
        help=(
            "Write all generated rows to one CSV file instead of one file per prompt."
        ),
    )
    parser.add_argument(
        "--text-log-file",
        type=Path,
        default=None,
        help=(
            "Optional .log file with only generated text entries, separated by blank lines."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV files instead of overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Ollama, write placeholder responses instead.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomli.load(handle)


def load_poses(poses_dir: Path, limit_poses: int | None) -> List[Dict[str, Any]]:
    poses: List[Dict[str, Any]] = []
    for pose_dir in sorted(poses_dir.iterdir()):
        if not pose_dir.is_dir():
            continue
        pose_file = pose_dir / "pose.toml"
        if not pose_file.exists():
            continue

        pose_data = load_toml(pose_file)
        pose_data["pose_name"] = pose_dir.name
        poses.append(pose_data)

    if limit_poses is not None:
        poses = poses[: max(0, limit_poses)]
    return poses


def build_mock_correction(pose: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    targets = pose.get("pose", {})
    selected_keys: List[str] = []

    for key in ("Right Wrist", "Left Wrist", "R-Wrist", "L-Wrist"):
        if key in targets:
            selected_keys.append(key)

    if not selected_keys:
        selected_keys = list(targets.keys())[:2]

    if not selected_keys:
        selected_keys = ["Right Wrist", "Left Wrist"]

    correction: Dict[str, Dict[str, float]] = {}
    for key in selected_keys[:2]:
        target = float(targets.get(key, 160.0))
        current = max(0.0, min(180.0, target - 10.0))
        correction[key] = {
            "current_angle": round(current, 2),
            "target_angle": round(target, 2),
            "error": round(current - target, 2),
        }
    return correction


def build_mock_end_feedback(pose_name: str) -> Dict[str, Any]:
    return {
        "pose_name": pose_name,
        "overall_feedback": "Decent stability with room to improve alignment consistency.",
        "weak_points": ["hip alignment", "knee extension"],
        "improvement_tip": "Press evenly through both feet and lift through the crown.",
    }


def has_description(pose: Dict[str, Any]) -> bool:
    description = pose.get("description")
    return isinstance(description, dict) and "howto" in description


def build_jobs(
    poses: Iterable[Dict[str, Any]],
    general_model: str,
    corrective_model: str,
    selected_prompts: set[str],
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    if "intro" in selected_prompts:
        jobs.append(
            {
                "prompt_name": "intro",
                "pose_name": "",
                "model": general_model,
                "messages": build_intro_messages(),
            }
        )

    for pose in poses:
        pose_name = str(pose["pose_name"])
        pose_has_description = has_description(pose)

        if "show_pose" in selected_prompts and pose_has_description:
            jobs.append(
                {
                    "prompt_name": "show_pose",
                    "pose_name": pose_name,
                    "model": general_model,
                    "messages": build_show_pose_messages(pose),
                }
            )

        if "load_pose" in selected_prompts and pose_has_description:
            jobs.append(
                {
                    "prompt_name": "load_pose",
                    "pose_name": pose_name,
                    "model": general_model,
                    "messages": build_load_pose_messages(pose),
                }
            )

        if "corrective_feedback" in selected_prompts and pose_has_description:
            jobs.append(
                {
                    "prompt_name": "corrective_feedback",
                    "pose_name": pose_name,
                    "model": corrective_model,
                    "messages": build_corrective_feedback_messages(
                        build_mock_correction(pose), pose
                    ),
                }
            )

        if "end_pose_feedback" in selected_prompts:
            jobs.append(
                {
                    "prompt_name": "end_pose_feedback",
                    "pose_name": pose_name,
                    "model": general_model,
                    "messages": build_end_pose_feedback_messages(
                        build_mock_end_feedback(pose_name)
                    ),
                }
            )

    return jobs


def call_ollama(model: str, messages: List[Dict[str, str]]) -> str:
    import ollama

    response = ollama.chat(model=model, messages=messages)
    return str(response.get("message", {}).get("content", "")).strip()


def write_rows(output_file: Path, rows: List[Dict[str, str]], append: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not append or not output_file.exists()
    mode = "a" if append else "w"

    with output_file.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_text_log(output_file: Path, rows: List[Dict[str, str]], append: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with output_file.open(mode, encoding="utf-8") as handle:
        for row in rows:
            text = row.get("response_text", "")
            handle.write(text)
            handle.write("\n\n\n")


def main() -> int:
    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    selected_prompts = set(args.only_prompt or PROMPT_NAMES)
    poses = load_poses(args.poses_dir, args.limit_poses)
    jobs = build_jobs(
        poses=poses,
        general_model=args.general_model,
        corrective_model=args.corrective_model,
        selected_prompts=selected_prompts,
    )

    if args.pose_name:
        jobs = [job for job in jobs if job["pose_name"] == args.pose_name]

    if args.single_job:
        if not jobs:
            raise SystemExit(
                "No generation jobs found for --single-job. Check --only-prompt and --pose-name."
            )
        jobs = [jobs[0]]
        print(
            "Single-job mode selected: "
            f"{jobs[0]['prompt_name']} pose={jobs[0]['pose_name'] or '-'}"
        )

    if not jobs:
        raise SystemExit(
            "No generation jobs found. Check --poses-dir, --only-prompt, and --pose-name."
        )

    print(f"Generating {len(jobs)} prompt jobs x {args.runs} run(s)...")

    rows_by_prompt: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    generated_rows: List[Dict[str, str]] = []

    for run_index in range(1, args.runs + 1):
        for job_index, job in enumerate(jobs, start=1):
            print(
                f"[{run_index}/{args.runs}] job {job_index}/{len(jobs)} "
                f"{job['prompt_name']} pose={job['pose_name'] or '-'} model={job['model']}"
            )
            started = time.perf_counter()
            response_text = ""
            error = ""

            try:
                if args.dry_run:
                    response_text = (
                        f"[DRY RUN] {job['prompt_name']} "
                        f"for pose={job['pose_name'] or 'none'}"
                    )
                else:
                    response_text = call_ollama(job["model"], job["messages"])
            except Exception as exc:
                error = str(exc)

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            row = {
                "timestamp_utc": utc_now_iso(),
                "prompt_name": job["prompt_name"],
                "pose_name": job["pose_name"],
                "model": job["model"],
                "latency_ms": f"{elapsed_ms:.2f}",
                "messages_json": json.dumps(job["messages"], ensure_ascii=False),
                "response_text": response_text,
                "error": error,
            }
            rows_by_prompt[job["prompt_name"]].append(row)
            generated_rows.append(row)

    if args.single_output_file is not None:
        write_rows(args.single_output_file, generated_rows, append=args.append)
        print(f"Wrote {len(generated_rows)} rows -> {args.single_output_file}")
    else:
        for prompt_name, rows in rows_by_prompt.items():
            output_file = args.output_dir / f"{prompt_name}.csv"
            write_rows(output_file, rows, append=args.append)
            print(f"Wrote {len(rows)} rows -> {output_file}")

    text_log_file = args.text_log_file
    if text_log_file is None and args.single_output_file is not None:
        text_log_file = args.single_output_file.with_suffix(".log")

    if text_log_file is not None:
        write_text_log(text_log_file, generated_rows, append=args.append)
        print(f"Wrote {len(generated_rows)} text entries -> {text_log_file}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
