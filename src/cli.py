"""Folder-level orchestrator: Stage A (retrieval + deepsearch) -> Stage C (judge).

Usage::

    python -m src.cli <folder>

Common flags::

    [--manifest Edit.json]
    [--output _results]
    [--limit N]
    [--skip-judge]
    [--resume / --no-resume]
    [--judge-model ...]
    ... plus retrieval hyperparams (--top-k, --max-deepsearch-rounds, ...)

Outputs:
    <folder>/<output>/per_video/<video_id>.json
    <folder>/<output>/summary.json
    <folder>/<output>/summary.md
    <folder>/<output>/logs/run_<ts>_<vid>.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.judge import judge_points  # noqa: E402
from src.agent.oracle_eval import (  # noqa: E402
    extract_groundtruth,
    load_manifest_entries,
)
from src.agent_pipeline_v2 import VisualRetrievalAgentV2  # noqa: E402
from src.utils.config import OPENAI_MODEL, get_llm_client  # noqa: E402
from src.utils.pipeline_log import default_log_path, tee_stdout  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deepsearch retrieval + judge on a folder of forged videos.",
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to a folder containing forged .mp4 files and a manifest JSON (default name: Edit.json).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Edit.json",
        help="Manifest filename inside <folder> (default: Edit.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="_results",
        help="Output subdirectory under <folder> for per-video results & summary (default: _results).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N manifest entries (for debugging).",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default="",
        help="Comma-separated list of video ids to restrict the run to.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run retrieval but skip judge (useful when GT is not yet filled in).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="If a per-video result already exists, reuse it (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force re-run even when a per-video result exists.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the LLM model for Stage C judging (default: $OPENAI_MODEL).",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help=(
            "Fast retrieval eval: skip visual verify. Only checks "
            "whether the candidate YouTube id is in manifest expected_source_ids."
        ),
    )
    # Stage A hyperparameters
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--total-sample-frames", type=int, default=64)
    parser.add_argument("--query-temperature", type=float, default=0.4)
    parser.add_argument(
        "--download-dir",
        type=str,
        default="downloads",
        help="Where to save downloaded candidate videos (relative path resolved from CWD).",
    )
    parser.add_argument("--quiet", action="store_true")
    # Deepsearch-specific hyperparameters
    parser.add_argument(
        "--max-reflect-rounds",
        type=int,
        default=3,
        help="Max REFLECT rounds when search fails (default 3).",
    )
    parser.add_argument(
        "--candidate-sample-frames",
        type=int,
        default=64,
        help="Frames sampled from each candidate video for fine extraction (default 64).",
    )
    parser.add_argument(
        "--max-deepsearch-rounds",
        type=int,
        default=5,
        help="Max search rounds before giving up (default 5).",
    )
    parser.add_argument(
        "--coarse-sample-frames",
        type=int,
        default=16,
        help="Frames sampled for coarse relevance filter (default 16).",
    )
    parser.add_argument(
        "--use-cot",
        dest="use_cot",
        action="store_true",
        default=True,
        help="Use COT reasoning for keyword generation (default).",
    )
    parser.add_argument(
        "--no-cot",
        dest="use_cot",
        action="store_false",
        help="Skip COT, use filename as fallback keyword (ablation mode).",
    )
    parser.add_argument(
        "--parallel-videos",
        type=int,
        default=1,
        help="Number of videos to process in parallel (default 1 = sequential).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory path (default: <output>/logs).",
    )
    return parser


def _filter_entries(
    entries: list[dict[str, Any]],
    *,
    only_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if only_ids:
        keep = set(only_ids)
        entries = [e for e in entries if str(e.get("id") or "").strip() in keep]
    if limit is not None and limit > 0:
        entries = entries[:limit]
    return entries


def _gather_local_sources(
    *,
    retrieved_truth_ids: list[str],
    source_video_paths: list[str],
    downloads_dir: Path,
    dataset_folder: Path,
) -> tuple[list[str], list[str]]:
    """Return (final_source_paths, final_truth_ids_used).

    Prefer the per-truth-id YouTube id when we have it (downloaded by the agent
    or already lying next to the manifest in the dataset folder). The agent
    already returns local paths via `source_video_paths`; we just verify them
    and supplement with any expected truth ids that happen to be present locally.
    """
    out_paths: list[str] = []
    out_ids: list[str] = []

    # 1. Trust the agent's reported paths first.
    for p in source_video_paths:
        if p and Path(p).is_file():
            if p not in out_paths:
                out_paths.append(p)

    # 2. For any retrieved_truth_id that maps to a known file (downloads or dataset
    #    folder) also include it. This is mostly redundant but bullet-proofs the
    #    case where the agent forgot to record the local path.
    for yid in retrieved_truth_ids or []:
        out_ids.append(yid)
        candidates = [
            downloads_dir / f"{yid}.mp4",
            downloads_dir / f"{yid}.mkv",
            downloads_dir / f"{yid}.webm",
            dataset_folder / f"{yid}.mp4",
            dataset_folder / f"{yid}.mkv",
            dataset_folder / f"{yid}.webm",
        ]
        for c in candidates:
            if c.is_file() and str(c) not in out_paths:
                out_paths.append(str(c))
                break
    return out_paths, out_ids


def _write_summary_md(summary: dict[str, Any], out_md: Path) -> None:
    """Write a human-readable markdown summary."""
    lines: list[str] = []
    lines.append(f"# Forgery Pipeline Summary")
    lines.append("")
    lines.append(f"- Folder: `{summary['folder']}`")
    lines.append(f"- Manifest: `{summary['manifest']}`")
    lines.append(f"- Videos processed: **{summary['n_videos']}**")
    lines.append(f"- Videos with GT (judged): **{summary['n_judged']}**")
    lines.append("")
    r = summary["retrieval"]
    lines.append("## Stage A: retrieval")
    lines.append("")
    lines.append(f"- Videos with at least one truth hit: **{r['videos_with_any_truth_hit']}** / {summary['n_videos']}")
    lines.append(f"- Retrieval rate (video-level): **{r['retrieval_rate']:.3f}**")
    lines.append(f"- Per-task retrieval rate: `{r['per_task_breakdown']}`")
    lines.append("")
    s = summary["scoring"]
    lines.append("## Stage C: forgery-point scoring")
    lines.append("")
    if s["total_points"] == 0:
        lines.append("_No GT-bearing videos in this run; scoring skipped._")
    else:
        lines.append(f"- Hits: **{s['hit_points']}** / {s['total_points']}")
        lines.append(f"- Accuracy: **{s['accuracy']:.3f}**")
        lines.append(f"- Per-task accuracy: `{s['per_task_breakdown']}`")
    lines.append("")
    lines.append("## Per-video table")
    lines.append("")
    lines.append("| Video ID | Task | Truth hit | Resolved groups | Score |")
    lines.append("|---|---|---|---|---|")
    for row in summary["per_video"]:
        score_str = '-' if row.get('score') is None else f'{row["score"]:.3f}'
        lines.append(
            f"| `{row['video_id']}` | {row.get('task', '')} | "
            f"{'YES' if row.get('truth_hit') else 'no'} | "
            f"{row.get('resolved_groups', 0)}/{row.get('total_groups', 0)} | "
            f"{score_str} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_partial_record(*, entry: dict[str, Any], video_id: str, error: str) -> dict[str, Any]:
    """Build a minimal record when a video is interrupted before completion."""
    return {
        "video_id": video_id,
        "topic": str(entry.get("topic") or ""),
        "task": str(entry.get("task") or ""),
        "retrieval": {"matched_urls": [], "retrieved_truth_ids": [], "truth_hit": False},
        "deepsearch": {"collected_forgery_points": [], "evidence_videos": []},
        "tokens": {},
        "analysis": {"mode": "interrupted", "points": []},
        "score": None,
        "error": error,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _process_one(
    *,
    entry: dict[str, Any],
    dataset_folder: Path,
    output_dir: Path,
    downloads_dir: Path,
    agent: VisualRetrievalAgentV2,
    judge_model: str,
    judge_client,
    skip_judge: bool,
    resume: bool,
    log_dir: Path,
    parallel_mode: bool = False,
) -> dict[str, Any] | None:
    """Run A->C on one video. Returns the per-video result dict or None if skipped."""
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        return None
    video_path = dataset_folder / f"{video_id}.mp4"
    if not video_path.is_file():
        print(f"[Orchestrator] skip {video_id}: video file not found ({video_path})", flush=True)
        return None

    per_video_path = output_dir / "per_video" / f"{video_id}.json"
    if resume and per_video_path.is_file():
        try:
            cached = json.loads(per_video_path.read_text(encoding="utf-8"))
            print(f"[Orchestrator] reuse cached result for {video_id} ({per_video_path})", flush=True)
            return cached
        except Exception:
            print(f"[Orchestrator] cached result unreadable for {video_id}; re-running", flush=True)

    log_path = default_log_path(log_dir, video_id)
    print(f"[Orchestrator] {video_id}: log -> {log_path}", flush=True)

    if parallel_mode:
        # In parallel mode, tee_stdout modifies global sys.stdout and would
        # interleave all threads.  Instead, let the agent write its own log
        # file via _log → _log_fh.
        try:
            if hasattr(agent, "set_log_file"):
                agent.set_log_file(str(log_path))
            return _process_one_inner(
                entry=entry, video_id=video_id, video_path=video_path,
                dataset_folder=dataset_folder, output_dir=output_dir,
                downloads_dir=downloads_dir, agent=agent,
                judge_model=judge_model, judge_client=judge_client,
                skip_judge=skip_judge,
                per_video_path=per_video_path,
            )
        finally:
            if hasattr(agent, "set_log_file"):
                agent.set_log_file(None)
    else:
        with tee_stdout(log_path):
            return _process_one_inner(
                entry=entry, video_id=video_id, video_path=video_path,
                dataset_folder=dataset_folder, output_dir=output_dir,
                downloads_dir=downloads_dir, agent=agent,
                judge_model=judge_model, judge_client=judge_client,
                skip_judge=skip_judge,
                per_video_path=per_video_path,
            )


def _process_one_inner(
    *,
    entry: dict[str, Any],
    video_id: str,
    video_path: Path,
    dataset_folder: Path,
    output_dir: Path,
    downloads_dir: Path,
    agent: VisualRetrievalAgentV2,
    judge_model: str,
    judge_client,
    skip_judge: bool,
    per_video_path: Path,
) -> dict[str, Any]:
    """Core logic of _process_one, extracted so tee_stdout wrapping is clean."""
    print(f"=== Stage A: retrieval ({video_id}) ===", flush=True)
    try:
        retrieval = agent.run_retrieval(str(video_path))
    except KeyboardInterrupt:
        # Save partial result so user can see progress
        print(f"[Orchestrator] {video_id}: interrupted during retrieval; saving partial result", flush=True)
        partial = _build_partial_record(entry=entry, video_id=video_id, error="interrupted_during_retrieval")
        per_video_path.parent.mkdir(parents=True, exist_ok=True)
        per_video_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        return partial
    except Exception as exc:
        # Save partial result on any error so we can debug
        print(f"[Orchestrator] {video_id}: retrieval failed: {exc}", flush=True)
        partial = _build_partial_record(entry=entry, video_id=video_id, error=f"retrieval_failed: {exc}")
        per_video_path.parent.mkdir(parents=True, exist_ok=True)
        per_video_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        return partial

    retrieved_truth_ids = retrieval.get("retrieved_truth_ids") or []
    local_sources, used_ids = _gather_local_sources(
        retrieved_truth_ids=retrieved_truth_ids,
        source_video_paths=retrieval.get("source_video_paths") or [],
        downloads_dir=downloads_dir,
        dataset_folder=dataset_folder,
    )

    # Forgery points are collected during Stage A (deepsearch).
    # Skip Stage B entirely and use collected_forgery_points directly.
    collected_points = retrieval.get("collected_forgery_points") or []
    analysis = {
        "mode": "deepsearch",
        "summary_zh": "",
        "summary_en": "",
        "points": collected_points,
    }
    print(
        f"[Orchestrator] {video_id}: using {len(collected_points)} "
        f"collected forgery points (skipping Stage B)",
        flush=True,
    )

    gt_points = extract_groundtruth(entry)
    score_result: dict[str, Any] | None = None
    if not skip_judge and gt_points:
        print(f"=== Stage C: judging ({video_id}) ===", flush=True)
        try:
            score_result = judge_points(
                video_id=video_id,
                gt_points=gt_points,
                pred_points=analysis.get("points") or [],
                client=judge_client,
                model=judge_model,
                logger=lambda m: print(m, flush=True),
            )
        except Exception as exc:
            print(f"[Orchestrator] {video_id}: Stage C failed: {exc}", flush=True)
            score_result = {
                "video_id": video_id,
                "hits": 0,
                "score": 0.0,
                "matches": [],
                "comment": "",
                "ok": False,
                "error": f"judge_failed: {exc}",
            }
    elif skip_judge:
        print(f"[Orchestrator] {video_id}: --skip-judge in effect; skipping Stage C", flush=True)
    else:
        print(f"[Orchestrator] {video_id}: no groundtruth in manifest; Stage C skipped", flush=True)

    expected_ids = retrieval.get("oracle_eval", {}).get("expected_source_ids", []) or []
    record = {
        "video_id": video_id,
        "topic": str(entry.get("topic") or ""),
        "task": str(entry.get("task") or ""),
        "manifest_entry_snapshot": {
            "id": entry.get("id"),
            "topic": entry.get("topic"),
            "task": entry.get("task"),
            "video2": entry.get("video2", ""),
            "video3": entry.get("video3", ""),
        },
        "retrieval": {
            "input_video_id": retrieval.get("input_video_id"),
            "matched_urls": retrieval.get("matched_urls", []),
            "matched_youtube_ids": retrieval.get("matched_youtube_ids", []),
            "retrieved_truth_ids": retrieved_truth_ids,
            "source_video_paths": local_sources,
            "forgery_type": retrieval.get("forgery_type"),
            "is_multi_source": retrieval.get("is_multi_source"),
            "stats": retrieval.get("stats", {}),
            "expected_source_ids": expected_ids,
            "truth_hit": bool(retrieved_truth_ids),
            "search_only": bool(retrieval.get("search_only", False)),
        },
        "deepsearch": {
            "collected_forgery_points": retrieval.get("collected_forgery_points", []),
            "evidence_videos": retrieval.get("evidence_videos", []),
        },
        "tokens": retrieval.get("tokens", {}),
        "analysis": analysis,
        "ground_truth": gt_points,
        "score": score_result,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    per_video_path.parent.mkdir(parents=True, exist_ok=True)
    per_video_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Orchestrator] {video_id}: wrote {per_video_path}", flush=True)
    return record


def _build_summary(
    *,
    folder: Path,
    manifest: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    n_videos = len(records)
    n_judged = sum(1 for r in records if (r.get("score") or {}).get("ok"))

    retrieval_hits = sum(1 for r in records if r.get("retrieval", {}).get("truth_hit"))
    per_task_retr: dict[str, list[bool]] = defaultdict(list)
    per_task_score: dict[str, list[float]] = defaultdict(list)
    hit_points = 0
    total_points = 0
    per_video_rows = []
    for r in records:
        task = r.get("task") or "?"
        truth_hit = bool(r.get("retrieval", {}).get("truth_hit"))
        per_task_retr[task].append(truth_hit)
        score_obj = r.get("score") or {}
        if score_obj.get("ok"):
            sc = float(score_obj.get("score", 0.0))
            per_task_score[task].append(sc)
            hits = int(score_obj.get("hits", 0))
            n_gt = int(score_obj.get("n_gt", 3) or 3)
            hit_points += hits
            total_points += n_gt
        per_video_rows.append(
            {
                "video_id": r.get("video_id"),
                "task": task,
                "truth_hit": truth_hit,
                "resolved_groups": r.get("retrieval", {}).get("stats", {}).get("resolved_groups", 0),
                "total_groups": r.get("retrieval", {}).get("stats", {}).get("total_groups", 0),
                "score": (score_obj.get("score") if score_obj.get("ok") else None),
                "matched_urls": r.get("retrieval", {}).get("matched_urls", []),
            }
        )

    retrieval_rate = retrieval_hits / n_videos if n_videos else 0.0
    accuracy = hit_points / total_points if total_points else 0.0

    # Token aggregation
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    for r in records:
        t = r.get("tokens") or {}
        total_prompt_tokens += int(t.get("prompt_tokens", 0))
        total_completion_tokens += int(t.get("completion_tokens", 0))
        total_all_tokens += int(t.get("total_tokens", 0))

    return {
        "folder": str(folder),
        "manifest": manifest,
        "n_videos": n_videos,
        "n_judged": n_judged,
        "retrieval": {
            "videos_with_any_truth_hit": retrieval_hits,
            "retrieval_rate": round(retrieval_rate, 4),
            "per_task_breakdown": {
                t: round(sum(v) / len(v), 4) if v else 0.0
                for t, v in sorted(per_task_retr.items())
            },
        },
        "scoring": {
            "total_points": total_points,
            "hit_points": hit_points,
            "accuracy": round(accuracy, 4),
            "per_task_breakdown": {
                t: round(sum(v) / len(v), 4) if v else 0.0
                for t, v in sorted(per_task_score.items())
            },
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_all_tokens,
            "per_video_avg_tokens": round(total_all_tokens / n_videos) if n_videos else 0,
        },
        "per_video": per_video_rows,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    manifest_path = folder / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    output_dir = folder / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_video").mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_subdir = "logs"
        log_dir = output_dir / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = Path(args.download_dir).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    entries_all = load_manifest_entries(manifest_path)
    only_ids = [s.strip() for s in (args.only_ids or "").split(",") if s.strip()]
    entries = _filter_entries(entries_all, only_ids=only_ids or None, limit=args.limit)
    print(
        f"[Orchestrator] folder={folder} manifest={manifest_path.name} "
        f"entries={len(entries)}/{len(entries_all)} output={output_dir} log_dir={log_dir}",
        flush=True,
    )

    agent = VisualRetrievalAgentV2(
        top_k=args.top_k,
        total_sample_frames=args.total_sample_frames,
        candidate_sample_frames=args.candidate_sample_frames,
        max_reflect_rounds=args.max_reflect_rounds,
        download_output_dir=str(downloads_dir),
        oracle_manifest_path=str(manifest_path),
        search_only=args.search_only,
        query_temperature=args.query_temperature,
        infra_consecutive_threshold=3,
        verbose=not args.quiet,
        max_deepsearch_rounds=args.max_deepsearch_rounds,
        coarse_sample_frames=args.coarse_sample_frames,
        use_cot=args.use_cot,
    )
    print(
        f"[Orchestrator] deepsearch hyperparameters: "
        f"total_sample_frames={args.total_sample_frames}, "
        f"top_k={args.top_k}, "
        f"candidate_sample_frames={args.candidate_sample_frames}, "
        f"coarse_sample_frames={args.coarse_sample_frames}, "
        f"max_deepsearch_rounds={args.max_deepsearch_rounds}, "
        f"max_reflect_rounds={args.max_reflect_rounds}, "
        f"query_temperature={args.query_temperature}, "
        f"use_cot={args.use_cot}",
        flush=True,
    )

    judge_model = args.judge_model or OPENAI_MODEL
    client = get_llm_client()

    records: list[dict[str, Any]] = []

    def _process_one_wrapper(entry):
        """Wrapper that catches exceptions per-video for thread safety."""
        try:
            return _process_one(
                entry=entry,
                dataset_folder=folder,
                output_dir=output_dir,
                downloads_dir=downloads_dir,
                agent=agent,
                judge_model=judge_model,
                judge_client=client,
                skip_judge=args.skip_judge,
                resume=args.resume,
                log_dir=log_dir,
                parallel_mode=(n_workers > 1),
            )
        except KeyboardInterrupt:
            vid = str(entry.get("id") or "?")
            print(f"[Orchestrator] entry {vid!r}: interrupted", flush=True)
            return None
        except Exception as exc:
            vid = str(entry.get("id") or "?")
            print(f"[Orchestrator] entry {vid!r}: fatal: {exc}", flush=True)
            return None

    n_workers = max(1, args.parallel_videos)
    if n_workers <= 1:
        # Sequential processing (original behavior)
        for entry in entries:
            try:
                result = _process_one_wrapper(entry)
            except KeyboardInterrupt:
                print("\n[Orchestrator] Interrupted by user; saving partial summary.", flush=True)
                break
            if result is not None:
                records.append(result)
    else:
        # Parallel processing with ThreadPoolExecutor
        import concurrent.futures

        print(f"[Orchestrator] parallel mode: {n_workers} workers", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_one_wrapper, entry): entry
                for entry in entries
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        records.append(result)
            except KeyboardInterrupt:
                print("\n[Orchestrator] Interrupted by user; cancelling remaining tasks", flush=True)
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    # Save summary even if some videos were interrupted
    summary = _build_summary(folder=folder, manifest=manifest_path.name, records=records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_md(summary, output_dir / "summary.md")
    print(f"[Orchestrator] wrote {summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
