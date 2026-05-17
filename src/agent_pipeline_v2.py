"""Stage A v2 (deepsearch): iterative deep-search retrieval agent.

Replaces the flat search+verify loop with a two-stage retrieval pipeline:
  1. Uniform 64-frame sampling (no shot detection)
  2. Single VLM COT call -> overlay triage + temporal analysis + keyword generation
  3. Deepsearch loop (per source group, serial):
     a. Search 1 keyword -> 1 candidate video
     b. Coarse filter: sample 16 frames, VLM judges relevance
     c. Fine filter: sample 64 frames, VLM extracts 3-5 forgery points
     d. Sufficiency check: VLM judges if evidence is complete
     e. If not sufficient, reflect -> new keyword -> repeat (max N rounds)
  4. Return results + collected forgery points

StageC (judge.py) is unchanged. StageB (forgery_analyzer.py) is skipped
since forgery points are already extracted during fine retrieval.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from src.agent.oracle_eval import (
    evaluate_oracle,
    extract_youtube_id,
    load_oracle_map,
)
from src.agent.prompts import (
    PROMPT_COT_RETRIEVAL,
    PROMPT_REFLECT_REFINE,
    PROMPT_VERIFY_MATCH,
)
from src.agent.session_state import CandidateRecord, SessionState
from src.agent.tools import AgentTools
from src.utils.agent_helpers import (
    _build_multimodal_content,
    call_vlm_with_retry,
    extract_json_from_text,
    sanitize_queries,
    uniform_sample_frames,
)


class VisualRetrievalAgentV2:
    """Deepsearch visual retrieval agent (iterative coarse->fine pipeline).

    Public API: `run_retrieval(video_path)` -> dict with `matched_urls`,
    `retrieved_truth_ids`, `source_video_paths`, stats, etc.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        total_sample_frames: int = 64,
        candidate_sample_frames: int = 64,
        max_reflect_rounds: int = 3,
        download_output_dir: str = "downloads",
        oracle_manifest_path: str | None = None,
        search_only: bool = False,
        query_temperature: float = 0.4,
        infra_consecutive_threshold: int = 3,
        verbose: bool = True,
        # Deepsearch-specific params
        max_deepsearch_rounds: int = 5,
        coarse_sample_frames: int = 16,
        use_cot: bool = True,
    ) -> None:
        from src.utils.config import OPENAI_MODEL, get_llm_client

        self.client = get_llm_client()
        self.model = OPENAI_MODEL
        self.top_k = top_k
        self.total_sample_frames = total_sample_frames
        self.candidate_sample_frames = candidate_sample_frames
        self.max_reflect_rounds = max_reflect_rounds
        self.download_output_dir = str(Path(download_output_dir).resolve())
        self.oracle_map = load_oracle_map(oracle_manifest_path)
        self.search_only = bool(search_only)
        self.query_temperature = query_temperature
        self.infra_consecutive_threshold = infra_consecutive_threshold
        self.verbose = verbose
        self.max_deepsearch_rounds = max_deepsearch_rounds
        self.coarse_sample_frames = coarse_sample_frames
        self.use_cot = use_cot
        self._tls = threading.local()

    def set_log_file(self, path: str | None) -> None:
        """Attach/detach a per-video log file for the current thread."""
        old = getattr(self._tls, "log_fh", None)
        if old:
            try:
                old.close()
            except OSError:
                pass
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._tls.log_fh = open(path, "w", encoding="utf-8")
        else:
            self._tls.log_fh = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        fh = getattr(self._tls, "log_fh", None)
        if fh:
            fh.write(line + "\n")
            fh.flush()

    # ------------------------------------------------------------------
    # COT reasoning via VLM
    # ------------------------------------------------------------------
    def _cot_reasoning(self, frame_paths: list[str]) -> dict[str, Any]:
        """Single VLM call: overlay triage + temporal analysis + query generation."""
        self._log(f"[COT] sending {len(frame_paths)} frames to VLM for chain-of-thought analysis")
        contents = _build_multimodal_content(PROMPT_COT_RETRIEVAL, image_paths=frame_paths)
        raw, tokens = call_vlm_with_retry(
            self.client,
            self.model,
            contents,
            max_retries=3,
            temperature=self.query_temperature,
            json_mode=True,
            logger=self._log,
            log_prefix="[COT] ",
        )
        data = extract_json_from_text(raw)

        raw_forbidden = data.get("forbidden_overlay_text") or []
        if not isinstance(raw_forbidden, list):
            raw_forbidden = []
        forbidden = [str(x).strip() for x in raw_forbidden if str(x).strip()]
        estimated_sources = int(data.get("estimated_sources") or 1)

        # Parse source_queries (now 1 query per source)
        raw_source_queries = data.get("source_queries")
        source_queries: list[dict[str, Any]] = []
        if isinstance(raw_source_queries, list) and raw_source_queries:
            for sq in raw_source_queries:
                if not isinstance(sq, dict):
                    continue
                label = str(sq.get("source_label") or "unknown_source").strip()
                qs = sanitize_queries(sq.get("queries"), limit=1)
                if qs:
                    source_queries.append({"source_label": label, "queries": qs})
        elif data.get("queries"):
            qs = sanitize_queries(data.get("queries"), limit=1)
            if qs:
                source_queries.append({"source_label": "source_1", "queries": qs})

        if not source_queries:
            source_queries.append({"source_label": "source_1", "queries": []})

        result = {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "forbidden_overlay_text": forbidden,
            "physical_observations": str(data.get("physical_observations") or "").strip(),
            "temporal_analysis": str(data.get("temporal_analysis") or "").strip(),
            "estimated_sources": estimated_sources,
            "source_queries": source_queries,
            "tokens": tokens,
        }

        all_queries = [q for sq in source_queries for q in sq["queries"]]
        self._log(f"[COT] estimated_sources={estimated_sources}, total_queries={len(all_queries)}")
        for sq in source_queries:
            self._log(f"[COT]   {sq['source_label']}: {sq['queries']}")
        if forbidden:
            self._log(f"[COT] forbidden_overlay_text: {forbidden[:5]}")
        if result["physical_observations"]:
            self._log(f"[COT] physical_observations: {result['physical_observations'][:300]}")
        if result["temporal_analysis"]:
            self._log(f"[COT] temporal_analysis: {result['temporal_analysis'][:300]}")

        return result

    # ------------------------------------------------------------------
    # REFLECT: generate new keyword when search fails
    # ------------------------------------------------------------------
    def _reflect_query(
        self,
        frame_paths: list[str],
        prev_queries: list[str],
        wrong_titles: list[str],
    ) -> str | None:
        """Generate a single fresh keyword when previous one failed."""
        prompt = PROMPT_REFLECT_REFINE.replace(
            "{prev_queries}", json.dumps(prev_queries, ensure_ascii=False)
        ).replace("{candidate_titles}", json.dumps(wrong_titles, ensure_ascii=False))

        contents = _build_multimodal_content(prompt, image_paths=frame_paths)
        raw, tokens = call_vlm_with_retry(
            self.client,
            self.model,
            contents,
            max_retries=3,
            temperature=self.query_temperature,
            json_mode=True,
            logger=self._log,
            log_prefix="[Reflect] ",
        )
        data = extract_json_from_text(raw)
        new_queries = sanitize_queries(data.get("new_queries"), limit=1)
        self._log(f"[Reflect] new_queries: {new_queries}")
        if data.get("reflection"):
            self._log(f"[Reflect] reflection: {str(data['reflection'])[:300]}")
        return new_queries[0] if new_queries else None

    # ------------------------------------------------------------------
    # Deepsearch loop: unified single-loop
    # ------------------------------------------------------------------
    async def _deepsearch(
        self,
        *,
        initial_keyword: str,
        source_descriptions: list[str],
        session: SessionState,
        tools: AgentTools,
        input_frame_paths: list[str],
        cache_root: str,
        cot_physical_observations: str = "",
        cot_temporal_analysis: str = "",
        cot_forbidden_overlay_text: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single unified deepsearch loop.

        Each round: keyword → search → download → coarse → fine → next_step.
        The VLM self-decides what keyword to use next based on all context
        (COT source descriptions, collected points, examined videos).
        """
        collected_points: list[dict[str, Any]] = []
        evidence_videos: list[dict[str, str]] = []
        matched_urls: list[str] = []

        current_keyword = initial_keyword
        wrong_titles: list[str] = []
        all_prev_queries: list[str] = []

        for round_num in range(1, self.max_deepsearch_rounds + 1):
            if not current_keyword:
                self._log("[DeepSearch] no keyword available; stopping")
                break

            self._log(
                f"[DeepSearch] round {round_num}/{self.max_deepsearch_rounds} "
                f"keyword={current_keyword!r}"
            )
            all_prev_queries.append(current_keyword)

            # --- Step 1: Search (async) ---
            search_result = await tools.search_one_async(session, current_keyword)
            if not search_result.get("ok"):
                reason = search_result.get("reason", "unknown")
                self._log(f"[DeepSearch] search failed: {reason}")
                if round_num < self.max_deepsearch_rounds:
                    current_keyword = self._reflect_query(
                        input_frame_paths, all_prev_queries, wrong_titles[-8:]
                    )
                else:
                    current_keyword = None
                continue

            cand_ref = search_result["candidate_ref"]
            cand_title = search_result.get("title", "")
            self._log(f"[DeepSearch] found: {cand_title!r} ref={cand_ref}")

            # --- Step 2: Download (async) ---
            dl_result = await tools.download_candidate_async(
                session, cand_ref, self.download_output_dir
            )
            if not dl_result.get("ok"):
                code = str(dl_result.get("reason_code") or dl_result.get("reason") or "download_failed")
                stderr = str(dl_result.get("stderr_tail") or "")[-300:]
                self._log(f"[DeepSearch] download failed: {code}")
                if stderr:
                    self._log(f"[DeepSearch] download stderr: {stderr}")
                wrong_titles.append(cand_title)
                if round_num < self.max_deepsearch_rounds:
                    current_keyword = self._reflect_query(
                        input_frame_paths, all_prev_queries, wrong_titles[-8:]
                    )
                else:
                    current_keyword = None
                continue

            candidate = session.candidates.get(cand_ref)
            if not candidate:
                wrong_titles.append(cand_title)
                continue

            # --- Step 3: Coarse filter (16 frames -> relevance check) ---
            coarse_result = tools.sample_candidate_frames(
                candidate,
                output_dir="",
                num_frames=self.coarse_sample_frames,
                prefix="coarse",
                cache_root=cache_root,
            )
            if not coarse_result.get("ok"):
                self._log("[DeepSearch] coarse sample failed")
                wrong_titles.append(cand_title)
                continue

            relevance = tools.check_coarse_relevance(
                coarse_result["frame_paths"],
                physical_observations=cot_physical_observations,
                temporal_analysis=cot_temporal_analysis,
            )
            self._log(
                f"[DeepSearch] coarse relevance: {relevance['is_relevant']} "
                f"({relevance['reasoning'][:100]})"
            )

            if not relevance["is_relevant"]:
                wrong_titles.append(cand_title)
                if round_num < self.max_deepsearch_rounds:
                    current_keyword = self._reflect_query(
                        input_frame_paths, all_prev_queries, wrong_titles[-8:]
                    )
                else:
                    current_keyword = None
                continue

            # --- Step 4: Fine filter (64 frames -> forgery points) ---
            fine_result = tools.sample_candidate_frames(
                candidate,
                output_dir="",
                num_frames=self.candidate_sample_frames,
                prefix="fine",
                cache_root=cache_root,
            )
            if not fine_result.get("ok"):
                self._log("[DeepSearch] fine sample failed")
                continue

            forgery_result = tools.extract_fine_forgery_points(
                fine_result["frame_paths"],
                physical_observations=cot_physical_observations,
                temporal_analysis=cot_temporal_analysis,
                forbidden_overlay_text=", ".join(cot_forbidden_overlay_text or []),
            )
            new_points = forgery_result.get("points") or []
            self._log(f"[DeepSearch] fine extraction: {len(new_points)} forgery points")

            if new_points:
                # Deduplicate: skip points too similar to already-collected ones
                unique_new = []
                for pt in new_points:
                    desc = str(pt.get("description") or "").lower()
                    if not desc:
                        continue
                    desc_words = set(desc.split())
                    is_dup = False
                    for existing in collected_points:
                        ex_desc = str(existing.get("description") or "").lower()
                        ex_words = set(ex_desc.split())
                        if not ex_words:
                            continue
                        overlap = len(desc_words & ex_words) / max(len(desc_words), 1)
                        if overlap > 0.55:
                            is_dup = True
                            break
                    if not is_dup:
                        pt["evidence_video_title"] = cand_title
                        pt["evidence_video_url"] = candidate.url or ""
                        pt["evidence_video_ref"] = cand_ref
                        unique_new.append(pt)

                if unique_new:
                    self._log(f"[DeepSearch] after dedup: {len(unique_new)}/{len(new_points)} unique points")
                    collected_points.extend(unique_new)
                    evidence_videos.append({
                        "title": cand_title,
                        "url": candidate.url or "",
                        "ref": cand_ref,
                        "points_count": len(unique_new),
                    })
                    if candidate.url and candidate.url not in matched_urls:
                        matched_urls.append(candidate.url)

            session.verification_events.append({
                "turn": round_num,
                "phase": "deepsearch",
                "candidate_ref": cand_ref,
                "candidate_url": candidate.url or "",
                "candidate_title": cand_title,
                "coarse_relevant": relevance["is_relevant"],
                "fine_points_count": len(new_points),
            })

            # --- Step 5: Sufficiency + next keyword (single VLM call) ---
            try:
                next_step = tools.deepsearch_next_step(
                    collected_points,
                    [{"title": v["title"], "url": v["url"]} for v in evidence_videos],
                    source_descriptions=source_descriptions,
                    prev_queries=all_prev_queries,
                    physical_observations=cot_physical_observations,
                    temporal_analysis=cot_temporal_analysis,
                    forbidden_overlay_text=", ".join(cot_forbidden_overlay_text or []),
                )
            except Exception as exc:
                self._log(f"[DeepSearch] next_step VLM call failed: {exc}")
                if round_num < self.max_deepsearch_rounds:
                    current_keyword = self._reflect_query(
                        input_frame_paths, all_prev_queries, wrong_titles[-8:]
                    )
                else:
                    current_keyword = None
                continue

            self._log(
                f"[DeepSearch] sufficiency: {next_step['is_sufficient']} "
                f"({next_step['reasoning'][:150]})"
            )

            if next_step["is_sufficient"]:
                self._log("[DeepSearch] evidence sufficient; stopping")
                break

            current_keyword = next_step.get("next_keyword", "")
            if current_keyword:
                self._log(f"[DeepSearch] next keyword: {current_keyword!r}")
            else:
                self._log("[DeepSearch] no next keyword provided; stopping")
                break

        return {
            "collected_points": collected_points,
            "evidence_videos": evidence_videos,
            "matched_urls": matched_urls,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_retrieval(self, video_path: str) -> dict[str, Any]:
        """Run Stage A v2 (deepsearch) retrieval on a single forged video.

        Returns the same dict format as v1 agent so StageC works unchanged.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        start_ts = time.time()
        temp_root = tempfile.mkdtemp(prefix="s_deepsearch_v2_")
        Path(self.download_output_dir).mkdir(parents=True, exist_ok=True)
        self._log("========== Deepsearch Agent v2 run started ==========")
        self._log(f"Input video: {video_path}")
        self._log(
            f"Config: model={self.model}, "
            f"total_sample_frames={self.total_sample_frames}, "
            f"top_k={self.top_k}, "
            f"candidate_sample_frames={self.candidate_sample_frames}, "
            f"coarse_sample_frames={self.coarse_sample_frames}, "
            f"max_deepsearch_rounds={self.max_deepsearch_rounds}, "
            f"max_reflect_rounds={self.max_reflect_rounds}, "
            f"query_temperature={self.query_temperature}, "
            f"use_cot={self.use_cot}, "
            f"search_only={self.search_only}"
        )

        # Token tracking
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            # Step 1: Uniform frame sampling
            sampled_root = str(Path(temp_root) / "sampled_frames")
            frame_paths = uniform_sample_frames(
                video_path,
                num_frames=self.total_sample_frames,
                output_dir=sampled_root,
                prefix="frame",
            )
            self._log(f"[Sampling] uniform: {len(frame_paths)} frames")

            # Create a minimal SessionState
            input_video_id = Path(video_path).stem
            expected_ids = self.oracle_map.get(input_video_id, [])
            if expected_ids:
                self._log(f"Oracle eval expected ids: {expected_ids}")

            from src.agent.session_state import ShotGroup

            session = SessionState(
                video_path=video_path,
                input_video_id=input_video_id,
                shots=[{"shot_id": 0, "start_frame": 0, "end_frame": 0, "start_sec": 0.0, "end_sec": 0.0}],
                shot_frame_map={0: frame_paths},
                expected_source_ids=expected_ids,
                max_rounds=100,
                per_group_budget=100,
                infra_consecutive_threshold=self.infra_consecutive_threshold,
            )
            session.groups[0] = ShotGroup(
                group_id=0,
                shot_ids=[0],
                physical_observations="",
                queries=[],
            )

            tools = AgentTools(
                client=self.client,
                model=self.model,
                candidate_sample_frames=self.candidate_sample_frames,
                prompts={
                    "extract": "",
                    "reflect": PROMPT_REFLECT_REFINE,
                    "verify": PROMPT_VERIFY_MATCH,
                    "cluster": "",
                },
                logger=self._log,
                coarse_frames=self.coarse_sample_frames,
                dense_frames=self.total_sample_frames,
                query_temperature=self.query_temperature,
            )
            cache_root = str(Path(temp_root) / "sampled_candidates_cache")

            # Step 2: COT reasoning (or skip for ablation)
            if self.use_cot:
                cot_result = self._cot_reasoning(frame_paths)
                session.groups[0].physical_observations = cot_result["physical_observations"]
                # Track COT tokens
                cot_tokens = cot_result.get("tokens", {})
                for k in total_tokens:
                    total_tokens[k] += cot_tokens.get(k, 0)
            else:
                # No COT: use a generic fallback keyword from the video filename
                self._log("[COT] skipped (use_cot=False); using filename as fallback keyword")
                cot_result = {
                    "reasoning": "COT skipped (ablation)",
                    "forbidden_overlay_text": [],
                    "physical_observations": "",
                    "temporal_analysis": "",
                    "estimated_sources": 1,
                    "source_queries": [
                        {"source_label": "source_1", "queries": [input_video_id.replace("_", " ")]}
                    ],
                    "tokens": {},
                }

            source_queries = cot_result["source_queries"]

            # Build source descriptions for the VLM to reference during deepsearch
            source_descriptions = [
                sq["source_label"] for sq in source_queries
            ]

            # Pick the best initial keyword from COT sources
            initial_keyword = ""
            for sq in source_queries:
                if sq.get("queries"):
                    initial_keyword = sq["queries"][0]
                    break

            cot_phys = cot_result.get("physical_observations", "")
            cot_temp = cot_result.get("temporal_analysis", "")
            cot_forbidden = cot_result.get("forbidden_overlay_text", [])

            # Step 3: Unified deepsearch loop (single loop, VLM decides keyword each round)
            ds_result = asyncio.run(self._deepsearch(
                initial_keyword=initial_keyword,
                source_descriptions=source_descriptions,
                session=session,
                tools=tools,
                input_frame_paths=frame_paths,
                cache_root=cache_root,
                cot_physical_observations=cot_phys,
                cot_temporal_analysis=cot_temp,
                cot_forbidden_overlay_text=cot_forbidden,
            ))

            all_collected_points = ds_result["collected_points"]
            all_evidence_videos = ds_result["evidence_videos"]
            matched_urls = ds_result["matched_urls"]

            # Build output
            elapsed = time.time() - start_ts
            summary = session.to_summary()

            if len(set(matched_urls)) > 1:
                forgery_type = "multi_source_splicing"
            elif len(set(matched_urls)) == 1:
                forgery_type = "single_source"
            else:
                forgery_type = "unresolved"

            matched_youtube_ids: list[str] = []
            source_video_paths: list[str] = []
            for url in matched_urls:
                yid = extract_youtube_id(url)
                if not yid or yid in matched_youtube_ids:
                    continue
                matched_youtube_ids.append(yid)
                for c in session.candidates.values():
                    if c.url == url and c.downloaded_video_path:
                        if c.downloaded_video_path not in source_video_paths:
                            source_video_paths.append(c.downloaded_video_path)
                        break
            retrieved_truth_ids = [
                yid for yid in matched_youtube_ids if yid in set(expected_ids)
            ]

            oracle_eval = evaluate_oracle(
                input_video_id=input_video_id,
                expected_source_ids=expected_ids,
                action_trace=session.action_trace,
                verification_events=session.verification_events,
                group_to_shot_ids={gid: list(g.shot_ids) for gid, g in session.groups.items()},
                matched_group_ids=sorted(session.matched_group_ids()),
            )

            self._log(
                f"Run finished: matched_urls={len(matched_urls)}, "
                f"collected_points={len(all_collected_points)}, "
                f"evidence_videos={len(all_evidence_videos)}, "
                f"elapsed={elapsed:.2f}s, "
                f"total_tokens={total_tokens['total_tokens']}"
            )

            return {
                "input_video_id": input_video_id,
                "matched_urls": matched_urls,
                "matched_youtube_ids": matched_youtube_ids,
                "retrieved_truth_ids": retrieved_truth_ids,
                "source_video_paths": source_video_paths,
                "forgery_type": forgery_type,
                "is_multi_source": len(set(matched_urls)) > 1,
                "search_only": self.search_only,
                "action_trace": session.action_trace,
                "verification_events": session.verification_events,
                # Deepsearch-specific outputs
                "collected_forgery_points": all_collected_points,
                "evidence_videos": all_evidence_videos,
                "stats": {
                    "non_empty_shots": 1,
                    "resolved_shots": summary["resolved_shots"],
                    "unresolved_shots": summary["unresolved_shots"],
                    "skipped_shots": {},
                    "total_groups": 1,
                    "resolved_groups": summary["resolved_groups"],
                    "skipped_groups": {},
                    "tool_calls_used": session.tool_calls_used,
                    "rounds_used": session.round_index,
                    "stop_reason": session.stop_reason or "completed",
                    "elapsed_seconds": round(elapsed, 3),
                },
                "diagnostics": {
                    "agent_version": "v2_deepsearch",
                    "cot_estimated_sources": cot_result.get("estimated_sources", 0),
                    "cot_forbidden_overlay_text": cot_result.get("forbidden_overlay_text", []),
                    "use_cot": self.use_cot,
                    "max_deepsearch_rounds": self.max_deepsearch_rounds,
                    "coarse_sample_frames": self.coarse_sample_frames,
                    "prompt_related": {
                        "search_actions": len(session.verification_events),
                        "reflect_actions": 0,
                        "extract_clues_actions": 0,
                        "verify_actions": 0,
                        "download_actions": 0,
                        "stop_actions": 0,
                        "error_actions": 0,
                        "empty_query_actions": 0,
                    },
                    "tooling_related": {
                        "download_failures": 0,
                        "download_failure_codes": {},
                        "download_failure_samples": [],
                        "sample_failures": 0,
                        "sample_failure_codes": {},
                        "verify_events": len(session.verification_events),
                    },
                    "retrieval_funnel": {
                        "search_events": len(session.verification_events),
                        "verification_events": len(session.verification_events),
                        "verified_matches": len(all_evidence_videos),
                        "back_propagation_events": 0,
                        "back_propagation_hits": 0,
                    },
                    "groups": {
                        "total": 1,
                        "resolved": len(session.matched_group_ids()),
                    },
                    "token_usage": total_tokens,
                },
                "oracle_eval": oracle_eval,
                "tokens": total_tokens,
            }
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            self._log("Temporary workspace cleaned")
            self._log("========== Deepsearch Agent v2 run ended ==========")


def main() -> None:
    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
