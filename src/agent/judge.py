"""Stage C: LLM-as-judge for forgery-point scoring.

Given a video's GT 3-5 points and the model's predicted 3-5 points, the judge
returns a structured 0/1 verdict per GT point in {0..n} hits along with
short reasons. A `verdict=1` requires the misleading-point dimension (维度B)
to be "roughly consistent" (see PROMPT_JUDGE_POINTS).
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts import PROMPT_JUDGE_POINTS
from src.utils.agent_helpers import (
    _build_multimodal_content,
    call_vlm_with_retry,
    extract_json_from_text,
)


def _format_points_block(points: list[dict[str, Any]] | list[str], *, label: str) -> str:
    """Render either:
      - a list of {"zh": ..., "en": ..., "manipulation_type": ...} dicts (predictions), or
      - a list of {"description": ...} dicts (deepsearch v2 predictions), or
      - a list of plain strings (GT)
    as a numbered block for the judge prompt."""
    lines: list[str] = []
    for i, p in enumerate(points):
        if isinstance(p, str):
            lines.append(f"[{label}-{i + 1}] {p.strip()}")
        elif isinstance(p, dict):
            # v2 deepsearch uses "description"; v1 uses "zh"
            text = str(p.get("description") or p.get("zh") or "").strip()
            mt = str(p.get("manipulation_type") or "").strip()
            mp = str(p.get("misleading_point") or "").strip()
            if mt or mp:
                lines.append(
                    f"[{label}-{i + 1}] {text}\n"
                    f"    (declared_manipulation_type={mt!r}, declared_misleading_point={mp!r})"
                )
            else:
                lines.append(f"[{label}-{i + 1}] {text}")
        else:
            lines.append(f"[{label}-{i + 1}] {p!r}")
    return "\n".join(lines)


def judge_points(
    *,
    video_id: str,
    gt_points: list[str],
    pred_points: list[dict[str, Any]],
    client,
    model: str,
    temperature: float = 0.0,
    logger=None,
) -> dict[str, Any]:
    """Run LLM-as-judge. Always emits a result dict, even on errors.

    Returns:
        {
          "video_id": str,
          "hits": int in {0,1,2,3},
          "score": float in [0,1],
          "matches": [{"gt_idx": int, "pred_idx": int|None, "verdict": 0/1,
                       "matched_dim_method": bool, "matched_dim_misleading": bool,
                       "reason": str}, ...],
          "comment": str,
          "ok": bool,
          "error": str | None,
        }
    """
    gt = [g for g in (gt_points or []) if g and g.strip()]
    if not gt:
        return {
            "video_id": video_id,
            "hits": 0,
            "score": 0.0,
            "matches": [],
            "comment": "no_groundtruth",
            "ok": False,
            "error": "no_groundtruth",
        }

    gt_block = _format_points_block(gt, label="GT")
    pred_block = _format_points_block(pred_points or [], label="PRED")
    prompt = PROMPT_JUDGE_POINTS.format(gt_block=gt_block, pred_block=pred_block)

    try:
        raw, _tokens = call_vlm_with_retry(
            client,
            model,
            _build_multimodal_content(prompt),
            max_retries=2,
            temperature=temperature,
            json_mode=True,
            logger=logger,
            log_prefix="[Judge] ",
        )
        data = extract_json_from_text(raw)
    except Exception as exc:
        return {
            "video_id": video_id,
            "hits": 0,
            "score": 0.0,
            "matches": [],
            "comment": "",
            "ok": False,
            "error": f"judge_call_failed: {exc}",
        }

    # Parse robustly. We trust judge's `hits` but clamp to [0, 3] and
    # cross-check against `matches`.
    raw_matches = data.get("matches") if isinstance(data, dict) else None
    matches: list[dict[str, Any]] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            if not isinstance(item, dict):
                continue
            try:
                gt_idx = int(item.get("gt_idx", -1))
            except Exception:
                gt_idx = -1
            try:
                p_raw = item.get("pred_idx")
                pred_idx = None if p_raw is None else int(p_raw)
            except Exception:
                pred_idx = None
            try:
                verdict = 1 if int(item.get("verdict", 0)) >= 1 else 0
            except Exception:
                verdict = 0
            matches.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "verdict": verdict,
                    "matched_dim_method": bool(item.get("matched_dim_method", False)),
                    "matched_dim_misleading": bool(item.get("matched_dim_misleading", False)),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
    n_gt = max(1, len(gt))
    # If judge gave a `hits` field trust it (clamped); else derive from matches.
    if isinstance(data, dict) and "hits" in data:
        try:
            hits = int(data.get("hits", 0))
        except Exception:
            hits = sum(m["verdict"] for m in matches)
    else:
        hits = sum(m["verdict"] for m in matches)
    hits = max(0, min(hits, n_gt))
    score = hits / float(n_gt)
    comment = ""
    if isinstance(data, dict):
        comment = str(data.get("comment") or "").strip()

    return {
        "video_id": video_id,
        "hits": hits,
        "score": round(score, 4),
        "n_gt": n_gt,
        "matches": matches,
        "comment": comment,
        "ok": True,
        "error": None,
    }


def sanity_self_judge(
    *,
    client,
    model: str,
    sample_gt: list[str],
    logger=None,
) -> dict[str, Any]:
    """Feed a GT list to the judge AS THE PREDICTION. Expect score == 1.0.

    Useful for catching prompt regressions / model drift before a real run.
    """
    pred = [
        {
            "zh": s,
            "en": "(self-test, same as GT)",
            "manipulation_type": "",
            "misleading_point": "",
        }
        for s in sample_gt
    ]
    result = judge_points(
        video_id="__sanity__",
        gt_points=sample_gt,
        pred_points=pred,
        client=client,
        model=model,
        logger=logger,
    )
    result["passed"] = result.get("ok") and result.get("score", 0.0) >= 0.99
    return result
