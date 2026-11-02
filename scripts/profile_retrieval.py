"""Where does the retrieval time go, and which setting makes it faster WITHOUT hurting quality?

Run from the project root (the project's virtual environment active):
    python scripts/profile_retrieval.py                 # 6 checklist questions, default variants
    python scripts/profile_retrieval.py --n 15 --variants 20:512 10:512 20:256 10:256 8:256

A variant is  fused_k:max_length  (how many candidates the cross-encoder scores : tokens per pair).
Quality proxy: for each of the 15 PDPL checklist questions the right article (document + number) is known, so we
measure how often it is still in the top-5 after reranking. Compare against the baseline row.
NOTE: the abstention threshold was calibrated on the reranker scores of the baseline. If you change max_length,
the model, or quantise it, the score distribution moves: recalibrate the threshold and re-run E5.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "gap_analyzer", ROOT / "app"):
    sys.path.insert(0, str(p))

CHECKLIST = ROOT / "data" / "pdpl_checklist" / "member1_pdpl_compliance_checklist.csv"


def hardware_report():
    lines = []
    try:
        import torch
        lines.append(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
                     + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> everything runs on CPU"))
        lines.append(f"torch CPU threads: {torch.get_num_threads()}")
    except Exception as exc:
        lines.append(f"torch not importable: {exc}")
    return lines


def _length_attr(model):
    """Newer sentence-transformers renamed `max_length` to `max_seq_length` (the old name is deprecated and may not
    take effect). Prefer the new name; fall back to the old one."""
    for name in ("max_seq_length", "max_length"):
        try:
            if getattr(model, name, None) is not None:
                return name
        except Exception:
            continue
    return None


def set_max_length(pipe, length):
    """Set the reranker's max tokens per pair. Returns (previous_value, attribute_name_used)."""
    rr = getattr(pipe, "reranker", None)
    if rr is None:
        return None, None
    model = rr.model
    name = _length_attr(model)
    if name is None:
        return None, None
    old = getattr(model, name)
    setattr(model, name, length)
    return old, name


def article_key(doc):
    import gap_service as gs
    prefix = str(doc.get("source", "")).lower().split("_")[0]
    return (prefix, gs._num(doc.get("article_number", "")))


def run_variant(pipe, reqs, fused_k, max_length, baseline_top1=None, log=print):
    import gap_service as gs
    old, attr = set_max_length(pipe, max_length)
    if attr is not None:
        now = getattr(pipe.reranker.model, attr)
        log(f"  [variant fused_k={fused_k} max_length={max_length}] reranker.{attr} = {now}"
            + ("" if now == max_length else "   !! the setting did NOT take effect"))
    rows, top1 = [], {}
    try:
        for i, r in enumerate(reqs, 1):
            res = pipe.retrieve(r.verification_question, fused_k=fused_k)
            t = res["timings"]
            hit = gs.pick_article(res["final"], r.source_document, r.article_number) is not None
            top1[r.requirement_id] = article_key(res["final"][0]) if res["final"] else None
            rows.append({"retrieval_s": t["retrieval_s"], "rerank_s": t["rerank_s"], "hit5": hit,
                         "max_score": res.get("max_rerank_score")})
            log(f"    q{i}/{len(reqs)} {r.requirement_id}: retrieval {t['retrieval_s']:.1f}s, rerank {t['rerank_s']:.1f}s")
    finally:
        if old is not None:
            setattr(pipe.reranker.model, attr, old)
    same_top1 = None
    if baseline_top1:
        same_top1 = sum(top1[k] == baseline_top1[k] for k in top1) / len(top1)
    return {"fused_k": fused_k, "max_length": max_length,
            "retrieval_s": statistics.mean(x["retrieval_s"] for x in rows),
            "rerank_s": statistics.mean(x["rerank_s"] for x in rows),
            "recall5": sum(x["hit5"] for x in rows) / len(rows),
            "top1_same": same_top1,
            "mean_max_score": statistics.mean(x["max_score"] for x in rows if x["max_score"] is not None)
            if any(x["max_score"] is not None for x in rows) else None}, top1


def profile(pipe, reqs, variants, log=print):
    # warm-up: the very first call pays lazy initialisation; measure it separately
    t0 = time.perf_counter()
    pipe.retrieve(reqs[0].verification_question)
    first_call = time.perf_counter() - t0
    log(f"first call (warm-up, includes lazy initialisation): {first_call:.1f}s")
    out, baseline_top1 = [], None
    for fk, ml in variants:
        row, top1 = run_variant(pipe, reqs, fk, ml, baseline_top1, log)
        if baseline_top1 is None:
            baseline_top1 = top1
        out.append(row)
    return first_call, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="how many checklist questions to use (max 15). "
                    "On a CPU every question can take tens of seconds per variant: start small.")
    ap.add_argument("--variants", nargs="+", default=["20:512", "10:512", "10:256"],
                    help="fused_k:max_length pairs; the FIRST is the baseline")
    args = ap.parse_args()
    variants = [tuple(int(x) for x in v.split(":")) for v in args.variants]

    from checklist_v2 import load_requirements
    reqs = load_requirements(CHECKLIST)[:args.n]

    print("\n".join(hardware_report()))
    t0 = time.perf_counter()
    from rag_pipeline import RAGPipeline
    pipe = RAGPipeline()
    print(f"RAGPipeline init: {time.perf_counter() - t0:.1f}s | reranker: "
          f"{getattr(getattr(pipe, 'reranker', None), 'model_name', None)}\n")

    first, rows = profile(pipe, reqs, variants, log=lambda m: print(m, flush=True))
    print()
    print(f"{'fused_k':>7s} {'max_len':>7s} {'retrieval_s':>12s} {'rerank_s':>9s} {'recall@5':>9s} {'top1 same':>10s} {'mean max score':>15s}")
    for i, r in enumerate(rows):
        ts = "-" if r["top1_same"] is None else f"{r['top1_same']*100:.0f}%"
        ms = "-" if r["mean_max_score"] is None else f"{r['mean_max_score']:.2f}"
        tag = "  <- baseline" if i == 0 else ""
        print(f"{r['fused_k']:7d} {r['max_length']:7d} {r['retrieval_s']:12.2f} {r['rerank_s']:9.2f} "
              f"{r['recall5']*100:8.0f}% {ts:>10s} {ms:>15s}{tag}")
    print("\nHow to read: pick the fastest variant whose recall@5 and 'top1 same' stay at the baseline level.")
    print("If 'mean max score' moves, recalibrate the abstention threshold (E5) before adopting it.")
    for r in rows[1:]:
        if r["fused_k"] == rows[0]["fused_k"] and r["max_length"] < rows[0]["max_length"] \
                and r["rerank_s"] > 0.9 * rows[0]["rerank_s"]:
            print(f"\n! max_length {r['max_length']} was not clearly faster than {rows[0]['max_length']}: the setting may not "
                  "be taking effect in your sentence-transformers version, or most articles are already short.")
    base = rows[0]
    print(f"\nShare of time (baseline): retrieval {base['retrieval_s']:.2f}s vs rerank {base['rerank_s']:.2f}s per question")


if __name__ == "__main__":
    main()
