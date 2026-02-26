import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = ROOT / "_kai_audit" / "repo_audit.json"
OUT_PATH = ROOT / "PORTFOLIO_REVIEW.md"


def load_reviews():
    data = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    return data["reviews"]


def classify(r):
    f = r["features"]
    name = (r["name"] or "").lower()
    if f.get("dsl"):
        return "DSL / Compilers"
    if f.get("unity") or "unity" in name:
        return "Game / Unity"
    if f.get("game") and f.get("ai_ml"):
        return "AI Simulations / Games"
    if f.get("game"):
        return "Games / Simulations"
    if f.get("ai_ml"):
        return "AI / ML Tools"
    if f.get("webapp") and (f.get("javascript") or f.get("html")):
        return "Web Apps / Generators"
    if f.get("desktop_gui"):
        return "Desktop Utilities / GUI"
    if f.get("python"):
        return "Python Utilities / Scripts"
    return "Misc / Experiments"


def top_features(r):
    order = [
        "webapp",
        "html",
        "python",
        "ai_ml",
        "game",
        "desktop_gui",
        "javascript",
        "typescript",
        "unity",
        "dsl",
        "tests",
        "ci",
    ]
    return [k for k in order if r["features"].get(k)]


def fmt_list(items):
    return ", ".join(items) if items else "None"


def main():
    reviews = sorted(load_reviews(), key=lambda r: r["name"].lower())
    lang_counts = Counter((r.get("primary_language") or "Unknown") for r in reviews)
    cat_counts = Counter(classify(r) for r in reviews)
    feat_counts = Counter()
    for r in reviews:
        for k, v in r["features"].items():
            if v:
                feat_counts[k] += 1

    med_files = sorted(r["file_count"] for r in reviews)[len(reviews) // 2]
    avg_files = sum(r["file_count"] for r in reviews) / len(reviews)

    lines = []
    lines.append("# Portfolio Review for `kai9987kai`")
    lines.append("")
    lines.append("Generated from an automated shallow audit of every public repository (169 total).")
    lines.append("")
    lines.append("## Cross-Repo Findings")
    lines.append("")
    lines.append(f"- Total repos reviewed: **{len(reviews)}**")
    lines.append(f"- Median repo file count: **{med_files}** (many small prototypes)")
    lines.append(f"- Average repo file count: **{avg_files:.1f}** (a few large projects raise the mean)")
    lines.append("- Primary language mix (top 10):")
    for lang, count in lang_counts.most_common(10):
        lines.append(f"  - {lang}: {count}")
    lines.append("- Portfolio category mix:")
    for cat, count in cat_counts.most_common():
        lines.append(f"  - {cat}: {count}")
    lines.append("- Repeated strengths:")
    lines.append("  - Strong prototype throughput across HTML, Python, and simulation/game concepts")
    lines.append("  - Frequent experimentation with AI/agent/evolution ideas")
    lines.append("  - Increasing sophistication in larger projects (docs, security files, packaging)")
    lines.append("- Repeated gaps:")
    lines.append("  - Testing and CI are uncommon across most repos")
    lines.append("  - Many repos are single-purpose prototypes without standardized packaging")
    lines.append("  - Similar patterns repeat (UI wiring, config, simulation loops, export paths)")
    lines.append("- Language opportunity:")
    lines.append("  - A single DSL that targets **rapid web + simulation + AI experiment prototyping** would reduce boilerplate and make projects easier to evolve into maintainable tools.")
    lines.append("")
    lines.append("## Per-Repository Review (All Repos)")
    lines.append("")

    for idx, r in enumerate(reviews, start=1):
        category = classify(r)
        feats = top_features(r)
        risk = "; ".join(r.get("risks") or []) or "No major automated flags"
        opp = "; ".join(r.get("opportunities") or []) or "No immediate automation suggestion"
        strengths = "; ".join(r.get("strengths") or []) or "No notable automated strengths detected"
        snippet = r.get("readme_snippet") or (r.get("description") or "")
        if snippet:
            snippet = snippet.replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."

        lines.append(f"### {idx}. `{r['name']}`")
        lines.append("")
        lines.append(f"- URL: {r.get('html_url')}")
        lines.append(f"- Category: {category}")
        lines.append(f"- Primary language: {r.get('primary_language') or 'Unknown'}")
        lines.append(f"- Size/File count: {r.get('size_kb', 0)} KB / {r.get('file_count', 0)} files")
        lines.append(f"- Features: {fmt_list(feats)}")
        if snippet:
            lines.append(f"- Focus: {snippet}")
        lines.append(f"- Strengths: {strengths}")
        lines.append(f"- Risks: {risk}")
        lines.append(f"- Opportunities: {opp}")
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(reviews)} repo entries.")


if __name__ == "__main__":
    main()
