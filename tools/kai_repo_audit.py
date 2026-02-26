import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "repos_metadata.json"
AUDIT_DIR = ROOT / "_kai_audit"
CLONE_ROOT = AUDIT_DIR / "repos"
OUT_PATH = AUDIT_DIR / "repo_audit.json"
FAIL_PATH = AUDIT_DIR / "clone_failures.json"


README_CANDIDATES = (
    "README.md",
    "Readme.md",
    "readme.md",
    "README",
    "readme",
)


def run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def git(repo_dir: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo_dir), *args], timeout=timeout)


def sanitize_repo_dir(repo_name: str, idx: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_name).strip("-")
    slug = slug[:48] if slug else "repo"
    return f"{idx:03d}_{slug}"


def parse_metadata() -> List[Dict[str, Any]]:
    data = json.loads(META_PATH.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "value" in data:
        data = data["value"]
    assert isinstance(data, list), "repos_metadata.json must contain a list"
    return sorted(data, key=lambda x: (x.get("name") or "").lower())


def ensure_clone(repo: Dict[str, Any], repo_dir: Path) -> Optional[str]:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        return None
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = repo.get("clone_url")
    if not clone_url:
        full_name = repo.get("full_name")
        if not full_name:
            return "missing clone_url and full_name"
        clone_url = f"https://github.com/{full_name}.git"
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-checkout",
        clone_url,
        str(repo_dir),
    ]
    proc = run(cmd, timeout=900)
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip()
    return None


def get_head(repo_dir: Path) -> Optional[str]:
    proc = git(repo_dir, "rev-parse", "HEAD")
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def list_tree(repo_dir: Path) -> List[str]:
    proc = git(repo_dir, "ls-tree", "-r", "--name-only", "HEAD", timeout=900)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def list_top_level(repo_dir: Path) -> List[str]:
    proc = git(repo_dir, "ls-tree", "--name-only", "HEAD")
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def read_blob(repo_dir: Path, path: str, max_bytes: int = 20000) -> Optional[str]:
    proc = git(repo_dir, "show", f"HEAD:{path}", timeout=300)
    if proc.returncode != 0:
        return None
    text = proc.stdout
    if len(text.encode("utf-8", errors="ignore")) > max_bytes:
        text = text[:max_bytes]
    return text


def find_readme_path(tree_files: List[str]) -> Optional[str]:
    top_level = [p for p in tree_files if "/" not in p]
    for cand in README_CANDIDATES:
        if cand in top_level:
            return cand
    for p in top_level:
        if p.lower().startswith("readme"):
            return p
    for p in tree_files:
        if os.path.basename(p).lower().startswith("readme"):
            return p
    return None


def extension_counts(tree_files: List[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for path in tree_files:
        base = os.path.basename(path)
        if "." not in base:
            counts["<noext>"] += 1
            continue
        ext = "." + base.rsplit(".", 1)[-1].lower()
        counts[ext] += 1
    return dict(counts.most_common(20))


def detect_features(tree_files: List[str], readme: str) -> Dict[str, bool]:
    lower_tree = {p.lower() for p in tree_files}
    lower_readme = (readme or "").lower()

    def has_file(name: str) -> bool:
        n = name.lower()
        return any(p.endswith(n) for p in lower_tree)

    def has_dir(name: str) -> bool:
        n = name.lower().rstrip("/") + "/"
        return any(p.startswith(n) for p in lower_tree)

    return {
        "python": any(p.endswith(".py") for p in lower_tree),
        "javascript": any(p.endswith(".js") for p in lower_tree),
        "typescript": any(p.endswith(".ts") or p.endswith(".tsx") for p in lower_tree),
        "html": any(p.endswith(".html") for p in lower_tree),
        "css": any(p.endswith(".css") for p in lower_tree),
        "java": any(p.endswith(".java") for p in lower_tree),
        "csharp": any(p.endswith(".cs") for p in lower_tree),
        "cpp": any(p.endswith(".cpp") or p.endswith(".cc") or p.endswith(".cxx") for p in lower_tree),
        "unity": has_dir("Assets") or has_file("ProjectSettings/ProjectVersion.txt"),
        "godot": has_file("project.godot"),
        "pygame": "pygame" in lower_readme or has_file("requirements.txt"),
        "threejs": "three.js" in lower_readme or "threejs" in lower_readme,
        "ai_ml": any(
            kw in lower_readme
            for kw in [
                "neural",
                "pytorch",
                "tensorflow",
                "model",
                "llm",
                "agent",
                "evolution",
                "genetic",
                "machine learning",
            ]
        ),
        "game": any(
            kw in lower_readme
            for kw in ["game", "simulation", "unity", "godot", "pygame", "player", "enemy"]
        ),
        "webapp": has_file("package.json") or any(p.endswith(".html") for p in lower_tree),
        "desktop_gui": any(kw in lower_readme for kw in ["tkinter", "pyqt", "qt", "gui"]) or has_file("requirements.txt"),
        "tests": any("/test" in p or "/tests" in p or p.startswith("test") or p.startswith("tests") for p in lower_tree),
        "ci": any(p.startswith(".github/workflows/") for p in lower_tree),
        "docker": has_file("Dockerfile") or has_file("docker-compose.yml") or has_file("docker-compose.yaml"),
        "notebook": any(p.endswith(".ipynb") for p in lower_tree),
        "dsl": "dsl" in lower_readme or any("dsl" in p for p in lower_tree),
    }


def readme_summary(readme_text: str) -> Dict[str, Any]:
    if not readme_text:
        return {"title": None, "snippet": None}
    lines = [ln.strip() for ln in readme_text.splitlines()]
    non_empty = [ln for ln in lines if ln]
    title = None
    for ln in non_empty[:20]:
        if ln.startswith("#"):
            title = ln.lstrip("#").strip()
            break
    snippet_parts: List[str] = []
    for ln in non_empty:
        if ln.startswith("#"):
            continue
        if ln.startswith("![") or ln.startswith("[!"):
            continue
        if re.fullmatch(r"[-=*]{3,}", ln):
            continue
        snippet_parts.append(ln)
        if len(" ".join(snippet_parts)) > 280:
            break
    snippet = " ".join(snippet_parts)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 300:
        snippet = snippet[:297] + "..."
    return {"title": title, "snippet": snippet or None}


def make_review(repo: Dict[str, Any], tree_files: List[str], top_level: List[str], readme_text: str) -> Dict[str, Any]:
    ext_counts = extension_counts(tree_files)
    features = detect_features(tree_files, readme_text or "")
    readme_info = readme_summary(readme_text or "")
    manifests = [f for f in top_level if f in {"package.json", "pyproject.toml", "requirements.txt", "setup.py", "Cargo.toml", "pom.xml"}]

    strengths: List[str] = []
    risks: List[str] = []
    opportunities: List[str] = []

    if readme_info["snippet"]:
        strengths.append("README/doc present")
    else:
        risks.append("missing or minimal README")
    if features["tests"]:
        strengths.append("tests present")
    else:
        risks.append("no obvious tests")
    if features["ci"]:
        strengths.append("CI workflow configured")
    else:
        opportunities.append("add GitHub Actions for smoke tests/build")
    if features["dsl"]:
        strengths.append("experiments with DSL/abstraction")
    if features["ai_ml"]:
        strengths.append("AI/simulation focus")
        opportunities.append("reproducible experiment config and deterministic seeds")
    if features["game"] and not features["tests"]:
        opportunities.append("headless simulation test harness for gameplay logic")
    if features["webapp"] and not features["docker"]:
        opportunities.append("deploy preset + environment template")
    if repo.get("size", 0) <= 2:
        risks.append("very small repo size (possibly placeholder or early stub)")
    if not manifests and features["python"]:
        opportunities.append("standardize packaging with pyproject.toml")
    if features["webapp"] and not any(p in {"package.json"} for p in manifests) and features["javascript"]:
        opportunities.append("package scripts and lockfile for reproducible frontend builds")

    # Keep concise and deterministic.
    strengths = strengths[:3]
    risks = risks[:3]
    opportunities = opportunities[:3]

    return {
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "clone_url": repo.get("clone_url") or (f"https://github.com/{repo.get('full_name')}.git" if repo.get("full_name") else None),
        "html_url": repo.get("html_url") or (f"https://github.com/{repo.get('full_name')}" if repo.get("full_name") else None),
        "description": repo.get("description"),
        "default_branch": repo.get("default_branch"),
        "primary_language": repo.get("language"),
        "size_kb": repo.get("size"),
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "top_level": top_level[:40],
        "file_count": len(tree_files),
        "top_extensions": ext_counts,
        "manifests": manifests,
        "readme_title": readme_info["title"],
        "readme_snippet": readme_info["snippet"],
        "features": features,
        "strengths": strengths,
        "risks": risks,
        "opportunities": opportunities,
    }


def main() -> int:
    if not META_PATH.exists():
        print(f"Missing metadata file: {META_PATH}", file=sys.stderr)
        return 1
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    CLONE_ROOT.mkdir(parents=True, exist_ok=True)

    repos = parse_metadata()
    reviews: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    start = time.time()
    for idx, repo in enumerate(repos, start=1):
        repo_name = repo["name"]
        repo_dir = CLONE_ROOT / sanitize_repo_dir(repo_name, idx)
        print(f"[{idx}/{len(repos)}] {repo_name}", flush=True)

        err = ensure_clone(repo, repo_dir)
        if err:
            failures.append({"name": repo_name, "error": err})
            continue

        head = get_head(repo_dir)
        if not head:
            failures.append({"name": repo_name, "error": "unable to resolve HEAD"})
            continue

        tree_files = list_tree(repo_dir)
        top_level = list_top_level(repo_dir)
        readme_path = find_readme_path(tree_files)
        readme_text = read_blob(repo_dir, readme_path) if readme_path else None
        review = make_review(repo, tree_files, top_level, readme_text or "")
        review["head"] = head
        review["local_dir"] = str(repo_dir.relative_to(ROOT))
        review["readme_path"] = readme_path
        reviews.append(review)

    duration = round(time.time() - start, 1)
    payload = {
        "generated_at_epoch": int(time.time()),
        "repo_count": len(repos),
        "reviewed_count": len(reviews),
        "failure_count": len(failures),
        "duration_sec": duration,
        "reviews": reviews,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    FAIL_PATH.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved {OUT_PATH} ({len(reviews)} reviews, {len(failures)} failures) in {duration}s")
    return 0 if not failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
