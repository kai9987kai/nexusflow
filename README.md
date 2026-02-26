# NexusFlow

NexusFlow is a Python-based DSL and runtime for building rapid prototypes that combine:

- simulations and agents
- AI/ML workflows (PyTorch and native Python models)
- web dashboards and generated web tools
- Windows automation/integration steps
- 3D asset loading and scene metadata exports
- experimental science workflows (fusion/protein toy simulations)
- polyglot modules (Python, JavaScript, C++, PowerShell)

It was designed as a "prototype orchestration language" for small-to-medium experimental projects.

## What Is In This Repo

- `nexusflow/` - parser, runtime, CLI
- `examples/` - example `.nxf` programs and generated outputs
- `tests/` - unit/integration tests
- `NEXUSFLOW_LANGUAGE.md` - language reference and examples
- `PORTFOLIO_REVIEW.md` - source portfolio review that informed the DSL design

## Quick Start

Requirements:

- Python 3.10+ (recommended)
- Optional: `torch` for PyTorch-backed training/export
- Optional: `node` for JS modules, `g++/clang++/cl` for C++ modules
- Optional: Windows + PowerShell for Windows-native integration steps

Run the included example:

```powershell
python -m nexusflow lint examples/eco_ai_web.nxf
python -m nexusflow run examples/eco_ai_web.nxf
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

## CLI

NexusFlow ships a simple CLI:

```powershell
python -m nexusflow lint <file.nxf>
python -m nexusflow dump <file.nxf>
python -m nexusflow run <file.nxf> [--pipeline NAME] [--out-dir DIR] [--export-json PATH] [--export-html PATH]
```

- `lint` parses and validates the DSL file
- `dump` prints the parsed JSON IR
- `run` executes the program and pipelines, then prints a JSON snapshot

## Minimal Example

```nxf
project "HelloFlow" {
  state count = 0;

  agent bot count 3 {
    field energy = 1;
    on tick {
      count += 1;
      emit "tick_seen";
    }
  }

  pipeline main {
    step simulate_mt(5, 2);
    step export_json("out/hello.json");
    step summary();
  }
}
```

Run it:

```powershell
python -m nexusflow run hello.nxf --pipeline main
```

## Built-In IDE / IDLE (Web Tools)

NexusFlow can generate a browser-based IDE and IDLE for `.nxf` authoring and experimentation.

Pipeline steps:

- `nexus_ide(path[, title])`
- `nexus_idle(path[, title])`
- `nexus_dev_suite(dir)` (generates IDE + IDLE + supporting tools)

Example:

```nxf
pipeline tools {
  step nexus_ide("out/nexus_ide.html");
  step nexus_idle("out/nexus_idle.html");
  step nexus_dev_suite("out/dev_suite");
}
```

You can also generate them through the generic web tool APIs:

- `web_tool_generate("nexus_ide", "...")`
- `web_tool_generate("nexus_idle", "...")`
- `web_tool_suite("...", "nexus_dev")`

## Feature Overview

## 1) Core DSL + Runtime

- project/config/state/metric/channel declarations
- agent-based simulation (`on tick`)
- event emission and event counting
- multithreaded simulation (`simulate_mt`)
- pipelines with orchestration (`parallel`, `repeat`, `bench`, `try`, `retry`, `when`, `assert`)

## 2) AI / ML

- Native PyTorch training/export (`torch_train`, `torch_export`)
- Python-native model training (`backend = "python"`) including:
  - linear/ridge regression
  - logistic regression
  - perceptron
  - k-means
- model/dataset blocks and training pipeline integration

## 3) Web + GUI + Generated Tools

- `ui` blocks with panels, text/stat/progress/json/scene3d widgets
- HTML dashboard export (`export_html`)
- generated web tools and suites (labs, mock API tools, websocket tools, prompt tools)
- built-in web IDE/IDLE generators for NexusFlow development

## 4) 3D Tools

- 3D asset metadata loading for `OBJ`, `glTF`, `GLB`
- scene graph construction (`scene_new`, `scene_add`, `scene_light`, `scene_camera`)
- scene JSON/HTML exports

Note: current 3D support focuses on asset parsing/metadata/scene exports, not a full native renderer.

## 5) Windows Integrations

- PowerShell and CMD execution (`win_powershell`, `win_cmd`)
- clipboard, notifications, open/reveal, beeps
- process/service snapshots and queries
- registry reads (`win_registry_get`)

These features are intended for Windows hosts and may be unsupported or no-op on other platforms.

## 6) HTTP + Mock Servers

- HTTP requests/downloads (`http_get_json`, `http_post_json`, etc.)
- HTTP auth preset registry
- request history export
- local mock HTTP server start/stop steps

## 7) Experimental Science Modules (Prototype-Level)

- Fusion reactor toy simulations (single-zone + multizone + control/sweep helpers)
- Protein folding toy simulations (2D + 3D lattice-style variants)
- JSON/HTML exports for visualization

Important: these are coarse experimental/prototyping models, not high-fidelity scientific solvers.

## 8) Polyglot Modules

- register and run modules written in:
  - Python
  - JavaScript (Node.js)
  - C++ (with local compiler)
  - Rust (`rustc` / `cargo` when available)
  - C# / .NET (single-file or project)
  - PowerShell
- runtime captures module execution history and exposes it in snapshots/builtins

## 9) Systems Ops + Hardware Tools (New)

- managed process tools with lightweight virtualization profiles:
  - `proc_profile`, `proc_exec`, `proc_profile_run`
  - `proc_spawn`, `proc_wait`, `proc_kill`
  - `proc_history_json`, `proc_managed_json`
- Wi-Fi management helpers (Windows `netsh` wrappers):
  - `wifi_interfaces_json`, `wifi_profiles_json`, `wifi_scan_json`
  - `wifi_connect`, `wifi_disconnect`
  - builtins: `wifi_supported`, `wifi_interfaces`, `wifi_profiles`, `wifi_last_scan`
- accelerator / NPU probing:
  - `npu_probe_json(...)`
  - builtins: `accelerator_info`, `npu_info`, `npu_available`
- ISO packaging workflows (already includes build/list/extract) plus source manifest export:
  - `iso_manifest_json(source_dir, out_path)`
- built-in EXE builder / packager:
  - `exe_build(source_or_module, out_path[, name_or_cfg])`
  - supports Python scripts (PyInstaller/Nuitka when available), `.exe` copy packaging, and registered `rust/cpp/csharp` modules
  - builtins: `exe_tool_info`, `exe_count`, `exe_info`, `exe_last`

## 10) Media + Data + Graph Tools (New)

- built-in procedural photo creation (dependency-free SVG/PPM):
  - `photo_generate(path, width, height[, prompt_or_cfg])`
- data chart SVG generation:
  - `data_chart_svg(path, values[, cfg])`
- graph/network tooling:
  - `graph_create`, `graph_from_csv`, `graph_export_svg`, `graph_metrics_json`
  - builtins: `graph_stats`, `graph_shortest_path`, `graph_components`, `graph_degrees`
- native file conversion tools:
  - `file_convert(src, dst[, mode])`
  - modes include JSON/CSV/TSV/JSONL plus base64/hex text conversions

## 11) Math/Science + Portfolio Intelligence (New)

- math/science builtins for metrics and pipelines:
  - `mean`, `median`, `variance`, `stddev`
  - `linspace`, `dot`, `norm`, `distance`
  - `differentiate`, `integrate_trapz`, `poly_eval`, `linear_fit`
  - `stats_summary`
- local GitHub portfolio helpers (great for `repos_metadata.json` style exports):
  - `github_local_summary`, `github_repo_find`
  - `github_portfolio_report(meta_path, out_path)`
- idea generation / tool invention helper:
  - `idea_forge_json(out_path[, theme])`
  - produces feature/tool blueprints using host capabilities + local repo metadata

## Python API

Basic usage from Python:

```python
from pathlib import Path
from nexusflow import Executor, parse_file

project = parse_file("examples/eco_ai_web.nxf")
executor = Executor(project, source_path=Path("examples/eco_ai_web.nxf"))
executor.run_pipeline("preview")
snapshot = executor.snapshot()
print(snapshot["tick"])
```

## Example Outputs

Running `examples/eco_ai_web.nxf` generates a large set of artifacts under `examples/out/`, including:

- dashboard HTML and snapshot JSON
- scene JSON/HTML
- CSV reports and state exports
- generated web tool suites
- fusion/protein JSON and HTML visualizations
- (optional) PyTorch model export

## Documentation

- Language reference: `NEXUSFLOW_LANGUAGE.md`
- Example program: `examples/eco_ai_web.nxf`
- Tests: `tests/test_nexusflow.py`

## Current Status

NexusFlow is a fast-moving experimental project. The DSL and runtime are actively evolving, and some features are intentionally prototype-oriented (especially science modules and generated web labs).

If you want a stable subset for production use, treat the parser/runtime/CLI interfaces as the most reliable layer and pin your own tested `.nxf` patterns.
