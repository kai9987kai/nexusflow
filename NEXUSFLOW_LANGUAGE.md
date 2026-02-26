# NexusFlow (Portfolio-Driven DSL)

`NexusFlow` is a small interpreted language built from recurring patterns across the `kai9987kai` portfolio review (`PORTFOLIO_REVIEW.md`), now extended with:

- native multithreaded simulation execution
- native PyTorch training/export pipeline steps
- native 3D asset loading + scene graph/export (OBJ / glTF / GLB)
- richer GUI widgets + UI templates/themes for dashboard exports
- generated web tool suites + interaction labs (single-file HTML tools)
- built-in fusion reactor and protein folding simulation workflows
- advanced modes: multizone fusion + 3D lattice protein folding
- scientific visualization exports (fusion timeseries / protein trajectory HTML)
- live API tooling: auth presets + HTTP history export + mock/WebSocket labs
- metrics + channels + parallel pipeline substeps
- Windows integration (PowerShell, clipboard, notifications, open/reveal)

- Many small HTML/JS prototype tools
- Many Python simulations / agents / ecosystem experiments
- Several AI/ML and DSL experiments (`NeuroDSL-*`)
- Repeated need for quick exportable dashboards and reproducible pipelines

## What It Solves

Instead of rewriting the same boilerplate for:

- simulation state loops
- agent updates
- quick UI status panels
- pipeline steps (simulate/export/train stub)

you can define them in one file and export:

- JSON snapshots
- single-file HTML dashboards (static preview)

## Language Shape

Supported blocks:

- `project`
- `config`
- `state`
- `metric`
- `channel`
- `model`
- `dataset`
- `agent ... count ... { field ...; on tick { ... } }`
- `ui { template "..."; theme = "..."; panel { text/stat/progress/json/scene3d/button ...; } }`
- `pipeline { step ...; }`

Supported runtime pipeline steps:

- `simulate(ticks)`
- `simulate_mt(ticks, workers[, chunk_size])`
- `auto_simulate(ticks)` (uses `config threads`)
- `train(model, dataset[, epochs])` (metadata stub for reproducible workflow logging)
- `torch_train(model, dataset[, epochs[, lr[, batch_size]]])`
- `torch_export(model, path)`
- `export_json(path)`
- `export_html(path)`
- `load_3d(alias, path[, kind])` (`obj`, `gltf`, `glb`)
- `scene_new(name[, template])`
- `scene_add(scene, asset[, node_name[, transform]])`
- `scene_light(scene[, kind[, intensity[, color]]])`
- `scene_camera(scene, position, target[, fov])`
- `export_scene_json(scene, path)`
- `export_scene_html(scene, path)`
- `web_tool_generate(kind, path[, title])` (`json_lab`, `regex_lab`, `prompt_studio`, `interaction_lab`, `api_builder`)
- `web_tool_suite(dir[, preset])` (`lab`, `dev`, `ai`, `science`)
- `web_live_tool_suite(dir)` (alias for `web_tool_suite(..., "live_api")`)
- additional web tool kinds: `mock_server_lab`, `websocket_lab`
- `web_interaction_tool(path[, title])`
- `http_download(url, path[, timeout_sec])`
- `http_fetch_json(url, path[, timeout_sec])`
- `http_auth_preset(name, headers_map)` / `http_clear_auth_preset(name)`
- `export_http_history_json(path[, limit])`
- `fusion_sim(name[, steps[, config]])`
- `fusion_sim_multizone(name[, steps[, config]])`
- `export_fusion_json(run, path)`
- `export_fusion_html(run, path)`
- `protein_fold_sim(name, sequence[, steps[, config]])`
- `protein_fold_sim_3d(name, sequence[, steps[, config]])`
- `export_protein_json(run, path)`
- `export_protein_html(run, path)`
- `export_csv(path)` (state/metrics/agent summary report)
- `write_text(path, text)` / `append_text(path, text)`
- `save_state(path)` / `load_state(path)`
- `export_events_jsonl(path)`
- `sleep(ms)`
- `parallel(stepCallA(...), stepCallB(...), ...)`
- `repeat(stepCall(...), n)`
- `when(condition, stepCall(...))`
- `try(stepCall(...)[, fallbackStep(...)])`
- `retry(stepCall(...), attempts[, delay_ms])`
- `assert(condition[, message])`
- `run_pipeline(name)` (subpipeline composition)
- `emit_event(label[, payload])`
- `set_state(name, value)` / `inc_state(name[, delta])`
- `bench([label,] stepCall(...))`
- `win_powershell(command[, timeout_sec])`
- `win_cmd(command[, timeout_sec])`
- `win_processes_json(path[, top_n])`
- `win_services_json(path[, top_n])`
- `win_beep([freq[, duration_ms]])`
- `win_clipboard_set(text)`
- `win_notify(title, message[, seconds])`
- `win_open(target)` / `win_reveal(path)`
- `summary()`

Selected builtins (for agent logic / metrics / UI expressions):

- `avg_agent(type, field)`
- `sum_agent(type, field)`
- `count_events([label])`
- `send(channel, value)`, `recv(channel)`, `peek(channel)`, `queue_size(channel)`
- `rand()`, `rand(a,b)`, `randint(a,b)`, `randn([mu,sigma])`
- `now_iso()`, `platform()`, `hostname()`, `cwd()`
- `env(name[, default])`
- `path_exists(path)`, `read_text(path)`, `file_size(path)`
- `slugify(text)`, `html_escape(text)`
- `url_encode(text)`, `url_decode(text)`
- `query_parse(text)`, `query_stringify(map)`
- `web_tool_count()`, `web_tool_info(path_or_filename)`
- `http_history_count()`, `http_history([limit])`, `http_last()`
- `http_auth_preset_count()`, `http_auth_preset_info(name)`
- `http_get_text(url[, timeout_or_preset_or_headers[, timeout]])`
- `http_get_json(url[, timeout_or_preset_or_headers[, timeout]])`
- `http_post_json(url, payload[, timeout_or_preset_or_headers[, timeout_or_headers[, headers]]])`
- `asset3d_count()`, `scene3d_count()`
- `asset3d_info(name)`, `scene3d_info(name)`
- `mesh_vertex_count(asset)`, `mesh_face_count(asset)`, `scene_node_count(scene)`
- `fusion_run_count()`, `fusion_info(name)`, `fusion_metric(run, key)`
- `protein_run_count()`, `protein_info(name)`, `protein_metric(run, key)`
- `protein_length(sequence)`, `protein_hydrophobicity(sequence)`
- `get(container, key[, default])`, `keys(map)`, `values(map)`, `merge(map1[, map2...])`
- `len(x)`, `contains(container, item)`
- `lower(x)`, `upper(x)`, `strip(x[, chars])`, `split(text[, sep])`, `join(sep, items)`, `replace(text, old, new)`
- `to_string(x)`, `to_number(x)`, `json_stringify(value[, pretty])`, `json_parse(text)`
- `sha256(x)`, `uuid4()`
- `drain(channel[, limit])`
- `win_clipboard_get()`
- `win_service_status(name)`, `win_registry_get(path, name[, default])`

## Example

See `examples/eco_ai_web.nxf` for a combined example using:

- `metric` and `channel`
- `simulate_mt`
- `repeat(...)` + `bench(...)`
- `when(...)` + `try(...)` + `retry(...)` + `assert(...)` runtime checks
- subpipelines via `run_pipeline(...)`
- `parallel(...)` pipeline exports
- `torch_train` + `torch_export`
- object literals + dict helpers for structured state
- native 3D asset loading + scene export (`load_3d`, `scene_*`, `export_scene_*`)
- GUI templates/themes + `stat/progress/json/scene3d` widgets in `ui`
- generated web tool suites (`web_tool_suite`) and interaction lab pages
- native scientific simulations (`fusion_sim`, `protein_fold_sim`) with JSON exports + metric accessors
- higher-fidelity variants (`fusion_sim_multizone`, `protein_fold_sim_3d`) for richer experimentation
- scientific visualization HTML exports (`export_fusion_html`, `export_protein_html`)
- live API tooling primitives (HTTP auth presets/history + mock/WebSocket tool generators)
- Windows PowerShell/CMD/process/service/registry integration + CSV/state/event export

Run it:

```powershell
python -m nexusflow run examples/eco_ai_web.nxf
```

Outputs:

- `examples/out/eco_snapshot.json`
- `examples/out/eco_dashboard.html`
- `examples/out/eco_scene.json`
- `examples/out/eco_scene.html`
- `examples/out/web_tools/index.html`
- `examples/out/web_tools_live/index.html`
- `examples/out/fusion_demo.json`
- `examples/out/fusion_multizone.json`
- `examples/out/fusion_multizone.html`
- `examples/out/protein_demo.json`
- `examples/out/protein_demo_3d.json`
- `examples/out/protein_demo_3d.html`
- `examples/out/eco_report.csv`
- `examples/out/brain.pt` (PyTorch model checkpoint bundle)
- `examples/out/brain.pt.json` (checkpoint metadata)
- `examples/out/eco_state.json`
- `examples/out/events.jsonl`
- `examples/out/services_top10.json`

## Why This Fits the Portfolio

The portfolio is strongest at rapid idea exploration across web visuals, simulation, and AI experiments. `NexusFlow` keeps that speed but adds a reusable structure:

- deterministic configs (`seed`)
- multithreaded scale-up (`threads`, `simulate_mt`)
- native PyTorch when model backend is `pytorch`
- portable project definition file
- reusable pipeline steps
- generated inspection artifacts (JSON + HTML + optional `.pt` checkpoint)

That makes it easier to evolve prototypes into maintainable projects without losing the experimental workflow.

## 3D + GUI Example Snippet

```nxf
ui {
  template "ops_console";
  theme = "amber";
  panel "Viewport" {
    stat "Assets" = asset3d_count();
    progress "Build" = 42;
    scene3d "Main Scene" = "showcase";
    json "Scene Info" = scene3d_info("showcase");
  }
}

pipeline build3d {
  step load_3d("crate", "assets/crate.obj");
  step scene_new("showcase", "lab");
  step scene_add("showcase", "crate", "crate_0", {"position": [0,0,0]});
  step scene_light("showcase", "directional", 1.2, "#ffffff");
  step export_scene_json("showcase", "out/showcase.scene.json");
  step export_scene_html("showcase", "out/showcase.scene.html");
}
```

## Web Tools + Science Snippet

```nxf
metric q_demo = fusion_metric("tokamak_demo", "q_estimate");
metric fold_rg = protein_metric("pep", "radius_gyration");

pipeline research {
  step web_tool_suite("out/webtools", "lab");
  step web_interaction_tool("out/webtools/interaction_plus.html");
  step fusion_sim("tokamak_demo", 80, {"heating_mw": 145, "magnetic_field_t": 5.8});
  step protein_fold_sim("pep", "MKWVTFISLLFLFSSAYS", 140, {"temperature": 0.85});
  step export_fusion_json("tokamak_demo", "out/fusion_demo.json");
  step export_protein_json("pep", "out/protein_demo.json");
}
```

## Advanced Science + Live API Snippet

```nxf
metric http_ops = http_history_count();
metric q_mz = fusion_metric("tokamak_mz", "q_estimate");
metric rg3 = protein_metric("pep3d", "radius_gyration");

pipeline advanced {
  step web_live_tool_suite("out/live_tools");
  step http_auth_preset("lab", {"Authorization": "Bearer demo"});
  step export_http_history_json("out/http_history.json");
  step fusion_sim_multizone("tokamak_mz", 120, {"zones": 4, "heating_mw": 170});
  step export_fusion_html("tokamak_mz", "out/tokamak_mz.html");
  step protein_fold_sim_3d("pep3d", "MKWVTFISLLFLFSSAYS", 180, {"temperature": 0.8});
  step export_protein_html("pep3d", "out/pep3d.html");
}
```
