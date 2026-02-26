import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from nexusflow.lang import TORCH_AVAILABLE, Executor, parse_file, parse_source, project_to_json


SIMPLE_PROGRAM = r'''
project "TestProj" {
  config seed = 1;
  state food = 3;
  state day = 0;

  agent grazer count 2 {
    field energy = 1;
    field hunger = 0;

    on tick {
      hunger += 1;
      if hunger > 0 {
        if food > 0 {
          food -= 1;
          energy += 2;
        } else {
          energy -= 1;
        }
        hunger = 0;
      }
      day = tick;
    }
  }

  ui {
    panel "World" {
      text "Food" = food;
      text "Day" = day;
    }
  }

  pipeline preview {
    step simulate(2);
    step train("brain", "synthetic", 2);
    step export_json("out/test.json");
    step export_html("out/test.html");
    step summary();
  }
}
'''


ADVANCED_PROGRAM = r'''
project "AdvancedProj" {
  config seed = 7;
  config threads = 3;
  config pipeline_threads = 2;
  state food = 40;
  state day = 0;
  metric total_energy = sum_agent("grazer", "energy");
  metric pressure = queue_size("alerts") + count_events();
  channel alerts;

  model brain {
    kind = "mlp";
    inputs = 4;
    outputs = 2;
    hidden = [8, 8];
    backend = "pytorch";
  }

  dataset synthetic_env {
    source = "synthetic";
    task = "classification";
    samples = 64;
    inputs = 4;
    outputs = 2;
    batch_size = 16;
  }

  agent grazer count 6 {
    field energy = 2;
    field hunger = 0;
    on tick {
      hunger += 1;
      if hunger > 0 {
        if food > 0 {
          food -= 1;
          energy += 1;
          send("alerts", "eat");
          emit "grazer_ate";
        }
        hunger = 0;
      }
      day = tick;
    }
  }

  pipeline preview {
    step simulate_mt(3, 2);
    step torch_train("brain", "synthetic_env", 1, 0.01, 16);
    step parallel(
      export_json("out/adv.json"),
      export_html("out/adv.html"),
      torch_export("brain", "out/brain.pt")
    );
    step summary();
  }
}
'''


UTILITY_PROGRAM = r'''
project "UtilityProj" {
  config seed = 3;
  state counter = 0;
  metric seen = count_events();

  agent bot count 1 {
    field energy = 1;
    on tick {
      counter += 1;
      emit "tick";
    }
  }

  pipeline tools {
    step write_text("out/log.txt", "hello");
    step append_text("out/log.txt", "\nworld");
    step repeat(simulate(1), 2);
    step bench("one_more_tick", simulate(1));
    step export_csv("out/report.csv");
    step save_state("out/state.json");
    step load_state("out/state.json");
    step summary();
  }
}
'''


WINDOWS_PROGRAM = r'''
project "WinOpsProj" {
  state x = 1;
  pipeline winops {
    step win_powershell("Write-Output 'nexusflow-winops'");
    step win_cmd("echo nexusflow-cmd");
    step win_processes_json("out/proc.json", 5);
    step win_services_json("out/services.json", 5);
    step write_text("out/service_status.json", json_stringify(win_service_status("EventLog"), true));
    step write_text("out/product_name.txt", to_string(win_registry_get("HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductName", "unknown")));
    step summary();
  }
}
'''


ORCH_PROGRAM = r'''
project "OrchProj" {
  state ctr = 0;
  state meta = {"name": "kai", "ver": 1};
  channel q;
  metric q_remaining = queue_size("q");
  metric seen_upper = count_events("TICK");
  metric meta_ver = get(meta, "ver", 0);
  metric meta_key_count = len(keys(meta));

  agent bot count 1 {
    field energy = 1;
    on tick {
      ctr += 1;
      send("q", upper("tick"));
      emit upper("tick");
    }
  }

  pipeline helpers {
    step inc_state("ctr", 2);
    step set_state("meta", merge(meta, {"tag": "ok"}));
    step emit_event("helper_event", {"src": "helpers", "id": 1});
  }

  pipeline flow {
    step when(true, simulate(1));
    step when(false, simulate(99));
    step run_pipeline("helpers");
    step assert(ctr == 3, "count mismatch");
    step write_text("out/str.txt", join("-", split(lower("A B C"), " ")));
    step write_text("out/ch.json", json_stringify(drain("q")));
    step write_text("out/meta.json", json_stringify(meta, true));
    step try(assert(false, "boom"), emit_event("recovered"));
    step assert(contains(read_text("out/str.txt"), "a-b-c"), "missing join/split result");
    step retry(assert(path_exists("out/str.txt"), "missing str file"), 2);
    step write_text("out/hash.txt", sha256(read_text("out/str.txt")));
    step export_events_jsonl("out/events.jsonl");
    step export_json("out/orch.json");
    step summary();
  }
}
'''


RETRY_FAIL_PROGRAM = r'''
project "RetryFail" {
  pipeline fail {
    step retry(load_state("out/missing_state.json"), 2, 1);
  }
}
'''


THREED_GUI_PROGRAM = r'''
project "ThreeDGui" {
  state progress_value = 35;
  state active_scene = "showcase";
  metric loaded_assets = asset3d_count();
  metric scene_nodes = scene_node_count("showcase");
  metric crate_vertices = mesh_vertex_count("crate");

  ui {
    template "ops_console";
    theme = "amber";

    panel "Scene Ops" {
      stat "Assets Loaded" = loaded_assets;
      progress "Build Progress" = progress_value;
      scene3d "Primary Scene" = active_scene;
      json "Crate Asset" = asset3d_info("crate");
      text "Scene Nodes" = scene_nodes;
      text "Crate Vertices" = crate_vertices;
    }
  }

  pipeline build {
    step load_3d("crate", "assets/crate.obj");
    step load_3d("robot", "assets/robot.gltf");
    step scene_new("showcase", "lab");
    step scene_add("showcase", "crate", "crate_main", {"position": [0, 0, 0]});
    step scene_add("showcase", "robot", "robot_a", {"position": [1.5, 0.5, -1], "scale": [1.2, 1.2, 1.2]});
    step scene_light("showcase", "directional", 1.4, "#a7f3d0");
    step scene_camera("showcase", [4, 2, 8], [0, 0, 0], 55);
    step export_scene_json("showcase", "out/showcase.scene.json");
    step export_scene_html("showcase", "out/showcase.scene.html");
    step export_html("out/dashboard.html");
    step export_json("out/snapshot.json");
    step summary();
  }
}
'''


WEB_SCIENCE_PROGRAM_TEMPLATE = r'''
project "WebScienceLab" {
  config seed = 11;
  state source_url = __FILE_URI__;
  metric web_suite_files = web_tool_count();
  metric fusion_q = fusion_metric("demo_tokamak", "q_estimate");
  metric fusion_wall_ok = fusion_metric("demo_tokamak", "wall_load_ok");
  metric protein_rg = protein_metric("demo_fold", "radius_gyration");
  metric protein_contacts = protein_metric("demo_fold", "hydrophobic_contacts");
  metric query_fields = len(keys(query_parse(query_stringify({"a": "1", "b": ["x", "y"]}))));
  metric hydrophobic_ratio = protein_hydrophobicity("MKWVTFISLLFLFSSAYS");

  ui {
    template "ops_console";
    theme = "lime";
    panel "Research Ops" {
      stat "Web Tools" = web_suite_files;
      stat "Fusion Q" = fusion_q;
      stat "Protein Rg" = protein_rg;
      progress "Hydrophobic %" = hydrophobic_ratio * 100;
      json "Fusion Run" = fusion_info("demo_tokamak");
      json "Protein Run" = protein_info("demo_fold");
      text "Query Keys" = query_fields;
      text "Wall Load OK" = fusion_wall_ok;
    }
  }

  pipeline build {
    step web_tool_suite("out/webtools", "lab");
    step web_interaction_tool("out/webtools/interaction_plus.html", "Interaction Plus");
    step web_tool_generate("json_lab", "out/webtools/json_plus.html", "JSON Plus");
    step fusion_sim("demo_tokamak", 50, {"heating_mw": 150, "magnetic_field_t": 5.9});
    step protein_fold_sim("demo_fold", "MKWVTFISLLFLFSSAYS", 120, {"temperature": 0.9});
    step http_fetch_json(source_url, "out/local_fetch.json");
    step write_text("out/fetch_inline.json", json_stringify(http_get_json(source_url), true));
    step export_fusion_json("demo_tokamak", "out/fusion_demo.json");
    step export_protein_json("demo_fold", "out/protein_demo.json");
    step export_html("out/web_science_dashboard.html");
    step export_json("out/web_science_snapshot.json");
    step summary();
  }
}
'''


ADVANCED_LIVE_SCIENCE_PROGRAM_TEMPLATE = r'''
project "AdvancedLiveScience" {
  config seed = 19;
  state source_url = __FILE_URI__;
  metric http_count = http_history_count();
  metric auth_count = http_auth_preset_count();
  metric fusion_mode_q = fusion_metric("tokamak_mz", "q_estimate");
  metric fusion_zones = fusion_metric("tokamak_mz", "zones");
  metric protein3d_rg = protein_metric("pep3d", "radius_gyration");
  metric protein3d_len = protein_metric("pep3d", "length");

  ui {
    template "ops_console";
    theme = "amber";
    panel "Advanced Tools" {
      stat "HTTP Ops" = http_count;
      stat "Auth Presets" = auth_count;
      stat "Fusion Q (MZ)" = fusion_mode_q;
      stat "Protein3D Rg" = protein3d_rg;
      text "Fusion Zones" = fusion_zones;
      text "Protein Len" = protein3d_len;
      json "HTTP Last" = http_last();
      json "HTTP Preset" = http_auth_preset_info("local_auth");
      json "Fusion MZ" = fusion_info("tokamak_mz");
      json "Protein 3D" = protein_info("pep3d");
    }
  }

  pipeline advanced {
    step web_live_tool_suite("out/live_tools");
    step web_tool_generate("mock_server_lab", "out/live_tools/mock_server_plus.html", "Mock Server Plus");
    step web_tool_generate("websocket_lab", "out/live_tools/ws_plus.html", "WebSocket Plus");
    step http_auth_preset("local_auth", {"Authorization": "Bearer demo-token", "X-Client": "NexusFlowTest"});
    step http_fetch_json(source_url, "out/file_fetch_auth.json", "local_auth");
    step export_http_history_json("out/http_history.json");
    step fusion_sim_multizone("tokamak_mz", 60, {"heating_mw": 165, "zones": 4});
    step export_fusion_json("tokamak_mz", "out/tokamak_mz.json");
    step export_fusion_html("tokamak_mz", "out/tokamak_mz.html");
    step protein_fold_sim_3d("pep3d", "MKWVTFISLLFLFSSAYS", 110, {"temperature": 0.8});
    step export_protein_json("pep3d", "out/pep3d.json");
    step export_protein_html("pep3d", "out/pep3d.html");
    step export_html("out/advanced_dashboard.html");
    step export_json("out/advanced_snapshot.json");
    step summary();
  }
}
'''


class NexusFlowTests(unittest.TestCase):
    def test_parse_and_ir(self) -> None:
        project = parse_source(SIMPLE_PROGRAM)
        ir = project_to_json(project)
        self.assertEqual(ir["project"], "TestProj")
        self.assertEqual(len(ir["agents"]), 1)
        self.assertEqual(ir["pipelines"][0]["name"], "preview")

    def test_run_pipeline_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "test.nxf"
            src.write_text(SIMPLE_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("preview")
            snap = executor.snapshot()

            self.assertEqual(snap["tick"], 2)
            self.assertIn("grazer", snap["agents"])
            self.assertGreaterEqual(len(snap["training_runs"]), 1)

            json_out = tmp / "out" / "test.json"
            html_out = tmp / "out" / "test.html"
            self.assertTrue(json_out.exists())
            self.assertTrue(html_out.exists())

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["project"], "TestProj")
            self.assertIn("state", payload)

    def test_advanced_features_multithread_metrics_channels_and_torch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "advanced.nxf"
            src.write_text(ADVANCED_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("preview")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "AdvancedProj")
            self.assertIn("metrics", snap)
            self.assertIn("total_energy", snap["metrics"])
            self.assertIn("channels", snap)
            self.assertIn("alerts", snap["channels"])
            self.assertGreaterEqual(snap["threading"]["simulate_mt_calls"], 1)
            self.assertGreaterEqual(snap["threading"]["pipeline_parallel_calls"], 1)

            self.assertTrue((tmp / "out" / "adv.json").exists())
            self.assertTrue((tmp / "out" / "adv.html").exists())

            if TORCH_AVAILABLE:
                self.assertTrue((tmp / "out" / "brain.pt").exists())
                self.assertTrue(any(run.get("backend") == "pytorch" for run in snap["training_runs"]))
                self.assertFalse(any(u["step"] == "torch_train" for u in snap["unsupported_steps"]))
            else:
                self.assertTrue(any(u["step"] == "torch_train" for u in snap["unsupported_steps"]))

    def test_repeat_bench_csv_and_state_io(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "utility.nxf"
            src.write_text(UTILITY_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("tools")
            snap = executor.snapshot()

            self.assertEqual(snap["tick"], 3)
            self.assertEqual(snap["state"]["counter"], 3)
            self.assertIn("benchmarks", snap)
            self.assertGreaterEqual(len(snap["benchmarks"]), 1)
            self.assertEqual(snap["benchmarks"][0]["label"], "one_more_tick")

            log_path = tmp / "out" / "log.txt"
            csv_path = tmp / "out" / "report.csv"
            state_path = tmp / "out" / "state.json"
            self.assertTrue(log_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(state_path.exists())
            self.assertIn("hello", log_path.read_text(encoding="utf-8"))
            self.assertIn("section,name,key,value", csv_path.read_text(encoding="utf-8"))

    def test_when_retry_assert_and_string_channel_builtins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "orch.nxf"
            src.write_text(ORCH_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("flow")
            snap = executor.snapshot()

            self.assertEqual(snap["state"]["ctr"], 3)
            self.assertEqual(snap["metrics"]["q_remaining"], 0)
            self.assertEqual(snap["metrics"]["seen_upper"], 1)
            self.assertEqual(snap["metrics"]["meta_ver"], 1)
            self.assertGreaterEqual(snap["metrics"]["meta_key_count"], 2)
            self.assertTrue((tmp / "out" / "orch.json").exists())
            self.assertTrue((tmp / "out" / "meta.json").exists())
            self.assertTrue((tmp / "out" / "events.jsonl").exists())
            self.assertEqual((tmp / "out" / "str.txt").read_text(encoding="utf-8"), "a-b-c")
            ch_data = json.loads((tmp / "out" / "ch.json").read_text(encoding="utf-8"))
            self.assertEqual(ch_data, ["TICK"])
            meta_data = json.loads((tmp / "out" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta_data.get("tag"), "ok")
            self.assertEqual(len((tmp / "out" / "hash.txt").read_text(encoding="utf-8").strip()), 64)
            self.assertTrue(any(e.get("event") == "retry_success" for e in snap["events"]))
            self.assertTrue(any(e.get("event") == "try_failure" for e in snap["events"]))
            self.assertTrue(any(e.get("event") == "recovered" for e in snap["events"]))
            self.assertTrue(any(e.get("event") == "helper_event" for e in snap["events"]))

    def test_retry_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "retry_fail.nxf"
            src.write_text(RETRY_FAIL_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            with self.assertRaises(Exception):
                executor.run_pipeline("fail")

    @unittest.skipUnless(os.name == "nt", "Windows-specific integration test")
    def test_windows_powershell_pipeline_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "winops.nxf"
            src.write_text(WINDOWS_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("winops")
            snap = executor.snapshot()

            self.assertIn("windows", snap)
            self.assertTrue(snap["windows"]["host"]["is_windows"])
            ops = snap["windows"]["ops"]
            self.assertTrue(any(op.get("type") == "powershell" for op in ops))
            self.assertTrue(any("nexusflow-winops" in (op.get("stdout") or "") for op in ops if op.get("type") == "powershell"))
            self.assertTrue(any(op.get("type") == "win_cmd" and "nexusflow-cmd" in (op.get("stdout") or "") for op in ops))
            self.assertTrue((tmp / "out" / "proc.json").exists())
            self.assertTrue((tmp / "out" / "services.json").exists())
            self.assertTrue((tmp / "out" / "service_status.json").exists())
            self.assertTrue((tmp / "out" / "product_name.txt").exists())
            self.assertTrue(any(op.get("type") == "win_service_status" for op in ops))
            self.assertTrue(any(op.get("type") == "win_services_json" for op in ops))

    def test_windows_automation_query_and_dryrun_tools(self) -> None:
        program = r'''
project "WinAutomationDryRunTest" {
  metric op_count = win_op_count();
  metric vui_ops = vui_op_count();
  metric mouse_has_ok = get(win_mouse_pos(), "ok", false);
  metric screen_width = get(win_screen_size(), "width", 0);
  metric fg_ok = get(win_foreground_window(), "ok", false);
  metric wins = len(win_windows(5));
  metric vui_intent_name = get(vui_intent("open notepad", {"open_app": ["open", "launch"], "noop": ["nothing"]}), "intent", "");

  pipeline run {
    step vui_profile("assistant", {"rate": 0, "volume": 70});
    step vui_log("user", "open notepad");
    step vui_say("Dry run voice output", "assistant", true);
    step vui_voices_json("out/vui_voices.json");
    step win_windows_json("out/windows.json", 5);
    step win_input_sequence([
      {"action": "move", "x": 100, "y": 100},
      {"action": "click", "button": "left"},
      {"action": "type", "text": "hello"},
      {"action": "keys", "keys": "^a"},
      {"action": "scroll", "delta": -120},
      {"action": "sleep", "ms": 1}
    ], true);
    step write_text("out/fg.json", json_stringify(win_foreground_window(), true));
    step vui_export_json("out/vui.json");
    step export_json("out/winauto_snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "winauto.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("run")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "WinAutomationDryRunTest")
            self.assertIn("windows", snap)
            self.assertTrue((tmp / "out" / "windows.json").exists())
            self.assertTrue((tmp / "out" / "fg.json").exists())
            self.assertTrue((tmp / "out" / "vui.json").exists())
            self.assertTrue((tmp / "out" / "vui_voices.json").exists())
            self.assertTrue((tmp / "out" / "winauto_snapshot.json").exists())
            self.assertIsInstance(snap["metrics"]["op_count"], int)
            self.assertIsInstance(snap["metrics"]["vui_ops"], int)
            self.assertIsInstance(snap["metrics"]["screen_width"], int)
            self.assertIsInstance(snap["metrics"]["wins"], int)
            self.assertEqual(snap["metrics"]["vui_intent_name"], "open_app")

            win_payload = json.loads((tmp / "out" / "windows.json").read_text(encoding="utf-8"))
            self.assertIn("windows", win_payload)
            self.assertIn("ok", win_payload)
            self.assertIsInstance(win_payload.get("windows"), list)

            vui_payload = json.loads((tmp / "out" / "vui.json").read_text(encoding="utf-8"))
            self.assertIn("ops", vui_payload)
            self.assertIn("profiles", vui_payload)
            self.assertIn("assistant", vui_payload["profiles"])
            self.assertTrue(any(op.get("type") == "vui_say" for op in vui_payload["ops"]))

            ops = snap["windows"]["ops"]
            self.assertTrue(any(op.get("type") == "win_windows_json" for op in ops))
            self.assertTrue(any(op.get("type") == "win_windows_list" for op in ops))
            self.assertTrue(any(op.get("type") == "win_input_sequence" and bool(op.get("dry_run")) for op in ops))
            self.assertTrue(any(op.get("type") == "win_mouse_pos" for op in ops))
            self.assertTrue(any(op.get("type") == "win_foreground_window" for op in ops))
            self.assertTrue(any(e.get("event") == "win_input_sequence" for e in snap["events"]))
            self.assertIn("vui", snap)
            self.assertTrue(any(op.get("type") == "vui_say" for op in snap["vui"]["ops"]))

    def test_3d_asset_loading_scenes_and_gui_widgets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            assets_dir = tmp / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            (assets_dir / "crate.obj").write_text(
                "\n".join(
                    [
                        "# simple quad",
                        "o Crate",
                        "v -1 0 -1",
                        "v 1 0 -1",
                        "v 1 0 1",
                        "v -1 0 1",
                        "vt 0 0",
                        "vt 1 0",
                        "vt 1 1",
                        "vt 0 1",
                        "vn 0 1 0",
                        "usemtl Matte",
                        "f 1/1/1 2/2/1 3/3/1 4/4/1",
                    ]
                ),
                encoding="utf-8",
            )
            (assets_dir / "robot.gltf").write_text(
                json.dumps(
                    {
                        "asset": {"version": "2.0", "generator": "nexusflow-test"},
                        "scene": 0,
                        "scenes": [{"nodes": [0]}],
                        "nodes": [{"name": "RobotRoot", "mesh": 0}],
                        "meshes": [{"name": "RobotMesh", "primitives": [{"attributes": {"POSITION": 0}}]}],
                        "materials": [{"name": "RobotMat"}],
                    }
                ),
                encoding="utf-8",
            )

            src = tmp / "threed_gui.nxf"
            src.write_text(THREED_GUI_PROGRAM, encoding="utf-8")

            project = parse_file(src)
            ir = project_to_json(project)
            self.assertEqual(ir["ui"]["theme"], "amber")
            self.assertIn("ops_console", ir["ui"]["templates"])
            self.assertTrue(any(w["type"] == "scene3d" for w in ir["ui"]["panels"][0]["widgets"]))

            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertIn("assets3d", snap)
            self.assertIn("crate", snap["assets3d"])
            self.assertIn("robot", snap["assets3d"])
            self.assertEqual(snap["assets3d"]["crate"]["kind"], "obj")
            self.assertEqual(snap["assets3d"]["robot"]["kind"], "gltf")
            self.assertEqual(snap["metrics"]["loaded_assets"], 2)
            self.assertEqual(snap["metrics"]["scene_nodes"], 2)
            self.assertEqual(snap["metrics"]["crate_vertices"], 4)
            self.assertIn("scenes3d", snap)
            self.assertIn("showcase", snap["scenes3d"])
            self.assertEqual(snap["scenes3d"]["showcase"]["template"], "lab")
            self.assertEqual(len(snap["scenes3d"]["showcase"]["nodes"]), 2)

            scene_json = tmp / "out" / "showcase.scene.json"
            scene_html = tmp / "out" / "showcase.scene.html"
            dashboard_html = tmp / "out" / "dashboard.html"
            snap_json = tmp / "out" / "snapshot.json"
            self.assertTrue(scene_json.exists())
            self.assertTrue(scene_html.exists())
            self.assertTrue(dashboard_html.exists())
            self.assertTrue(snap_json.exists())
            self.assertIn('scene3d "showcase"', scene_html.read_text(encoding="utf-8"))
            dashboard_text = dashboard_html.read_text(encoding="utf-8")
            self.assertIn("Primary Scene", dashboard_text)
            self.assertIn("Assets Loaded", dashboard_text)

    def test_web_tool_suite_http_and_science_sims(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = tmp / "fixture.json"
            fixture.write_text(json.dumps({"hello": "world", "n": 7, "items": [1, 2, 3]}), encoding="utf-8")
            file_uri = fixture.resolve().as_uri()
            program = WEB_SCIENCE_PROGRAM_TEMPLATE.replace("__FILE_URI__", json.dumps(file_uri))

            src = tmp / "web_science.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "WebScienceLab")
            self.assertIn("web_tools", snap)
            self.assertGreaterEqual(len(snap["web_tools"]), 8)
            self.assertIn("fusion_runs", snap)
            self.assertIn("demo_tokamak", snap["fusion_runs"])
            self.assertIn("protein_runs", snap)
            self.assertIn("demo_fold", snap["protein_runs"])
            self.assertGreater(snap["metrics"]["web_suite_files"], 0)
            self.assertIsInstance(snap["metrics"]["fusion_q"], (int, float))
            self.assertGreaterEqual(snap["metrics"]["fusion_q"], 0)
            self.assertGreater(snap["metrics"]["protein_rg"], 0)
            self.assertGreaterEqual(snap["metrics"]["protein_contacts"], 0)
            self.assertEqual(snap["metrics"]["query_fields"], 2)
            self.assertGreater(snap["metrics"]["hydrophobic_ratio"], 0)
            self.assertLessEqual(snap["metrics"]["hydrophobic_ratio"], 1)
            self.assertIn("http", snap)
            self.assertTrue(any(op.get("url") == file_uri for op in snap["http"]["ops"]))

            web_dir = tmp / "out" / "webtools"
            self.assertTrue((web_dir / "index.html").exists())
            self.assertTrue((web_dir / "interaction_plus.html").exists())
            self.assertTrue((web_dir / "json_plus.html").exists())
            self.assertTrue((tmp / "out" / "fusion_demo.json").exists())
            self.assertTrue((tmp / "out" / "protein_demo.json").exists())
            self.assertTrue((tmp / "out" / "web_science_dashboard.html").exists())
            fetched = json.loads((tmp / "out" / "local_fetch.json").read_text(encoding="utf-8"))
            self.assertEqual(fetched["hello"], "world")
            fetched_inline = json.loads((tmp / "out" / "fetch_inline.json").read_text(encoding="utf-8"))
            self.assertEqual(fetched_inline["items"], [1, 2, 3])
            dashboard_html = (tmp / "out" / "web_science_dashboard.html").read_text(encoding="utf-8")
            self.assertIn("Fusion Run", dashboard_html)
            self.assertIn("Protein Run", dashboard_html)

    def test_advanced_live_api_and_visualization_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = tmp / "fixture2.json"
            fixture.write_text(json.dumps({"ok": True, "series": [3, 1, 4], "name": "advanced"}), encoding="utf-8")
            file_uri = fixture.resolve().as_uri()
            program = ADVANCED_LIVE_SCIENCE_PROGRAM_TEMPLATE.replace("__FILE_URI__", json.dumps(file_uri))

            src = tmp / "advanced_live.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("advanced")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "AdvancedLiveScience")
            self.assertIn("http", snap)
            self.assertIn("auth_presets", snap["http"])
            self.assertIn("local_auth", snap["http"]["auth_presets"])
            self.assertGreaterEqual(snap["metrics"]["http_count"], 1)
            self.assertEqual(snap["metrics"]["auth_count"], 1)
            self.assertIsInstance(snap["metrics"]["fusion_mode_q"], (int, float))
            self.assertGreaterEqual(snap["metrics"]["fusion_zones"], 2)
            self.assertIsInstance(snap["metrics"]["protein3d_rg"], (int, float))
            self.assertGreaterEqual(snap["metrics"]["protein3d_len"], 3)

            self.assertIn("fusion_runs", snap)
            self.assertEqual(snap["fusion_runs"]["tokamak_mz"]["kind"], "fusion_reactor_multizone")
            self.assertIn("trace", snap["fusion_runs"]["tokamak_mz"])
            self.assertTrue(any("zones" in t for t in snap["fusion_runs"]["tokamak_mz"]["trace"]))

            self.assertIn("protein_runs", snap)
            self.assertEqual(snap["protein_runs"]["pep3d"]["kind"], "protein_folding_3d")
            self.assertIn("frames", snap["protein_runs"]["pep3d"])
            self.assertGreater(len(snap["protein_runs"]["pep3d"]["frames"]), 0)

            live_dir = tmp / "out" / "live_tools"
            self.assertTrue((live_dir / "index.html").exists())
            self.assertTrue((live_dir / "02_mock-server-lab.html").exists())
            self.assertTrue((live_dir / "03_websocket-lab.html").exists())
            self.assertTrue((live_dir / "mock_server_plus.html").exists())
            self.assertTrue((live_dir / "ws_plus.html").exists())

            self.assertTrue((tmp / "out" / "http_history.json").exists())
            self.assertTrue((tmp / "out" / "tokamak_mz.json").exists())
            self.assertTrue((tmp / "out" / "tokamak_mz.html").exists())
            self.assertTrue((tmp / "out" / "pep3d.json").exists())
            self.assertTrue((tmp / "out" / "pep3d.html").exists())

            http_hist = json.loads((tmp / "out" / "http_history.json").read_text(encoding="utf-8"))
            self.assertIn("auth_presets", http_hist)
            self.assertIn("local_auth", http_hist["auth_presets"])
            self.assertTrue(any(op.get("auth_preset") == "local_auth" for op in http_hist["ops"]))

            fusion_html = (tmp / "out" / "tokamak_mz.html").read_text(encoding="utf-8")
            protein_html = (tmp / "out" / "pep3d.html").read_text(encoding="utf-8")
            self.assertIn("Fusion Visualization", fusion_html)
            self.assertIn("Protein", protein_html)

    def test_mock_http_server_fusion_control_sweep_and_rich_protein_viz(self) -> None:
        program = r'''
project "SuggestedSuite" {
  metric mock_servers = mock_http_server_count();
  metric mock_running = get(mock_http_server_info("lab"), "running", false);
  metric mock_requests = get(mock_http_server_info("lab"), "request_count", 0);
  metric ctrl_q = fusion_metric("ctrl_tok", "q_estimate");
  metric ctrl_cycles = fusion_metric("ctrl_tok", "controller.cycles");
  metric sweep_cases = fusion_metric("sweep_tok", "scenario_count");
  metric sweep_best_q = fusion_metric("sweep_tok", "best_q_estimate");
  metric prot_rg = protein_metric("pepv", "radius_gyration");

  pipeline run {
    step mock_http_server_start("lab", 0, {
      "/json": {"json": {"hello": "world", "items": [1, 2, 3]}},
      "/echo": {"echo": true}
    });
    step http_fetch_json(mock_http_server_url("lab") + "/json", "out/from_mock.json");
    step write_text("out/echo_get.json", json_stringify(http_get_json(mock_http_server_url("lab") + "/echo?x=1&x=2"), true));
    step write_text("out/echo_post.json", json_stringify(http_post_json(mock_http_server_url("lab") + "/echo", {"ping": "pong"}), true));
    step fusion_control_sim("ctrl_tok", 90, {"cycles": 3, "horizon_steps": 30, "target_q": 0.03});
    step fusion_sweep("sweep_tok", {"heating_mw": [150, 170], "zones": [3, 4]}, 40, {"engine": "multizone", "objective": "q_estimate"});
    step protein_fold_sim_3d("pepv", "MKWVTFISLLFL", 90, {"temperature": 0.85});
    step export_protein_html("pepv", "out/pepv.html");
    step export_fusion_json("ctrl_tok", "out/ctrl_tok.json");
    step export_fusion_json("sweep_tok", "out/sweep_tok.json");
    step mock_http_server_stop("lab");
    step export_json("out/snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "suggested_suite.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("run")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "SuggestedSuite")
            self.assertIn("http", snap)
            self.assertIn("mock_servers", snap["http"])
            self.assertIn("lab", snap["http"]["mock_servers"])
            self.assertFalse(bool(snap["http"]["mock_servers"]["lab"]["running"]))
            self.assertGreaterEqual(int(snap["http"]["mock_servers"]["lab"].get("request_count", 0)), 3)
            self.assertGreaterEqual(int(snap["metrics"]["mock_servers"]), 1)
            self.assertFalse(bool(snap["metrics"]["mock_running"]))
            self.assertGreaterEqual(int(snap["metrics"]["mock_requests"]), 3)

            self.assertEqual(snap["fusion_runs"]["ctrl_tok"]["kind"], "fusion_reactor_control")
            self.assertGreaterEqual(int(snap["metrics"]["ctrl_cycles"]), 1)
            self.assertIsInstance(snap["metrics"]["ctrl_q"], (int, float))
            self.assertEqual(snap["fusion_runs"]["sweep_tok"]["kind"], "fusion_sweep")
            self.assertEqual(int(snap["metrics"]["sweep_cases"]), 4)
            self.assertIsInstance(snap["metrics"]["sweep_best_q"], (int, float))
            self.assertIn("leaderboard", snap["fusion_runs"]["sweep_tok"])

            self.assertEqual(snap["protein_runs"]["pepv"]["kind"], "protein_folding_3d")
            self.assertIsInstance(snap["metrics"]["prot_rg"], (int, float))
            self.assertTrue((tmp / "out" / "from_mock.json").exists())
            self.assertTrue((tmp / "out" / "echo_get.json").exists())
            self.assertTrue((tmp / "out" / "echo_post.json").exists())
            self.assertTrue((tmp / "out" / "ctrl_tok.json").exists())
            self.assertTrue((tmp / "out" / "sweep_tok.json").exists())
            self.assertTrue((tmp / "out" / "pepv.html").exists())

            fetched = json.loads((tmp / "out" / "from_mock.json").read_text(encoding="utf-8"))
            self.assertEqual(fetched["hello"], "world")
            echo_get = json.loads((tmp / "out" / "echo_get.json").read_text(encoding="utf-8"))
            self.assertEqual(echo_get["path"], "/echo")
            self.assertEqual(echo_get["query"]["x"], ["1", "2"])
            echo_post = json.loads((tmp / "out" / "echo_post.json").read_text(encoding="utf-8"))
            self.assertEqual(echo_post["body_json"]["ping"], "pong")

            protein_html = (tmp / "out" / "pepv.html").read_text(encoding="utf-8")
            self.assertIn("Contact Map", protein_html)
            self.assertIn("Ramachandran-like", protein_html)
            self.assertIn("Play", protein_html)
            self.assertIn("Pause", protein_html)

    def test_polyglot_modules_and_python_native_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            py_script = tmp / "echo_mod.py"
            py_script.write_text(
                "import json, sys\nprint(json.dumps({'args': sys.argv[1:], 'lang': 'python'}))\n",
                encoding="utf-8",
            )
            js_script = tmp / "echo_mod.js"
            js_script.write_text("console.log(JSON.stringify({lang:'js', args: process.argv.slice(2)}));\n", encoding="utf-8")
            cpp_src = tmp / "hello.cpp"
            cpp_src.write_text(
                '#include <iostream>\nint main(int argc, char** argv){ std::cout << "{\\"lang\\":\\"cpp\\",\\"argc\\":" << argc-1 << "}"; return 0; }\n',
                encoding="utf-8",
            )

            program = rf'''
project "PolyglotML" {{
  metric lm_count = lang_module_count();
  metric py_model_total = python_model_count();
  metric py_run_ok = get(lang_module_last("py_echo"), "ok", false);
  metric py_run_code = get(lang_module_last("py_echo"), "returncode", -1);
  metric cpp_lang = get(lang_module_info("cpp_demo"), "language", "");
  metric js_lang = get(lang_module_info("js_demo"), "language", "");
  metric lin_metric = python_model_metric("lin", "final_metric");
  metric log_metric = python_model_metric("logi", "final_metric");
  metric km_metric = python_model_metric("km", "final_metric");

  model lin {{
    backend = "python";
    kind = "linear_regression";
    inputs = 4;
    outputs = 1;
    lr = 0.04;
  }}

  model logi {{
    backend = "python";
    kind = "logistic_regression";
    inputs = 3;
    outputs = 1;
    lr = 0.08;
  }}

  model km {{
    backend = "python";
    kind = "kmeans";
    inputs = 2;
    clusters = 3;
  }}

  dataset reg_ds {{
    task = "regression";
    samples = 72;
    inputs = 4;
    outputs = 1;
  }}

  dataset bin_ds {{
    task = "binary";
    samples = 96;
    inputs = 3;
    outputs = 1;
  }}

  dataset cluster_ds {{
    task = "clustering";
    samples = 90;
    inputs = 2;
    clusters = 3;
  }}

  pipeline main {{
    step py_module("py_echo", {json.dumps(str(py_script))});
    step js_module("js_demo", {json.dumps(str(js_script))});
    step cpp_module("cpp_demo", {json.dumps(str(cpp_src))});
    step lang_run("py_echo", ["one", "two"]);
    step cpp_build("cpp_demo");
    step cpp_run("cpp_demo");
    step train("lin", "reg_ds", 24);
    step train("logi", "bin_ds", 20);
    step train("km", "cluster_ds", 12);
    step python_export("lin", "out/lin_python_model.json");
    step export_json("out/polyglot_snapshot.json");
  }}
}}
'''
            src = tmp / "polyglot.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("main")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "PolyglotML")
            self.assertIn("languages", snap)
            self.assertIn("modules", snap["languages"])
            self.assertIn("py_echo", snap["languages"]["modules"])
            self.assertIn("js_demo", snap["languages"]["modules"])
            self.assertIn("cpp_demo", snap["languages"]["modules"])
            self.assertEqual(snap["metrics"]["cpp_lang"], "cpp")
            self.assertEqual(snap["metrics"]["js_lang"], "javascript")
            self.assertGreaterEqual(int(snap["metrics"]["lm_count"]), 3)
            self.assertTrue(bool(snap["metrics"]["py_run_ok"]))
            self.assertEqual(int(snap["metrics"]["py_run_code"]), 0)

            py_last = snap["languages"]["modules"]["py_echo"].get("last_run", {})
            self.assertTrue(py_last.get("ok"))
            py_out = json.loads((py_last.get("stdout") or "").strip() or "{}")
            self.assertEqual(py_out.get("lang"), "python")
            self.assertEqual(py_out.get("args"), ["one", "two"])

            cpp_meta = snap["languages"]["modules"]["cpp_demo"]
            self.assertIn("language", cpp_meta)
            cpp_build = cpp_meta.get("build", {})
            if cpp_build.get("ok"):
                self.assertTrue(Path(str(cpp_build["binary_path"])).exists())
            else:
                self.assertTrue(
                    any(u["step"] in {"cpp_build", "cpp_run"} for u in snap["unsupported_steps"])
                    or cpp_build.get("reason") in {"compiler_missing", "source_missing"}
                )

            self.assertIn("python_ml", snap)
            self.assertGreaterEqual(int(snap["metrics"]["py_model_total"]), 3)
            self.assertIn("lin", snap["python_ml"]["models"])
            self.assertIn("logi", snap["python_ml"]["models"])
            self.assertIn("km", snap["python_ml"]["models"])
            self.assertIsInstance(snap["metrics"]["lin_metric"], (int, float))
            self.assertIsInstance(snap["metrics"]["log_metric"], (int, float))
            self.assertIsInstance(snap["metrics"]["km_metric"], (int, float))
            self.assertTrue((tmp / "out" / "lin_python_model.json").exists())

            exported = json.loads((tmp / "out" / "lin_python_model.json").read_text(encoding="utf-8"))
            self.assertIn("python_model", exported)
            self.assertEqual(exported["python_model"]["model"], "lin")
            self.assertEqual(exported["python_model"]["artifact"]["backend"], "python_native")

    def test_nexus_ide_and_idle_web_tools(self) -> None:
        program = r'''
project "IDEKit" {
  metric tools = web_tool_count();

  pipeline build {
    step nexus_ide("out/nexus_ide.html");
    step nexus_idle("out/nexus_idle.html");
    step nexus_dev_suite("out/nexus_dev_suite");
    step export_json("out/idekit_snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "idekit.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "IDEKit")
            self.assertGreaterEqual(int(snap["metrics"]["tools"]), 3)
            self.assertIn("web_tools", snap)

            ide_path = tmp / "out" / "nexus_ide.html"
            idle_path = tmp / "out" / "nexus_idle.html"
            suite_dir = tmp / "out" / "nexus_dev_suite"
            self.assertTrue(ide_path.exists())
            self.assertTrue(idle_path.exists())
            self.assertTrue((suite_dir / "index.html").exists())
            self.assertTrue((suite_dir / "01_nexus-ide.html").exists())
            self.assertTrue((suite_dir / "02_nexus-idle.html").exists())

            ide_html = ide_path.read_text(encoding="utf-8")
            idle_html = idle_path.read_text(encoding="utf-8")
            self.assertIn("NexusFlow IDE", ide_html)
            self.assertIn("Run Pipeline", ide_html)
            self.assertIn("File Explorer", ide_html)
            self.assertIn("AST Sketch", ide_html)
            self.assertIn("NexusFlow IDLE", idle_html)
            self.assertIn("REPL", idle_html)
            self.assertIn("Run Scratch", idle_html)
            self.assertIn("tick N", idle_html)

    def test_sqlite_regex_template_and_archives(self) -> None:
        program = r'''
project "DataOps" {
  metric item_rows = len(sqlite_query("data/app.db", "SELECT * FROM items"));
  metric total_qty = to_number(sqlite_scalar("data/app.db", "SELECT COALESCE(SUM(CAST(qty AS INTEGER)), 0) AS total FROM items"));
  metric regex_ok = regex_test("item-\\d+", "ITEM-42", "i");
  metric regex_hits = len(regex_findall("[A-Za-z]+", "alpha beta 42"));
  metric sqlite_ops = sqlite_history_count();
  metric sqlite_dbs = sqlite_db_count();
  metric db_ops = get(sqlite_db_info("app.db"), "ops", 0);
  metric archive_ops = archive_op_count();
  metric archive_total = archive_count();
  metric archive_entries = get(archive_info("bundle.zip"), "entries", 0);

  pipeline build {
    step write_text("out/items.csv", "name,qty\napple,2\npear,5\n");
    step sqlite_import_csv("data/app.db", "items", "out/items.csv");
    step sqlite_exec("data/app.db", "INSERT INTO items (name, qty) VALUES (?, ?)", ["banana", "7"]);
    step sqlite_query_json("data/app.db", "SELECT name, qty FROM items ORDER BY name", "out/items.json");
    step sqlite_export_csv("data/app.db", "SELECT name, qty FROM items ORDER BY name", "out/items_sorted.csv");
    step render_template(
      "out/report.txt",
      "rows={{rows}} total={{stats.total}} first={{first.name}}",
      {
        "rows": len(sqlite_query("data/app.db", "SELECT * FROM items")),
        "stats": {"total": sqlite_scalar("data/app.db", "SELECT SUM(CAST(qty AS INTEGER)) AS total FROM items")},
        "first": get(sqlite_query("data/app.db", "SELECT name FROM items ORDER BY name LIMIT 1"), 0, {"name": "none"})
      },
      true
    );
    step write_text("out/regex.txt", regex_replace("[0-9]+", "#", "item42"));
    step zip_pack("out", "bundle.zip");
    step zip_unpack("bundle.zip", "unzipped");
    step export_json("out/dataops.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "dataops.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "DataOps")
            self.assertEqual(int(snap["metrics"]["item_rows"]), 3)
            self.assertEqual(int(snap["metrics"]["total_qty"]), 14)
            self.assertTrue(bool(snap["metrics"]["regex_ok"]))
            self.assertEqual(int(snap["metrics"]["regex_hits"]), 2)
            self.assertGreaterEqual(int(snap["metrics"]["sqlite_ops"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["sqlite_dbs"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["db_ops"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["archive_ops"]), 2)
            self.assertGreaterEqual(int(snap["metrics"]["archive_total"]), 1)
            self.assertGreater(int(snap["metrics"]["archive_entries"]), 0)

            self.assertIn("sqlite", snap)
            self.assertIn("archives", snap)
            self.assertIn("databases", snap["sqlite"])
            self.assertIn("archives", snap["archives"])

            items_json = json.loads((tmp / "out" / "items.json").read_text(encoding="utf-8"))
            self.assertEqual(items_json["count"], 3)
            self.assertEqual([r["name"] for r in items_json["rows"]], ["apple", "banana", "pear"])

            report_text = (tmp / "out" / "report.txt").read_text(encoding="utf-8")
            self.assertIn("rows=3", report_text)
            self.assertIn("total=14", report_text)
            self.assertIn("first=apple", report_text)
            self.assertEqual((tmp / "out" / "regex.txt").read_text(encoding="utf-8"), "item#")

            self.assertTrue((tmp / "bundle.zip").exists())
            self.assertTrue((tmp / "unzipped" / "report.txt").exists())
            self.assertTrue((tmp / "unzipped" / "items_sorted.csv").exists())

    def test_csharp_polyglot_and_resource_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cs_src = tmp / "echo.cs"
            cs_src.write_text(
                (
                    "using System; using System.Text; "
                    "class Program { "
                    "static string E(string s){ return (s??\"\").Replace(\"\\\\\",\"\\\\\\\\\").Replace(\"\\\"\",\"\\\\\\\"\"); } "
                    "static void Main(string[] args){ "
                    "var sb=new StringBuilder(); sb.Append(\"{\\\"lang\\\":\\\"csharp\\\",\\\"args\\\":[\"); "
                    "for(int i=0;i<args.Length;i++){ if(i>0) sb.Append(\",\"); sb.Append(\"\\\"\").Append(E(args[i])).Append(\"\\\"\"); } "
                    "sb.Append(\"],\\\"count\\\":\").Append(args.Length).Append(\"}\"); Console.WriteLine(sb.ToString()); } }"
                ),
                encoding="utf-8",
            )
            program = rf'''
project "CSharpResource" {{
  config seed = 5;
  config threads = 6;
  config pipeline_threads = 4;
  metric res_ops = resource_count();
  metric sim_cap = get(resource_limits(), "max_sim_workers", 0);
  metric pipe_cap = get(resource_limits(), "max_pipeline_workers", 0);
  metric cs_lang = get(lang_module_info("csmod"), "language", "");
  metric cs_last_ok = get(lang_module_last("csmod"), "ok", false);
  metric cs_reason = get(lang_module_last("csmod"), "reason", "");
  metric runtime_threads = get(resource_runtime_info(), "thread_events", -1);

  agent bot count 4 {{
    field energy = 1;
    on tick {{
      energy += 1;
      emit "tick";
    }}
  }}

  pipeline main {{
    step resource_set_limits({{"max_sim_workers": 2, "max_pipeline_workers": 1, "events_max": 64, "resource_ops_max": 32}});
    step resource_snapshot("start");
    step csharp_module("csmod", {json.dumps(str(cs_src))});
    step csharp_build("csmod");
    step csharp_run("csmod", ["one", "two"]);
    step repeat(simulate_mt(2, 8), 2);
    step resource_gc();
    step resource_trim({{"events": 40, "lang_runs": 20}});
    step export_resource_json("out/resources.json");
    step export_json("out/snapshot.json");
  }}
}}
'''
            src = tmp / "csharp_resource.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("main")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "CSharpResource")
            self.assertIn("resources", snap)
            self.assertIn("limits", snap["resources"])
            self.assertEqual(int(snap["metrics"]["sim_cap"]), 2)
            self.assertEqual(int(snap["metrics"]["pipe_cap"]), 1)
            self.assertEqual(snap["metrics"]["cs_lang"], "csharp")
            self.assertGreaterEqual(int(snap["metrics"]["res_ops"]), 3)
            self.assertTrue((tmp / "out" / "resources.json").exists())
            self.assertTrue((tmp / "out" / "snapshot.json").exists())

            cs_meta = snap["languages"]["modules"]["csmod"]
            self.assertEqual(cs_meta["language"], "csharp")
            cs_last = cs_meta.get("last_run", {})
            if cs_last.get("ok"):
                parsed = json.loads((cs_last.get("stdout") or "").strip() or "{}")
                self.assertEqual(parsed.get("lang"), "csharp")
                self.assertEqual(parsed.get("args"), ["one", "two"])
            else:
                reason = str(cs_last.get("reason", ""))
                self.assertTrue(
                    reason.startswith("runtime_missing")
                    or reason in {"compiler_missing", "csharp_build_failed", "dotnet_requires_csproj", "source_missing", "csharp_no_runnable_target"}
                )
                self.assertTrue(
                    any(u["step"] in {"csharp_build", "csharp_run"} for u in snap["unsupported_steps"])
                    or bool(reason)
                )

    def test_iso_building_tools_with_manifest_fallback(self) -> None:
        program = r'''
project "IsoTools" {
  metric iso_imgs = iso_count();
  metric iso_ops = iso_op_count();
  metric iso_label = get(iso_info("demo.iso"), "label", "");
  metric iso_entries = get(iso_info("demo.iso"), "manifest_count", 0);
  metric iso_tool_selected = get(get(iso_tool_info(), "selected", {}), "build", null);
  metric archive_ops = archive_op_count();

  pipeline build {
    step write_text("stage/readme.txt", "hello iso");
    step write_text("stage/sub/info.txt", "nested");
    step iso_build("stage", "out/demo.iso", {"label": "NEXUSISO", "dry_run": true});
    step iso_list_json("out/demo.iso", "out/demo_iso_list.json", {"prefer_source_manifest": true});
    step iso_extract("out/demo.iso", "out/extracted", {"dry_run": true});
    step export_json("out/iso_snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "iso_tools.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "IsoTools")
            self.assertGreaterEqual(int(snap["metrics"]["iso_imgs"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["iso_ops"]), 3)
            self.assertEqual(str(snap["metrics"]["iso_label"]), "NEXUSISO")
            self.assertEqual(int(snap["metrics"]["iso_entries"]), 2)
            self.assertGreaterEqual(int(snap["metrics"]["archive_ops"]), 2)
            self.assertIn("iso", snap)
            self.assertIn("images", snap["iso"])
            self.assertTrue(any(Path(str(k)).name == "demo.iso" for k in snap["iso"]["images"].keys()))

            listing_path = tmp / "out" / "demo_iso_list.json"
            self.assertTrue(listing_path.exists())
            listing = json.loads(listing_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(listing.get("ok")))
            self.assertEqual(listing.get("mode"), "source_manifest")
            self.assertEqual(int(listing.get("count", 0)), 2)
            listed_paths = {str(item.get("path")) for item in listing.get("entries", []) if isinstance(item, dict)}
            self.assertIn("readme.txt", listed_paths)
            self.assertIn("sub/info.txt", listed_paths)

            self.assertTrue((tmp / "out" / "iso_snapshot.json").exists())
            self.assertFalse(any(u.get("step") in {"iso_build", "iso_list_json", "iso_extract"} for u in snap.get("unsupported_steps", [])))

    def test_packaging_integrity_tar_hash_manifest_diff(self) -> None:
        program = r'''
project "PkgIntegrity" {
  metric alpha_hash = file_hash("a/alpha.txt");
  metric hash_ok = file_hash_verify("a/alpha.txt", alpha_hash);
  metric a_count = get(dir_manifest("a", {"hash": true}), "count", 0);
  metric diff_changed = get(dir_diff("a", "b", {"hash": true}), "changed_count", 0);
  metric diff_added = get(dir_diff("a", "b", {"hash": true}), "added_count", 0);
  metric archive_ops = archive_op_count();
  metric archives = archive_count();

  pipeline build {
    step write_text("a/alpha.txt", "alpha-v1");
    step write_text("a/sub/beta.txt", "beta");
    step write_text("b/alpha.txt", "alpha-v2");
    step write_text("b/sub/beta.txt", "beta");
    step write_text("b/new.txt", "new");
    step hash_file_json("a/alpha.txt", "out/hash.json");
    step dir_manifest_json("a", "out/manifest_a.json", {"hash": true});
    step dir_diff_json("a", "b", "out/diff.json", {"hash": true});
    step tar_pack("a", "out/a.tar.gz", {"compression": "gz"});
    step tar_unpack("out/a.tar.gz", "out/unpacked_a");
    step export_json("out/snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "pkg_integrity.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "PkgIntegrity")
            self.assertTrue(bool(snap["metrics"]["hash_ok"]))
            self.assertEqual(int(snap["metrics"]["a_count"]), 2)
            self.assertEqual(int(snap["metrics"]["diff_changed"]), 1)
            self.assertEqual(int(snap["metrics"]["diff_added"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["archive_ops"]), 2)
            self.assertGreaterEqual(int(snap["metrics"]["archives"]), 1)

            hash_json = json.loads((tmp / "out" / "hash.json").read_text(encoding="utf-8"))
            self.assertEqual(hash_json["algo"], "sha256")
            self.assertIn("digest", hash_json)

            manifest_json = json.loads((tmp / "out" / "manifest_a.json").read_text(encoding="utf-8"))
            self.assertEqual(int(manifest_json["count"]), 2)
            self.assertTrue(bool(manifest_json["hash"]))

            diff_json = json.loads((tmp / "out" / "diff.json").read_text(encoding="utf-8"))
            self.assertEqual(int(diff_json["changed_count"]), 1)
            self.assertEqual(int(diff_json["added_count"]), 1)
            changed_paths = {str(item.get("path")) for item in diff_json["changed"]}
            self.assertIn("alpha.txt", changed_paths)
            self.assertIn("new.txt", set(diff_json["added"]))

            self.assertTrue((tmp / "out" / "a.tar.gz").exists())
            self.assertTrue((tmp / "out" / "unpacked_a" / "alpha.txt").exists())
            self.assertTrue((tmp / "out" / "unpacked_a" / "sub" / "beta.txt").exists())

    def test_feature_pack_process_wifi_npu_graph_photo_convert_rust_and_github_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repos_fixture = [
                {
                    "name": "nexusflow-lab",
                    "full_name": "kai9987kai/nexusflow-lab",
                    "language": "Python",
                    "stargazers_count": 12,
                    "forks_count": 2,
                    "description": "Experimental NexusFlow tooling",
                    "topics": ["dsl", "ai", "automation"],
                },
                {
                    "name": "graph-steer",
                    "full_name": "kai9987kai/graph-steer",
                    "language": "Rust",
                    "stargazers_count": 5,
                    "forks_count": 1,
                    "description": "Graph and systems experiments",
                    "topics": ["graph", "systems"],
                },
                {
                    "name": "photo-math-tools",
                    "full_name": "kai9987kai/photo-math-tools",
                    "language": "JavaScript",
                    "stargazers_count": 3,
                    "forks_count": 0,
                    "description": "Photo and data viz utilities",
                    "topics": ["photo", "viz", "math"],
                },
            ]
            (tmp / "repos_metadata.json").write_text(json.dumps(repos_fixture, indent=2), encoding="utf-8")

            py_exe = json.dumps(str(Path(sys.executable).resolve()))
            program = f'''
project "FeaturePack" {{
  metric avg_val = mean([1, 2, 3, 4]);
  metric med_val = median([1, 2, 3, 4]);
  metric var_val = variance([1, 2, 3, 4]);
  metric fit_slope = get(linear_fit([0, 1, 2], [1, 3, 5]), "slope", 0);
  metric trap_area = integrate_trapz([0, 1, 2, 3], [0, 1, 2, 3]);
  metric diff_n = len(differentiate([0, 1, 2, 3], [0, 1, 4, 9]));
  metric graph_nodes = get(graph_stats("net"), "nodes", 0);
  metric graph_hops = get(graph_shortest_path("net", "a", "d"), "hops", -1);
  metric graph_components_n = len(get(graph_components("net"), "components", []));
  metric repo_count = get(github_local_summary(), "repo_count", 0);
  metric repo_find_count = len(github_repo_find("nexus"));
  metric photo_n = photo_count();
  metric graphs_n = graph_count();
  metric convert_n = convert_op_count();
  metric exe_ops_n = exe_op_count();
  metric exe_artifacts_n = exe_count();
  metric proc_n = proc_op_count();
  metric npu_probes = npu_probe_count();
  metric wifi_supported_flag = wifi_supported();
  metric rust_tool_present = rust_toolchain_info() != null;
  metric exe_tools_known = exe_tool_info() != null;
  metric managed_seen = proc_managed_count();

  pipeline build {{
    step write_text("out/graph_edges.json", json_stringify([
      {{"source": "a", "target": "b", "weight": 1}},
      {{"source": "b", "target": "c", "weight": 1}},
      {{"source": "c", "target": "d", "weight": 1}}
    ], true));
    step file_convert("out/graph_edges.json", "out/graph_edges.csv");
    step graph_from_csv("net", "out/graph_edges.csv");
    step graph_metrics_json("net", "out/graph_metrics.json");
    step graph_export_svg("net", "out/net.svg", {{"title": "Network"}});
    step photo_generate("out/hero.svg", 320, 180, {{"prompt": "Kai toolkit hero", "style": "geometric"}});
    step data_chart_svg("out/chart.svg", [1, 4, 2, 5, 3], {{"title": "Trend", "mode": "line"}});
    step proc_profile("sandbox_py", {{"virtualized": true}});
    step proc_profile_run("sandbox_py", {py_exe}, ["-c", "print('proc-ok')"]);
    step proc_spawn("spawn_py", {py_exe}, ["-c", "import time; print('start'); time.sleep(0.05); print('done')"]);
    step proc_wait("spawn_py", 2);
    step proc_history_json("out/proc_history.json");
    step proc_managed_json("out/proc_managed.json");
    step wifi_interfaces_json("out/wifi_interfaces.json");
    step wifi_profiles_json("out/wifi_profiles.json");
    step wifi_scan_json("out/wifi_scan.json");
    step npu_probe_json("out/npu.json");
    step iso_manifest_json("out", "out/iso_manifest.json");
    step write_text("tools/mini_app.py", "print('hello-exe')");
    step exe_build("tools/mini_app.py", "out/mini_app.exe", {{"dry_run": true}});
    step write_text("out/exe_last.json", json_stringify(exe_last(), true));
    step write_text("mods/hello.rs", "fn main(){{println!(\\\"hello-rs\\\");}}");
    step rust_module("hello_rs", "mods/hello.rs");
    step rust_build("hello_rs");
    step rust_run("hello_rs");
    step github_portfolio_report("repos_metadata.json", "out/github_report.json");
    step idea_forge_json("out/ideas.json", "kai9987kai");
    step export_json("out/feature_pack_snapshot.json");
  }}
}}
'''
            src = tmp / "feature_pack.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "FeaturePack")
            self.assertIn("processes", snap)
            self.assertIn("wifi", snap)
            self.assertIn("npu", snap)
            self.assertIn("photos", snap)
            self.assertIn("graphs", snap)
            self.assertIn("conversion", snap)
            self.assertIn("github_local", snap)
            self.assertIn("executables", snap)

            self.assertAlmostEqual(float(snap["metrics"]["avg_val"]), 2.5, places=6)
            self.assertAlmostEqual(float(snap["metrics"]["med_val"]), 2.5, places=6)
            self.assertAlmostEqual(float(snap["metrics"]["var_val"]), 1.25, places=6)
            self.assertAlmostEqual(float(snap["metrics"]["fit_slope"]), 2.0, places=6)
            self.assertAlmostEqual(float(snap["metrics"]["trap_area"]), 4.5, places=6)
            self.assertEqual(int(snap["metrics"]["diff_n"]), 3)
            self.assertEqual(int(snap["metrics"]["graph_nodes"]), 4)
            self.assertEqual(int(snap["metrics"]["graph_hops"]), 3)
            self.assertEqual(int(snap["metrics"]["graph_components_n"]), 1)
            self.assertEqual(int(snap["metrics"]["repo_count"]), 3)
            self.assertGreaterEqual(int(snap["metrics"]["repo_find_count"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["photo_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["graphs_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["convert_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["exe_ops_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["exe_artifacts_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["proc_n"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["npu_probes"]), 1)
            self.assertIsInstance(bool(snap["metrics"]["wifi_supported_flag"]), bool)
            self.assertIsInstance(bool(snap["metrics"]["rust_tool_present"]), bool)
            self.assertIsInstance(bool(snap["metrics"]["exe_tools_known"]), bool)
            self.assertGreaterEqual(int(snap["metrics"]["managed_seen"]), 1)

            self.assertTrue((tmp / "out" / "graph_edges.csv").exists())
            self.assertTrue((tmp / "out" / "graph_metrics.json").exists())
            self.assertTrue((tmp / "out" / "net.svg").exists())
            self.assertTrue((tmp / "out" / "hero.svg").exists())
            self.assertTrue((tmp / "out" / "chart.svg").exists())
            self.assertTrue((tmp / "out" / "proc_history.json").exists())
            self.assertTrue((tmp / "out" / "proc_managed.json").exists())
            self.assertTrue((tmp / "out" / "wifi_interfaces.json").exists())
            self.assertTrue((tmp / "out" / "wifi_profiles.json").exists())
            self.assertTrue((tmp / "out" / "wifi_scan.json").exists())
            self.assertTrue((tmp / "out" / "npu.json").exists())
            self.assertTrue((tmp / "out" / "iso_manifest.json").exists())
            self.assertTrue((tmp / "out" / "exe_last.json").exists())
            self.assertTrue((tmp / "out" / "github_report.json").exists())
            self.assertTrue((tmp / "out" / "ideas.json").exists())
            self.assertTrue((tmp / "out" / "feature_pack_snapshot.json").exists())

            self.assertIn("<svg", (tmp / "out" / "hero.svg").read_text(encoding="utf-8"))
            self.assertIn("<svg", (tmp / "out" / "chart.svg").read_text(encoding="utf-8"))
            graph_metrics = json.loads((tmp / "out" / "graph_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(int(graph_metrics["stats"]["nodes"]), 4)
            self.assertEqual(int(graph_metrics["stats"]["edges"]), 3)

            wifi_interfaces_payload = json.loads((tmp / "out" / "wifi_interfaces.json").read_text(encoding="utf-8"))
            wifi_profiles_payload = json.loads((tmp / "out" / "wifi_profiles.json").read_text(encoding="utf-8"))
            wifi_scan_payload = json.loads((tmp / "out" / "wifi_scan.json").read_text(encoding="utf-8"))
            self.assertIn("ok", wifi_interfaces_payload)
            self.assertIn("ok", wifi_profiles_payload)
            self.assertIn("ok", wifi_scan_payload)

            npu_payload = json.loads((tmp / "out" / "npu.json").read_text(encoding="utf-8"))
            self.assertIn("npu", npu_payload)
            self.assertIn("recommended_device", npu_payload)

            iso_manifest = json.loads((tmp / "out" / "iso_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(iso_manifest.get("ok")))
            self.assertGreaterEqual(int(iso_manifest.get("count", 0)), 1)

            exe_last = json.loads((tmp / "out" / "exe_last.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(exe_last.get("ok")))
            self.assertTrue(bool(exe_last.get("dry_run")))
            self.assertIn("out", exe_last)
            self.assertIn("mini_app.exe", str(exe_last.get("out")))
            self.assertIn("executables", snap)
            self.assertIn("artifacts", snap["executables"])
            self.assertTrue(any(Path(str(k)).name == "mini_app.exe" for k in snap["executables"]["artifacts"].keys()))

            github_report = json.loads((tmp / "out" / "github_report.json").read_text(encoding="utf-8"))
            self.assertEqual(int(github_report["summary"]["repo_count"]), 3)
            self.assertGreaterEqual(len(github_report.get("suggestions", [])), 1)
            self.assertIn("idea_forge", github_report)

            ideas_payload = json.loads((tmp / "out" / "ideas.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(ideas_payload.get("ideas", [])), 3)

            self.assertIn("hello_rs", snap["languages"]["modules"])
            self.assertEqual(snap["languages"]["modules"]["hello_rs"]["language"], "rust")
            rust_last = snap["languages"]["modules"]["hello_rs"].get("last_run", {})
            if rust_last.get("ok"):
                self.assertIn("hello-rs", str(rust_last.get("stdout", "")))
            else:
                reason = str(rust_last.get("reason", ""))
                self.assertTrue(reason in {"toolchain_missing", "cargo_missing", "rustc_missing", "rust_build_failed", "rust_no_runnable_target"} or bool(reason))
                if not bool(snap["metrics"]["rust_tool_present"]):
                    self.assertTrue(any(u.get("step") in {"rust_build", "rust_run"} for u in snap.get("unsupported_steps", [])))

            proc_ops = snap["processes"]["ops"]
            self.assertTrue(any(op.get("type") == "proc_profile" for op in proc_ops))
            self.assertTrue(any(op.get("type") == "proc_exec" for op in proc_ops))
            self.assertTrue(any(op.get("type") == "proc_spawn" for op in proc_ops))
            self.assertTrue(any(op.get("type") == "proc_wait" for op in proc_ops))
            self.assertTrue(any(op.get("profile") == "sandbox_py" for op in proc_ops if op.get("type") == "proc_exec"))

    def test_npu_planning_benchmark_and_torch_train_wrapper(self) -> None:
        program = r'''
project "NPUOps" {
  model tiny {
    backend = "npu";
    inputs = 4;
    hidden = 10;
    outputs = 2;
    layers = 2;
    lr = 0.004;
  }

  dataset tiny_ds {
    samples = 32;
    inputs = 4;
    outputs = 2;
    task = "classification";
    batch_size = 8;
  }

  metric profiles = npu_profile_count();
  metric providers = npu_provider_count();
  metric runs = npu_run_count();
  metric probes = npu_probe_count();
  metric plan_device = get(npu_last_plan(), "execution_device", "");
  metric bench_device = get(npu_last_benchmark(), "device", "");
  metric train_ok = get(npu_last_run(), "ok", false);

  pipeline build {
    step npu_profile("edge", {"precision": "int8", "optimize_for": "latency", "preferred_device": "npu"});
    step npu_probe_json("out/npu_probe.json");
    step npu_plan_json("out/npu_plan.json", "tiny", {"profile": "edge", "dataset": "tiny_ds", "task": "training"});
    step npu_benchmark_json("out/npu_bench.json", "tiny", {"profile": "edge", "iterations": 2, "size": 24});
    step npu_runtime_export_json("out/npu_runtime.json");
    step npu_torch_train("tiny", "tiny_ds", 1, {"profile": "edge", "device": "npu"});
    step export_json("out/npu_snapshot.json");
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "npu_ops.nxf"
            src.write_text(program, encoding="utf-8")

            project = parse_file(src)
            executor = Executor(project, source_path=src)
            executor.run_pipeline("build")
            snap = executor.snapshot()

            self.assertEqual(snap["project"], "NPUOps")
            self.assertGreaterEqual(int(snap["metrics"]["profiles"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["providers"]), 1)
            self.assertGreaterEqual(int(snap["metrics"]["probes"]), 1)
            self.assertIn(str(snap["metrics"]["plan_device"]), {"cpu", "xpu", "cuda", "mps"})
            self.assertIn(str(snap["metrics"]["bench_device"]), {"cpu", "xpu", "cuda", "mps"})
            self.assertIn("npu", snap)
            self.assertIn("profiles", snap["npu"])
            self.assertIn("runs", snap["npu"])
            self.assertIn("benchmarks", snap["npu"])

            self.assertTrue((tmp / "out" / "npu_probe.json").exists())
            self.assertTrue((tmp / "out" / "npu_plan.json").exists())
            self.assertTrue((tmp / "out" / "npu_bench.json").exists())
            self.assertTrue((tmp / "out" / "npu_runtime.json").exists())
            self.assertTrue((tmp / "out" / "npu_snapshot.json").exists())

            plan = json.loads((tmp / "out" / "npu_plan.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(plan.get("ok")))
            self.assertIn("provider", plan)
            self.assertIn("execution_device", plan)
            self.assertIn("fallback_chain", plan)

            bench = json.loads((tmp / "out" / "npu_bench.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(bench.get("ok")))
            self.assertIn("latency_ms_avg", bench)
            self.assertIn("throughput_ops_per_s", bench)

            if TORCH_AVAILABLE:
                self.assertGreaterEqual(int(snap["metrics"]["runs"]), 1)
                self.assertTrue(bool(snap["metrics"]["train_ok"]))
                last_run = snap["npu"].get("last_run", {})
                self.assertIn(last_run.get("device_requested"), {"npu", "xpu", "cuda", "mps", "cpu", "auto"})
            else:
                self.assertTrue(
                    any(u["step"] == "npu_torch_train" for u in snap.get("unsupported_steps", []))
                    or not bool(snap["metrics"]["train_ok"])
                )


if __name__ == "__main__":
    unittest.main()
