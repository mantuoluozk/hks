"""Run the fixed official Qwen-Image T03/T04 conditions serially over SGLang."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import io
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import zipfile

import requests
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ROOT / "data/official-t03t04/conditions/qwen_image.jsonl"
FEEDBACK_FIELDS = ["condition_id", "model", "status", "output_path", "error", "parameter_changes"]
BASE_URL = "http://127.0.0.1:30003"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def conditions():
    rows = [json.loads(line) for line in CONDITIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 180
    assert len({row["id"] for row in rows}) == 180
    assert len({row["condition_id"] for row in rows}) == 60
    assert {row["seed"] for row in rows} == {42, 2026, 1234}
    for row in rows:
        assert row["model"] == "qwen_image"
        assert row["params"] == {
            "width": 2048,
            "height": 2048,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "schedule": "checkpoint",
            "use_kv_cache": True,
            "dtype": "bf16",
        }
        assert row["expected"] == {"kind": "image", "width": 2048, "height": 2048}
    return rows


def read_trace(path):
    files = sorted(path.glob("requests-*.jsonl"))
    rows = []
    for file in files:
        rows.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()
    rows = conditions()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = (args.job_dir or ROOT / "results" / ("official-t03t04-" + stamp)).resolve()
    assert job.is_relative_to((ROOT / "results").resolve())
    job.mkdir(parents=True, exist_ok=False)
    outputs = job / "outputs"
    outputs.mkdir()
    trace = job / "trace"
    trace.mkdir()
    log_path = job / "server.log"
    state = {"status": "preflight", "expected": len(rows), "processed": 0, "successful": 0, "failed": 0,
             "job": job.relative_to(ROOT).as_posix(), "service_pid": None}

    def save_state():
        atomic_json(job / "state.json", state)

    save_state()
    env = os.environ.copy()
    env["HKS_REQUIRE_BAKED_RUNTIME"] = "1"
    env["RUN_RESULTS_DIR"] = str(job / "preflight")
    (job / "preflight").mkdir()
    with (job / "preflight.log").open("w", encoding="utf-8") as log:
        subprocess.run(["bash", "env_check.sh"], cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    preflight = json.loads((job / "preflight/summary.json").read_text(encoding="utf-8"))
    assert preflight["baked_runtime"] is True
    assert preflight["runtime_root"] == "/opt/qwen-image21"
    launcher = preflight.get("launcher_integrity") or {}
    assert launcher.get("image_sha256") == launcher.get("executed_source_sha256")
    env_info = {
        "model": "Qwen/Qwen-Image-2.1-Turbo",
        "device": preflight["device"],
        "framework": f"SGLang {metadata.version('sglang')}",
        "torch": preflight["packages"]["torch"]["version"],
        "diffusers": preflight["packages"]["diffusers"]["version"],
        "transformers": preflight["packages"]["transformers"]["version"],
        "tokenizers": preflight["packages"]["tokenizers"]["version"],
        "image_runtime": preflight["runtime_root"],
        "precision": "bfloat16",
        "official_conditions": 180,
        "resolution": "2048x2048",
        "seeds": [42, 2026, 1234],
        "steps": 8,
        "guidance_scale": 1.0,
        "use_kv_cache": True,
        "kv_cache_mapping": "SGLang diffusers_kwargs -> QwenImage21Pipeline.use_kv_cache; checkpoint transformer causal_condition=True",
        "schedule": "checkpoint sample_sigmas",
    }
    model_path = Path(env.get("MODEL_PATH", "/root/private_data/SothisAI/model/Aihub/Qwen-Image-2.1-Turbo/main/Qwen-Image-2.1-Turbo"))
    transformer_config = json.loads((model_path / "transformer/config.json").read_text(encoding="utf-8"))
    assert transformer_config.get("causal_condition") is True
    atomic_json(job / "environment.json", env_info)
    (job / "环境说明.txt").write_text(
        "Qwen-Image-2.1-Turbo T03/T04 固定条件实测环境说明\n"
        f"设备：{env_info['device']['name']} / {env_info['device']['gcn_arch']}，1卡\n"
        f"框架：{env_info['framework']}\n"
        f"PyTorch：{env_info['torch']}\n"
        f"Diffusers：{env_info['diffusers']}；Transformers：{env_info['transformers']}；tokenizers：{env_info['tokenizers']}\n"
        "精度：BF16；输出格式：PNG；尺寸：2048×2048\n"
        "条件：60条GenEval prompt，各使用seed 42、2026、1234，共180条\n"
        "采样：8步，guidance_scale=1.0，checkpoint sample_sigmas，use_kv_cache=True\n",
        encoding="utf-8",
    )

    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 30003)) == 0:
            raise RuntimeError("Port 30003 is already occupied; no existing service was stopped")

    env.update({"HKS_TRACE_DIR": str(trace), "RUN_RESULTS_DIR": str(job / "service"),
                "HOST": "127.0.0.1", "PORT": "30003"})
    service = None
    session = requests.Session()
    session.trust_env = False
    try:
        state["status"] = "starting_service"
        save_state()
        with log_path.open("w", encoding="utf-8") as log:
            service = subprocess.Popen(["bash", "run.sh"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        state["service_pid"] = service.pid
        save_state()
        for _ in range(180):
            if service.poll() is not None:
                raise RuntimeError("SGLang exited during startup; see server.log")
            try:
                response = session.get(BASE_URL + "/health", timeout=3)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("SGLang readiness exceeded 15 minutes")

        models_response = session.get(BASE_URL + "/v1/models", timeout=10)
        models_response.raise_for_status()
        model = models_response.json()["data"][0]["id"]
        atomic_json(job / "models-response.json", models_response.json())
        state["status"] = "generating"
        save_state()

        feedback_path = job / "反馈表.csv"
        with feedback_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FEEDBACK_FIELDS)
            writer.writeheader()
            for index, case in enumerate(rows, start=1):
                output = outputs / (case["id"] + ".png")
                before = len(read_trace(trace))
                payload = {
                    "model": model,
                    "prompt": case["prompt"],
                    "size": "2048x2048",
                    "n": 1,
                    "response_format": "b64_json",
                    "seed": int(case["seed"]),
                    "generator_device": "cpu",
                    "num_inference_steps": 8,
                    "true_cfg_scale": 1.0,
                    "output_format": "png",
                    "diffusers_kwargs": {"use_kv_cache": True},
                }
                error = ""
                elapsed = None
                try:
                    start = time.perf_counter()
                    response = session.post(BASE_URL + "/v1/images/generations", json=payload, timeout=1800)
                    elapsed = time.perf_counter() - start
                    response.raise_for_status()
                    body = response.json()
                    image_bytes = base64.b64decode(body["data"][0]["b64_json"], validate=True)
                    image = Image.open(io.BytesIO(image_bytes))
                    image.load()
                    assert image.size == (2048, 2048), image.size
                    assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
                    trace_rows = read_trace(trace)
                    assert len(trace_rows) == before + 1, (before, len(trace_rows))
                    actual = trace_rows[-1]
                    assert actual["width"] == 2048 and actual["height"] == 2048
                    assert actual["seed"] == int(case["seed"])
                    assert actual["generator_devices"] == ["cpu"]
                    assert actual["true_cfg_scale"] == 1.0
                    assert actual["requested_num_inference_steps"] == 8
                    output.write_bytes(image_bytes)
                    result = {"condition_id": case["id"], "model": case["model"], "status": "成功",
                              "output_path": "outputs/" + output.name, "error": "", "parameter_changes": "无"}
                    state["successful"] += 1
                    record = {"condition_id": case["id"], "seed": case["seed"], "status": "成功",
                              "output_path": "outputs/" + output.name, "elapsed_seconds": elapsed,
                              "size": list(image.size), "generator_device": "cpu", "num_inference_steps": 8,
                              "guidance_scale": 1.0, "use_kv_cache": True, "checkpoint_schedule": "8 sample_sigmas"}
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:1200]
                    result = {"condition_id": case["id"], "model": case["model"], "status": "失败",
                              "output_path": "outputs/" + output.name if output.exists() else "",
                              "error": error, "parameter_changes": ""}
                    state["failed"] += 1
                    record = {"condition_id": case["id"], "seed": case["seed"], "status": "失败", "error": error}
                    if service.poll() is not None:
                        writer.writerow(result)
                        csv_file.flush()
                        state["processed"] += 1
                        atomic_json(job / "state.json", state)
                        raise RuntimeError("SGLang exited during official generation") from exc
                writer.writerow(result)
                csv_file.flush()
                with (job / "requests.jsonl").open("a", encoding="utf-8") as records:
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")
                state["processed"] = index
                state["last_condition_id"] = case["id"]
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                save_state()
                print(json.dumps({"processed": index, "expected": len(rows), "condition_id": case["id"],
                                  "status": result["status"], "elapsed_seconds": elapsed, "error": error},
                                 ensure_ascii=False), flush=True)
                if index == 1 and result["status"] != "成功":
                    state["status"] = "parameter_gate_failed"
                    state["error"] = "First official condition failed; stopped before continuing the full set."
                    save_state()
                    break

        if state["status"] != "parameter_gate_failed":
            state["status"] = "complete" if state["processed"] == len(rows) and state["failed"] == 0 else "completed_with_failures"
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()[-5000:]
        raise
    finally:
        save_state()
        if service is not None and service.poll() is None:
            try:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid, signal.SIGKILL)
                service.wait(timeout=10)
        if state["status"] in {"complete", "completed_with_failures"}:
            summary = {"expected": len(rows), "processed": state["processed"], "successful": state["successful"],
                       "failed": state["failed"], "status": state["status"], "task": "Qwen-Image-2.1-Turbo T03/T04 official conditions",
                       "datasets": {"GenEval": 60}, "seeds": [42, 2026, 1234], "size": "2048x2048",
                       "num_inference_steps": 8, "guidance_scale": 1.0, "use_kv_cache": True,
                       "schedule": "checkpoint sample_sigmas", "dtype": "bf16"}
            atomic_json(job / "summary.json", summary)
            if state["status"] == "complete":
                artifact_dir = ROOT / "artifacts" / job.name
                artifact_dir.mkdir(parents=True, exist_ok=True)
                bundle = artifact_dir / "Qwen-Image-outputs-and-feedback.zip"
                with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
                    for path in sorted(job.rglob("*")):
                        if path.is_file() and path != bundle and path.name not in {"server.log", "preflight.log"} \
                                and "preflight" not in path.parts and "service" not in path.parts:
                            archive.write(path, path.relative_to(job).as_posix())
            atomic_json(ROOT / "results/official-t03t04-current.json", {"job": job.relative_to(ROOT).as_posix(),
                         "status": state["status"], "artifact": str((ROOT / "artifacts" / job.name).relative_to(ROOT))})
            save_state()
        print(json.dumps({"job": str(job), "status": state["status"], "processed": state["processed"],
                          "successful": state["successful"], "failed": state["failed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
