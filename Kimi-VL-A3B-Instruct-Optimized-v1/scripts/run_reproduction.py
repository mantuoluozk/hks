"""Serialize performance and a fresh-server repeat after the accuracy run ends."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCNET = ROOT/"results/scnet"
FIRST = SCNET/"native-vllm-v3-20261028"
SECOND = SCNET/"fresh-vllm-20261028"


def run(command, log):
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def stop_owned(pid):
    cmd = Path(f"/proc/{pid}/cmdline")
    if not cmd.exists():
        return
    args = cmd.read_bytes()
    assert b"vllm.entrypoints.openai.api_server" in args and b"30004" in args, "Unexpected PID owner"
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        if not cmd.exists():
            return
        # A dead orphan may temporarily remain a zombie while being reaped.
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            return
        time.sleep(1)
    raise RuntimeError("Server did not exit cleanly; inspect before forcing shutdown")


def main():
    state_path = SCNET/"reproduction-state-20261028.json"
    def state(stage, **extra):
        state_path.write_text(json.dumps({"stage": stage, "controller_pid": os.getpid(), **extra}, indent=2)+"\n")
    accuracy = SCNET/"ocrbench-full-20261028/summary.json"
    state("waiting_for_accuracy")
    deadline = time.monotonic()+4*3600
    controller = int((SCNET/"ocrbench-controller-20261028.pid").read_text())
    while not accuracy.exists():
        if not Path(f"/proc/{controller}/cmdline").exists():
            raise RuntimeError("Accuracy process ended before writing summary")
        if time.monotonic() > deadline:
            raise RuntimeError("Accuracy wait deadline exceeded")
        time.sleep(5)
    score = json.loads(accuracy.read_text())
    assert score["samples"] == 1000 and score["errors"] == 0
    state("first_performance")
    smoke = FIRST/"functional/summary.json"
    run(["bash", "run-scnet.sh", "python", "scripts/bench_http.py", "--output", str(FIRST/"performance"),
         "--smoke-summary", str(smoke)], FIRST/"performance.log")
    stop_owned(int((FIRST/"server.pid").read_text()))
    SECOND.mkdir(exist_ok=False)
    with (SECOND/"server.log").open("w") as log:
        process = subprocess.Popen(["bash", "serve-scnet.sh"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (SECOND/"server.pid").write_text(str(process.pid)+"\n")
    state("fresh_server_functional", server_pid=process.pid)
    try:
        run(["bash", "run-scnet.sh", "python", "scripts/smoke_http.py", "--output", str(SECOND/"functional")],
            SECOND/"functional.log")
        state("fresh_server_performance", server_pid=process.pid)
        run(["bash", "run-scnet.sh", "python", "scripts/bench_http.py", "--output", str(SECOND/"performance"),
             "--smoke-summary", str(SECOND/"functional/summary.json")], SECOND/"performance.log")
    finally:
        stop_owned(process.pid)
    a = json.loads((FIRST/"performance/summary.json").read_text())
    b = json.loads((SECOND/"performance/summary.json").read_text())
    assert a["question"] == b["question"] and a["image_sha256"] == b["image_sha256"]
    assert a["script_sha256"] == b["script_sha256"] and a["http_client_sha256"] == b["http_client_sha256"]
    comparisons = []
    for x, y in zip(a["workloads"], b["workloads"]):
        assert (x["concurrency"], x["max_tokens"], x["requests"]) == (y["concurrency"], y["max_tokens"], y["requests"])
        delta = (y["output_tokens_per_second"]/x["output_tokens_per_second"]-1)*100
        comparisons.append({"concurrency": x["concurrency"], "max_tokens": x["max_tokens"],
                            "first_output_tokens_per_second": x["output_tokens_per_second"],
                            "repeat_output_tokens_per_second": y["output_tokens_per_second"],
                            "deviation_percent": delta})
    result = {"comparisons": comparisons, "maximum_absolute_deviation_percent": max(abs(x["deviation_percent"]) for x in comparisons),
              "scope": "Two fresh server processes on the same SCNet Notebook and image; not a new-instance test",
              "official_accuracy_pass": None}
    result["performance_reproduction_passed"] = result["maximum_absolute_deviation_percent"] <= 20
    (SCNET/"reproduction-summary-20261028.json").write_text(json.dumps(result, indent=2)+"\n")
    state("complete", **result)
    run(["python", "/root/private_data/hks/platform/scnet-notebook/backup_shared.py"], SCNET/"reproduction-backup-20261028.log")


if __name__ == "__main__":
    main()
