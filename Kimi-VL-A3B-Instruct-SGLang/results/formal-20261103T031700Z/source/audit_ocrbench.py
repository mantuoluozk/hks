"""Re-score all saved predictions using the actual upstream scoring loop AST."""
import ast
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);args=parser.parse_args()
run=args.run
reference = ROOT/"sources/ocrbench_example.py"
tree = ast.parse(reference.read_text())
loops = [node for node in ast.walk(tree) if isinstance(node, ast.For) and node.body
         and isinstance(node.body[0], ast.Assign)
         and isinstance(node.body[0].targets[0], ast.Name)
         and node.body[0].targets[0].id == "data_type"]
assert len(loops) == 1
rows = pq.read_table(ROOT/"data/ocrbench/test.parquet").to_pylist()
predictions = [json.loads(line) for line in (run/"predictions.jsonl").read_text().splitlines()]
scored = [json.loads(line) for line in (run/"scored.jsonl").read_text().splitlines()]
assert len(rows) == len(predictions) == len(scored) == 1000
data = []
for i, (row, prediction) in enumerate(zip(rows, predictions)):
    key = str(row.get("id", f"ocrbench_test_{i:04d}"))
    assert key == prediction["id"] == scored[i]["id"]
    assert not prediction["error"]
    data.append({"type": row["question_type"], "dataset_name": row["dataset"],
                 "answers": row["answer"], "predict": prediction["output"]})
namespace = {"data": data}
exec(compile(ast.Module(body=[loops[0]], type_ignores=[]), str(reference), "exec"), namespace)
mismatch = [predictions[i]["id"] for i, row in enumerate(data)
            if bool(row.get("result", 0)) != scored[i]["correct"]]
summary = json.loads((run/"summary.json").read_text())
result = {"samples": 1000, "upstream_correct": sum(row.get("result", 0) for row in data),
          "recorded_correct": summary["correct"], "mismatched_ids": mismatch,
          "actual_upstream_loop_lines": [loops[0].lineno, loops[0].end_lineno],
          "upstream_source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
          "predictions_sha256": hashlib.sha256((run/"predictions.jsonl").read_bytes()).hexdigest(),
          "all_matched": not mismatch}
assert result["upstream_correct"] == result["recorded_correct"] and not mismatch
(run/"upstream-scoring-audit.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
