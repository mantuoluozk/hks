"""Real HTTP functional checks. Fixture assertions are not benchmark accuracy."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw, ImageFont
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "kimi-vl-a3b-instruct"
URL = "http://127.0.0.1:30004"


def image_part(data):
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def fixture(color=None, text=None):
    image = Image.new("RGB", (512, 256), color or "white")
    if text:
        ImageDraw.Draw(image).text((40, 80), text, fill="black", font=ImageFont.load_default(size=48))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def request(session, question, images=(), stream=False, max_tokens=128):
    content = [image_part(data) for data in images] + [{"type": "text", "text": question}]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": max_tokens,
               "stream": stream}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    start = time.monotonic()
    result = {"request_payload": payload, "question": question, "images_sha256": [hashlib.sha256(data).hexdigest() for data in images],
              "generation": {k: v for k, v in payload.items() if k not in {"messages", "model"}}}
    response = session.post(URL+"/v1/chat/completions", json=payload, stream=stream, timeout=(10, 600))
    response.raise_for_status()
    if stream:
        chunks, texts, first, done, finish, usage = [], [], None, False, None, None
        for line in response.iter_lines(chunk_size=1):
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                token = choice.get("delta", {}).get("content") or ""
                if token and first is None:
                    first = time.monotonic()-start
                texts.append(token)
                finish = choice.get("finish_reason") or finish
        result.update(output="".join(texts), finish_reason=finish, usage=usage,
                      chunks=chunks, done_marker=done, ttft_seconds=first)
        assert done, "Stream did not end with [DONE]"
    else:
        raw = response.json()
        choice = raw["choices"][0]
        result.update(output=choice["message"].get("content") or "",
                      finish_reason=choice.get("finish_reason"), usage=raw.get("usage"), raw_response=raw)
    response.close()
    result["seconds"] = time.monotonic()-start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    deadline = time.monotonic()+1200
    while True:
        try:
            response = session.get(URL+"/health", timeout=3)
            response.raise_for_status()
            break
        except requests.RequestException:
            if time.monotonic() > deadline:
                raise RuntimeError("Server did not become ready within 20 minutes")
            time.sleep(3)
    models = session.get(URL+"/v1/models", timeout=10).json()
    assert any(row["id"] == MODEL for row in models["data"])
    (args.output/"models.json").write_text(json.dumps(models, indent=2)+"\n")
    red, blue, ocr = fixture("red"), fixture("blue"), fixture(text="HKS2026")
    for name, data in [("red",red),("blue",blue),("ocr",ocr)]:
        (args.output/(name+".png")).write_bytes(data)
    demo = (ROOT/"data/examples/demo.png").read_bytes()
    cases = [
        ("text", "What is 2 + 2? Answer with the number only.", (), lambda t: t.strip() == "4"),
        ("red", "What color fills this image? Answer with one English word.", (red,), lambda t: "red" in t.lower()),
        ("blue-zh", "这张图片主要是什么颜色？请只回答颜色。", (blue,), lambda t: "蓝" in t or "blue" in t.lower()),
        ("ocr", "Read the text in this image. Output only the text.", (ocr,), lambda t: "HKS2026" in re.sub(r"\s+", "", t)),
        ("demo", "What is the dome building in the picture?", (demo,), lambda t: bool(t.strip())),
        ("two-images", "Name the color of the first image and then the second image. Answer with two English color words in order.", (red, blue),
         lambda t: bool(re.search(r"red.*blue", t.lower(), re.S))),
    ]
    results = []
    for key, question, images, predicate in cases:
        try:
            row = request(session, question, images)
            row.update(id=key, passed=bool(predicate(row["output"])) and row["finish_reason"] == "stop")
        except Exception as error:
            row = {"id": key, "passed": False, "error": str(error)}
        results.append(row)
        (args.output/(key+".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({k: row.get(k) for k in ["id", "passed", "output", "error", "seconds"]}, ensure_ascii=False), flush=True)
    reference = next(row for row in results if row["id"] == "red")
    for key, stream in [("repeat", False), ("stream", True)]:
        try:
            row = request(session, cases[1][1], (red,), stream=stream)
            row.update(id=key, passed=reference["passed"] and
                       row["output"] == reference["output"] and row["finish_reason"] == "stop")
        except Exception as error:
            row = {"id": key, "passed": False, "error": str(error)}
        results.append(row)
        (args.output/(key+".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({k: row.get(k) for k in ["id", "passed", "output", "error", "seconds"]}, ensure_ascii=False), flush=True)
    summary = {"functional_passed": sum(row["passed"] for row in results), "functional_total": len(results),
               "all_passed": all(row["passed"] for row in results),
               "model": MODEL, "official_accuracy_pass": None,
               "scope": "Synthetic/official-demo HTTP functional checks; not public benchmark accuracy",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
