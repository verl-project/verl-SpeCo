# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
import importlib.util
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("prefill_benchmark", ROOT / "tools/benchmark_prefill_clients.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_only_deletes_returned_file_in_explicit_root(tmp_path):
    module = _module()
    root = tmp_path / "outputs"
    root.mkdir()
    allowed = root / "hidden"
    allowed.write_bytes(b"hidden")
    unrelated = root / "unrelated"
    unrelated.write_bytes(b"keep")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="outside"):
        module.release_file(outside, root, 1)
    module.release_file(allowed, root, 1)
    assert not allowed.exists()
    assert unrelated.exists() and outside.exists()


@pytest.mark.parametrize("client", ["speco", "speculators"])
def test_cli_export_and_native_client_against_mock_http(tmp_path, client):
    pytest.importorskip("openai")
    if client == "speco":
        pytest.importorskip("torch")
    speculators_root = Path(os.environ.get("SPECO_SPECULATORS_ROOT", ""))
    if client == "speculators":
        # Actual speculators client imports POSIX fcntl; it runs in Linux containers.
        pytest.importorskip("fcntl")
        if not (speculators_root / "src/speculators").is_dir():
            pytest.skip("Set SPECO_SPECULATORS_ROOT to test the native reference client")
    hidden_root = tmp_path / "hidden"
    hidden_root.mkdir()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            path = hidden_root / f"{uuid4().hex}.safetensors"
            path.write_bytes(b"not loaded by the request-only benchmark")
            response = json.dumps({
                "id": "cmpl-test", "object": "text_completion", "created": 0, "model": "test",
                "choices": [{"text": "", "index": 0, "finish_reason": "length",
                             "prompt_token_ids": body["prompt"], "token_ids": [1]}],
                "kv_transfer_params": {"hidden_states_path": str(path)},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"input_ids": [1, 2, index + 3]}) + "\n"
                              for index in range(8)), encoding="utf-8")
    exported = tmp_path / "requests.jsonl"
    subprocess.run([sys.executable, str(ROOT / "tools/export_prefill_requests.py"),
                    "--input", str(source), "--output", str(exported), "--samples", "8"],
                   check=True, capture_output=True, text=True, timeout=30)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result_path = tmp_path / "result.json"
    command = [sys.executable, str(ROOT / "tools/benchmark_prefill_clients.py"),
               "--client", client, "--input", str(exported), "--endpoint",
               f"http://127.0.0.1:{server.server_port}/v1", "--model", "test",
               "--cleanup-root", str(hidden_root), "--concurrency", "2", "--warmup", "2",
               "--samples", "6", "--client-label", "test-client", "--server-label", "mock",
               "--output", str(result_path)]
    if client == "speculators":
        command += ["--speculators-root", str(speculators_root)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    assert summary["successes"] == 6 and summary["failures"] == 0
    assert summary["prompt_tokens"] == 18
    assert len(received) == 8
    assert all(body["max_tokens"] == 1 and body["return_token_ids"] for body in received)
    assert not list(hidden_root.iterdir())
