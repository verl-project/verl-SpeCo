# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Resolve pinned HF snapshots and stage model files before concurrent NPU loading."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time


def complete(path):
    if not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / name).is_file():
            shards = set(json.loads((path / name).read_text())["weight_map"].values())
            if any(
                Path(shard).is_absolute() or ".." in Path(shard).parts
                for shard in shards
            ):
                raise ValueError("Model index contains a path outside its snapshot")
            return bool(shards) and all(
                (path / shard).is_file() and (path / shard).stat().st_size
                for shard in shards
            )
    return any(
        file.is_file() and file.stat().st_size
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for file in path.glob(pattern)
    )


def _download_snapshot(download, **options):
    for attempt in range(3):
        try:
            return download(**options)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 2 or status not in {429, 500, 502, 503, 504}:
                raise
            delay = 15 * (attempt + 1)
            print(
                f"Model download returned HTTP {status}; retrying in {delay}s "
                f"({attempt + 2}/3): {options['repo_id']}@{options['revision']}",
                flush=True,
            )
            time.sleep(delay)


def _fetch_snapshot(repo_id, revision, max_workers=2, force_download=False):
    from huggingface_hub import HfApi, hf_hub_download

    # The mirror's recursive tree API can return persistent 503s. Model metadata
    # already lists the files; pin every file request to its resolved commit.
    info = HfApi().model_info(repo_id, revision=revision)
    if not info.sha or (
        re.fullmatch(r"[a-f0-9]{40}", revision or "") and info.sha != revision
    ):
        raise ValueError(
            f"Model metadata did not resolve the requested revision: {repo_id}@{revision}"
        )
    filenames = [PurePosixPath(file.rfilename) for file in info.siblings or []]
    if not filenames or any(
        path.is_absolute() or ".." in path.parts or not path.parts for path in filenames
    ):
        raise ValueError(
            f"Model metadata contains an invalid file list: {repo_id}@{info.sha}"
        )
    print(f"Downloading {len(filenames)} model files: {repo_id}@{info.sha}", flush=True)

    def fetch(filename):
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename.as_posix(),
            revision=info.sha,
            force_download=force_download,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        paths = list(pool.map(fetch, filenames))
    return Path(paths[0]).parents[len(filenames[0].parts) - 1]


def prepare(source, destination, *, download=None):
    source = Path(source)
    if not complete(source):
        snapshot = re.search(
            r"models--([^/]+)[/\\]snapshots[/\\]([a-f0-9]{40})$", source.as_posix()
        )
        repo_id = snapshot[1].replace("--", "/") if snapshot else str(source)
        if snapshot is None and (
            source.is_absolute() or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id)
        ):
            raise ValueError(
                f"Missing or incomplete local model: {source}; set the model path to a complete checkpoint"
            )
        if download is None:
            download = _fetch_snapshot
        source = Path(
            _download_snapshot(
                download,
                repo_id=repo_id,
                revision=snapshot[2] if snapshot else None,
                max_workers=2,
                force_download=source.exists(),
            )
        )
        if not complete(source):
            raise ValueError(f"Downloaded snapshot is incomplete: {source}")
    files = [path for path in source.rglob("*") if path.is_file()]
    required = sum(path.stat().st_size for path in files)
    free = shutil.disk_usage(destination.parent).free
    if free < required + 512 * 1024**2:
        raise ValueError(
            f"Insufficient model staging space: need {required} bytes plus 512 MiB, have {free}; set SPECO_NPU_MODEL_STAGE_DIR to local storage with enough space"
        )
    started = time.monotonic()
    shutil.copytree(source, destination, symlinks=False)
    if not complete(destination):
        raise ValueError(f"Staged checkpoint is incomplete: {destination}")
    print(
        f"Staged {source} -> {destination}: {required} bytes in {time.monotonic() - started:.2f}s",
        flush=True,
    )
    return destination


def main():
    algorithm = sys.argv[1].upper()
    if algorithm not in {"EAGLE3", "DFLASH", "DSPARK"}:
        raise ValueError("Unsupported NPU drafter")
    staging = Path(os.environ.get("SPECO_NPU_MODEL_STAGE_DIR", "/tmp/speco-npu-models"))
    staging.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=algorithm.lower() + "-", dir=staging))
    try:
        target = prepare(os.environ["SPECO_TARGET_MODEL"], root / "target")
        draft = prepare(os.environ[f"SPECO_{algorithm}_DRAFT_MODEL"], root / "draft")
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(
                f"target_model={target}\ndraft_model={draft}\nstaging_root={root}\n"
            )
    except BaseException:
        shutil.rmtree(root)
        raise


if __name__ == "__main__":
    main()
