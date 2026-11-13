# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.standalone_resume import (
    load_standalone_resume,
    save_standalone_resume,
)


def test_resume_binds_persistent_lease_snapshot_without_granting_training_credit(tmp_path, monkeypatch):
    import hashlib
    from verl_speco.standalone_ray_runtime import _ReferenceLeaseJournal
    from verl_speco.trainer.standalone_resume import validate_reference_lease_snapshot

    source = tmp_path / "train.jsonl"
    source.write_text('{"prompt":"q","response":"a"}\n', encoding="utf-8")
    journal = _ReferenceLeaseJournal(tmp_path / "leases.sqlite", "run", "incarnation")
    journal.admit([{"key": "future-sample", "tag": {"sequence_no": 1}}])
    identity = {"receipt": "receipt", "keys": ["future-sample"], "owner_count": 2, "native_sources": {}}
    journal.transition(0, "DISPATCH_INTENT", identity)
    journal.transition(0, "LEASED", identity)
    journal.acknowledge(0, 0)
    snapshot = validate_reference_lease_snapshot(journal.snapshot())
    journal.close()
    checkpoint = tmp_path / "draft_step_1"
    checkpoint.mkdir()
    snapshot_file = checkpoint / "reference_leases.json"
    snapshot_file.write_text(json.dumps(snapshot), encoding="utf-8")
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source,
                           lease_snapshot_sha256=snapshot["sha256"])
    consumed, metadata = load_standalone_resume(checkpoint, input_path=source)
    assert consumed == {0} and metadata["reference_lease_snapshot_sha256"] == snapshot["sha256"]
    # A file growing after stat must still be read with a hard allocation bound.
    from io import BytesIO
    from verl_speco.trainer.standalone_resume import _load_reference_lease_snapshot

    class GrowingSnapshot(BytesIO):
        def read(self, size=-1):
            assert size == 64 * 1024**2 + 1
            return super().read(size)

    original_open = Path.open
    def bounded_open(path, *args, **kwargs):
        if path == snapshot_file and args == ("rb",):
            return GrowingSnapshot(json.dumps(snapshot).encode())
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", bounded_open)
        assert _load_reference_lease_snapshot(checkpoint, snapshot["sha256"]) == snapshot
    snapshot_file.unlink()
    with pytest.raises(ValueError, match="snapshot is missing"):
        load_standalone_resume(checkpoint)
    malformed = dict(snapshot, leases=[[0, "MATERIALIZED", snapshot["leases"][0][2]]])
    malformed.pop("sha256")
    malformed["sha256"] = hashlib.sha256(json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="missing owner acknowledgements"):
        validate_reference_lease_snapshot(malformed)
    missing_payload = dict(snapshot, payloads=[])
    missing_payload.pop("sha256")
    missing_payload["sha256"] = hashlib.sha256(json.dumps(missing_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="payload dependencies"):
        validate_reference_lease_snapshot(missing_payload)
    snapshot_file.write_text(json.dumps(dict(snapshot, incarnation="changed")), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_standalone_resume(checkpoint)


def test_standalone_resume_round_trip_and_input_validation(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text('{"prompt":"q","response":"a"}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_7"

    save_standalone_resume(
        checkpoint_path,
        [5, 1, 5, 3],
        optimizer_step=7,
        input_path=input_path,
    )

    consumed, metadata = load_standalone_resume(
        checkpoint_path, input_path=input_path
    )
    assert consumed == {1, 3, 5}
    assert metadata is not None
    assert metadata["optimizer_step"] == 7
    assert metadata["consumed_count"] == 3

    input_path.write_text('{"prompt":"changed","response":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(checkpoint_path, input_path=input_path)


def test_missing_standalone_resume_is_not_a_resume_checkpoint(
    tmp_path: Path,
) -> None:
    consumed, metadata = load_standalone_resume(tmp_path / "pretrained")
    assert consumed == set()
    assert metadata is None


def test_resume_rejects_replaced_tensor_even_when_count_matches(tmp_path):
    import torch

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1, 2], optimizer_step=1, input_path=source)
    metadata_path = checkpoint / "standalone_resume.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    progress = checkpoint / metadata["consumed_sequence_file"]
    torch.save(torch.tensor([3, 4], dtype=torch.int64), progress)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_standalone_resume(checkpoint)
    # Existing schema-v1 checkpoints without a checksum remain readable.
    progress.replace(checkpoint / "consumed_sequence_nos.pt")
    metadata["schema_version"] = 1
    metadata["consumed_sequence_file"] = "consumed_sequence_nos.pt"
    metadata.pop("storage_format")
    del metadata["consumed_sequence_sha256"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_standalone_resume(checkpoint)[0] == {3, 4}


def test_resume_detects_same_size_input_replacement_with_restored_mtime(tmp_path):
    import os

    source = tmp_path / "input.jsonl"
    source.write_bytes(b'{"prompt":"old"}\n')
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source)
    original = source.stat()
    source.write_bytes(b'{"prompt":"new"}\n')
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(checkpoint, input_path=source)


def test_checkpoint_rejects_input_changed_after_training_started(tmp_path):
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"original\n")
    original_stat = source.stat()
    baseline = input_content_identity(build_input_fingerprint(source, include_content_hash=True))
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source,
                           expected_input_identity=baseline)
    metadata = (checkpoint / "standalone_resume.json").read_bytes()
    progress_path = checkpoint / json.loads(metadata)["consumed_sequence_file"]
    progress = progress_path.read_bytes()
    source.write_bytes(b"modified\n")  # Same size and restored mtime must still be detected.
    import os
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed since training began"):
        save_standalone_resume(checkpoint, [0, 1], optimizer_step=2, input_path=source,
                               expected_input_identity=baseline)
    assert (checkpoint / "standalone_resume.json").read_bytes() == metadata
    assert progress_path.read_bytes() == progress


@pytest.mark.parametrize("storage_format", ["tensor", "gzip"])
def test_interrupted_resume_publish_preserves_old_checkpoint(tmp_path, monkeypatch, storage_format):
    import os

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source,
                           storage_format=storage_format)
    metadata_path = checkpoint / "standalone_resume.json"
    old_metadata = metadata_path.read_bytes()
    old_progress = checkpoint / json.loads(old_metadata)["consumed_sequence_file"]
    original_replace = os.replace

    def fail_metadata_replace(src, dst):
        if Path(dst) == metadata_path:
            raise OSError("injected metadata publish failure")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="injected metadata"):
        save_standalone_resume(checkpoint, [1, 2], optimizer_step=2, input_path=source,
                               storage_format=storage_format)
    assert metadata_path.read_bytes() == old_metadata
    assert old_progress.exists()
    assert load_standalone_resume(checkpoint, input_path=source)[0] == {1}
    assert not list(checkpoint.glob("*.incomplete"))


@pytest.mark.parametrize("storage_format", ["tensor", "gzip"])
def test_resume_input_check_failure_preserves_previous_progress(tmp_path, monkeypatch, storage_format):
    import verl_speco.trainer.standalone_resume as resume

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source,
                           storage_format=storage_format)
    def fail_fingerprint(*args, **kwargs):
        raise OSError("injected fingerprint failure")
    monkeypatch.setattr(resume, "build_input_fingerprint", fail_fingerprint)
    with pytest.raises(OSError, match="injected fingerprint"):
        save_standalone_resume(checkpoint, [1, 2], optimizer_step=2, input_path=source,
                               storage_format=storage_format)
    assert load_standalone_resume(checkpoint)[0] == {1}


def test_legacy_gzip_resume_remains_readable(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1, 4], optimizer_step=1, input_path=source,
                           storage_format="gzip")
    metadata_path = checkpoint / "standalone_resume.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    (checkpoint / metadata["consumed_sequence_file"]).replace(checkpoint / "consumed_sequence_nos.i64.gz")
    metadata.update(schema_version=2, consumed_sequence_file="consumed_sequence_nos.i64.gz")
    metadata.pop("storage_format")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_standalone_resume(checkpoint, input_path=source)[0] == {1, 4}


@pytest.mark.parametrize("storage_format", ["tensor", "gzip"])
def test_resume_reader_sees_old_or_new_progress_during_publish(tmp_path, monkeypatch, storage_format):
    import os

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source,
                           storage_format=storage_format)
    old_metadata = json.loads((checkpoint / "standalone_resume.json").read_text(encoding="utf-8"))
    original_replace = os.replace

    def observe_replace(src, dst):
        if Path(dst).name == "standalone_resume.json":
            assert load_standalone_resume(checkpoint)[0] == {1}
            original_replace(src, dst)
            assert load_standalone_resume(checkpoint)[0] == {1, 2}
        else:
            original_replace(src, dst)

    monkeypatch.setattr(os, "replace", observe_replace)
    save_standalone_resume(checkpoint, [1, 2], optimizer_step=2, input_path=source,
                           storage_format=storage_format)
    assert (checkpoint / old_metadata["consumed_sequence_file"]).exists()


def test_content_identity_allows_different_paths_with_identical_bytes(tmp_path):
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")
    identity = input_content_identity(build_input_fingerprint(first, include_content_hash=True))
    assert identity == input_content_identity(build_input_fingerprint(second, include_content_hash=True))
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=first, expected_input_identity=identity)
    assert load_standalone_resume(checkpoint, input_path=second)[0] == {0}
    metadata_path = checkpoint / "standalone_resume.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_content_identity"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with its fingerprint"):
        load_standalone_resume(checkpoint, input_path=second)
    with pytest.raises(ValueError, match="conflicts with its fingerprint"):
        load_standalone_resume(checkpoint)


def test_legacy_resume_fingerprint_without_content_hash_still_loads(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source)
    path = checkpoint / "standalone_resume.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["input_fingerprint"]["sha256"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert load_standalone_resume(checkpoint, input_path=source)[0] == {1}


def test_resume_binds_replay_manifest_snapshot(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayWriter

    source = tmp_path / "replay"
    writer = SequenceReplayWriter(source, identity={"model": "v1"}, segment_rows=1)
    writer.append({"sequence_no": 0, "sample_id": "x", "input_ids": [1, 2],
                   "loss_mask": [0, 1], "source_metadata": {}})
    writer.close()
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source)
    assert load_standalone_resume(checkpoint, input_path=source)[0] == {0}
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["model"] = "v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(checkpoint, input_path=source)


def test_bound_replay_identity_allows_equivalent_mount_and_rejects_changed_segment(tmp_path):
    import shutil
    from verl_speco.producer.sequence_replay import SequenceReplayWriter
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    source, copy = tmp_path / "first", tmp_path / "second"
    writer = SequenceReplayWriter(source, identity={"model": "v1"}, segment_rows=1)
    writer.append({"sequence_no": 0, "sample_id": "x", "input_ids": [1, 2],
                   "loss_mask": [0, 1], "source_metadata": {}})
    writer.close()
    shutil.copytree(source, copy)
    identity = input_content_identity(build_input_fingerprint(source, include_content_hash=True))
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=source, expected_input_identity=identity)
    assert load_standalone_resume(checkpoint, input_path=copy)[0] == {0}
    segment = next(copy.glob("segment-*.jsonl"))
    segment.write_bytes(segment.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_standalone_resume(checkpoint, input_path=copy)


def _sealed_checkpoint(root, step, run_id, *, input_path=None, input_identity=None):
    import torch.distributed.checkpoint as dcp
    from verl_speco.trainer.checkpoint import write_standalone_integrity_manifest, write_standalone_optimizer_layout

    checkpoint = root / f"draft_step_{step}"
    optimizer = checkpoint / "optimizer"
    optimizer.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(f"weights-{step}".encode())
    dcp_metadata = dcp.save({"state": torch.tensor([step])}, checkpoint_id=optimizer, no_dist=True)
    write_standalone_optimizer_layout(optimizer, dcp_metadata)
    (optimizer / "trainer_state.pt").write_bytes(b"state")
    metadata = {"step": step, "complete": False, "optimizer": {
        "format": "torch_distributed_checkpoint", "path": "optimizer",
        "trainer_state_file": "trainer_state.pt",
    }}
    path = checkpoint / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    if input_path is not None:
        save_standalone_resume(checkpoint, list(range(step)), optimizer_step=step, input_path=input_path,
                               expected_input_identity=input_identity)
    write_standalone_integrity_manifest(checkpoint, run_id=run_id, require_resume=input_path is not None)
    metadata["complete"] = True
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return checkpoint


def test_optimizer_save_records_layout_for_retention(tmp_path):
    from types import SimpleNamespace
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.checkpoint import write_standalone_integrity_manifest, verify_standalone_checkpoint_integrity

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    def sync(error, _phase):
        if error is not None:
            raise error
    trainer = SimpleNamespace(optimizer=optimizer, model=model, optimizer_steps_total=1, training_steps=1,
                              lr_scheduler=None, fsdp_device_mesh=None, _standalone_retention_keep_last=2,
                              _is_checkpoint_leader=lambda: True, _sync_checkpoint_phase_error=sync,
                              _use_replica_local_bubble_group=lambda: False, _checkpoint_process_group=lambda: None,
                              _atomic_torch_save=lambda data, path: torch.save(data, path),
                              _get_sp_world_size=lambda: 1, _get_dp_world_size=lambda: 1)
    checkpoint = tmp_path / "draft_step_1"
    optimizer_manifest = DrafterBaseTrainer._save_optimizer_checkpoint(trainer, str(checkpoint))
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    metadata = {"step": 1, "complete": False, "optimizer": optimizer_manifest}
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_standalone_integrity_manifest(checkpoint, run_id="a" * 32)
    metadata["complete"] = True
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert verify_standalone_checkpoint_integrity(checkpoint, require_optimizer_layout=True)


@pytest.mark.parametrize("damage", ["missing", "truncated", "metadata"])
def test_optimizer_dependencies_are_required_for_new_integrity_seals(tmp_path, damage):
    from verl_speco.trainer.checkpoint import (retire_standalone_checkpoints,
        verify_standalone_checkpoint_integrity, write_standalone_integrity_manifest)

    checkpoint = _sealed_checkpoint(tmp_path, 1, "a" * 32)
    manifest_path = checkpoint / "standalone_integrity.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = {**original, "schema_version": 1}
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert verify_standalone_checkpoint_integrity(checkpoint)
    assert not verify_standalone_checkpoint_integrity(checkpoint, require_optimizer_layout=True)
    with pytest.raises(ValueError, match="Current checkpoint"):
        retire_standalone_checkpoints(tmp_path, current_step=1, keep_last=2, delete=True)
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["complete"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    shard = next((checkpoint / "optimizer").glob("*.distcp"))
    if damage == "missing":
        shard.unlink()
    elif damage == "truncated":
        shard.write_bytes(b"")
    else:
        (checkpoint / "optimizer" / ".metadata").write_bytes(b"changed")
    with pytest.raises(ValueError, match="dependency is missing or truncated|dependency metadata differs"):
        write_standalone_integrity_manifest(checkpoint, run_id="a" * 32)


def test_rollback_accepts_bound_input_on_an_equivalent_mount(tmp_path):
    from verl_speco.trainer.checkpoint import resolve_verified_standalone_checkpoint
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    source, alternate = tmp_path / "source.jsonl", tmp_path / "copy.jsonl"
    source.write_bytes(b"same rows\n")
    alternate.write_bytes(source.read_bytes())
    identity = input_content_identity(build_input_fingerprint(source, include_content_hash=True))
    root = tmp_path / "checkpoints"
    first = _sealed_checkpoint(root, 1, "a" * 32, input_path=source, input_identity=identity)
    second = _sealed_checkpoint(root, 2, "a" * 32, input_path=source, input_identity=identity)
    (second / "pytorch_model.bin").write_bytes(b"corrupt")
    assert Path(resolve_verified_standalone_checkpoint(second, input_path=alternate)) == first
    alternate.write_bytes(b"other rows\n")
    with pytest.raises(ValueError, match="No verified compatible checkpoint"):
        resolve_verified_standalone_checkpoint(second, input_path=alternate)


def test_checkpoint_retention_plans_and_deletes_only_verified_owned_checkpoints(tmp_path):
    from verl_speco.trainer.checkpoint import retire_standalone_checkpoints, verify_standalone_checkpoint_integrity

    run_id = "a" * 32
    checkpoints = [_sealed_checkpoint(tmp_path, step, run_id) for step in range(1, 6)]
    foreign = _sealed_checkpoint(tmp_path, 6, "b" * 32)
    assert all(verify_standalone_checkpoint_integrity(path) for path in checkpoints)
    options = dict(current_step=5, keep_last=2, protected_paths=(str(checkpoints[1]),))
    planned = retire_standalone_checkpoints(tmp_path, **options)
    assert planned == [str(checkpoints[0]), str(checkpoints[2])]
    assert all(path.exists() for path in checkpoints)
    assert retire_standalone_checkpoints(tmp_path, delete=True, **options) == planned
    assert not checkpoints[0].exists() and not checkpoints[2].exists()
    assert checkpoints[1].exists() and checkpoints[3].exists() and checkpoints[4].exists() and foreign.exists()


def test_checkpoint_retention_inherits_only_latest_verified_same_directory_resume(tmp_path):
    from verl_speco.trainer.checkpoint import standalone_retention_resume_identity, retire_standalone_checkpoints
    source = tmp_path / "input.jsonl"
    source.write_text("sample\n", encoding="utf-8")
    root = tmp_path / "checkpoints"
    first = _sealed_checkpoint(root, 1, "a" * 32, input_path=source)
    resume = _sealed_checkpoint(root, 2, "a" * 32, input_path=source)
    identity = standalone_retention_resume_identity(root, resume, optimizer_step=2)
    assert identity == "a" * 32
    with pytest.raises(ValueError, match="owned directory"):
        standalone_retention_resume_identity(tmp_path, resume, optimizer_step=2)
    with pytest.raises(ValueError, match="owned directory"):
        standalone_retention_resume_identity(root, resume, optimizer_step=1)
    later = _sealed_checkpoint(root, 3, identity, input_path=source)
    with pytest.raises(ValueError, match="later checkpoint"):
        standalone_retention_resume_identity(root, resume, optimizer_step=2)
    _sealed_checkpoint(root, 4, identity, input_path=source)
    assert retire_standalone_checkpoints(root, current_step=4, keep_last=2,
                                         protected_paths=(str(resume),), delete=True) == [str(first)]
    assert not first.exists() and resume.exists() and later.exists()
    (resume / "pytorch_model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="verified checkpoint"):
        standalone_retention_resume_identity(root, resume, optimizer_step=2)


def test_checkpoint_retention_inheritance_requires_sealed_resume(tmp_path):
    from verl_speco.trainer.checkpoint import standalone_retention_resume_identity
    resume = _sealed_checkpoint(tmp_path, 2, "b" * 32)
    with pytest.raises(ValueError, match="sealed replay progress"):
        standalone_retention_resume_identity(tmp_path, resume, optimizer_step=2)


def test_checkpoint_retention_stops_if_lease_is_lost_during_removal(tmp_path):
    from verl_speco.trainer.checkpoint import retire_standalone_checkpoints, verify_standalone_checkpoint_integrity
    paths = [_sealed_checkpoint(tmp_path, step, "a" * 32) for step in range(1, 4)]
    checks = []
    def ownership_check():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("directory ownership lost")
    with pytest.raises(RuntimeError, match="ownership lost"):
        retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2,
                                      delete=True, ownership_check=ownership_check)
    assert paths[0].exists() and not verify_standalone_checkpoint_integrity(paths[0])
    assert all(verify_standalone_checkpoint_integrity(path) for path in paths[1:])


def test_checkpoint_directory_owner_persists_after_process_exit_and_rejects_conflict(tmp_path):
    import subprocess
    import sys
    from verl_speco.trainer.checkpoint import StandaloneCheckpointDirectoryLease

    code = ("from verl_speco.trainer.checkpoint import StandaloneCheckpointDirectoryLease as L; "
            "import sys; L(sys.argv[1], 'a'*32).acquire()")
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
    owner = StandaloneCheckpointDirectoryLease(tmp_path, "a" * 32)
    owner.verify()
    with pytest.raises(RuntimeError, match="already has an owner"):
        StandaloneCheckpointDirectoryLease(tmp_path, "b" * 32).acquire()
    with pytest.raises(RuntimeError, match="token differs"):
        StandaloneCheckpointDirectoryLease(tmp_path, "b" * 32).release()
    owner.release()
    successor = StandaloneCheckpointDirectoryLease(tmp_path, "b" * 32)
    successor.acquire()
    successor.release()


def test_interrupted_retention_invalidates_checkpoint_before_removal(tmp_path, monkeypatch):
    from verl_speco.trainer import checkpoint as module
    checkpoints = [_sealed_checkpoint(tmp_path, step, "a" * 32) for step in range(1, 4)]
    removed = []
    def interrupted(path):
        removed.append(Path(path))
        metadata = module.get_drafter_checkpoint_metadata(path)
        assert metadata["complete"] is False
        (Path(path) / "optimizer" / ".metadata").unlink()
        raise OSError("injected deletion interruption")
    monkeypatch.setattr(module.shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="deletion interruption"):
        module.retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=True)
    assert removed == [checkpoints[0]] and all(path.exists() for path in checkpoints)
    assert not module.is_pretrained_drafter_checkpoint(checkpoints[0])
    assert not module.verify_standalone_checkpoint_integrity(checkpoints[0])
    assert module.verify_standalone_checkpoint_integrity(checkpoints[1])
    assert module.retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2) == []


def test_retention_rechecks_run_identity_after_invalidation(tmp_path, monkeypatch):
    from verl_speco.trainer import checkpoint as module
    checkpoints = [_sealed_checkpoint(tmp_path, step, "a" * 32) for step in range(1, 4)]
    original = module.atomic_write_json
    removed = []
    def replace_identity(path, metadata):
        result = original(path, metadata)
        manifest_path = Path(path).parent / "standalone_integrity.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_id"] = "b" * 32
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result
    monkeypatch.setattr(module, "atomic_write_json", replace_identity)
    monkeypatch.setattr(module.shutil, "rmtree", removed.append)
    with pytest.raises(RuntimeError, match="identity changed after invalidation"):
        module.retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=True)
    assert not removed and all(path.exists() for path in checkpoints)
    assert module.get_drafter_checkpoint_metadata(checkpoints[0])["complete"] is False


def test_retention_invalidation_failure_preserves_complete_checkpoint(tmp_path, monkeypatch):
    from verl_speco.trainer import checkpoint as module
    checkpoints = [_sealed_checkpoint(tmp_path, step, "a" * 32) for step in range(1, 4)]
    def fail_write(*args, **kwargs):
        raise OSError("injected metadata write failure")
    monkeypatch.setattr(module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="metadata write failure"):
        module.retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=True)
    assert all(module.verify_standalone_checkpoint_integrity(path) for path in checkpoints)


@pytest.mark.parametrize("value", [1, "true", None])
def test_retention_requires_explicit_boolean_delete(tmp_path, value):
    from verl_speco.trainer.checkpoint import retire_standalone_checkpoints
    with pytest.raises(ValueError, match="boolean delete"):
        retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=value)


def test_checkpoint_retention_rejects_corruption_and_incomplete_current(tmp_path):
    from verl_speco.trainer.checkpoint import retire_standalone_checkpoints, verify_standalone_checkpoint_integrity

    run_id = "c" * 32
    checkpoints = [_sealed_checkpoint(tmp_path, step, run_id) for step in range(1, 4)]
    (checkpoints[2] / "pytorch_model.bin").write_bytes(b"corrupt")
    assert not verify_standalone_checkpoint_integrity(checkpoints[2])
    with pytest.raises(ValueError, match="Current checkpoint"):
        retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=True)
    assert all(path.exists() for path in checkpoints)


def test_checkpoint_retention_revalidates_successors_before_deleting(tmp_path, monkeypatch):
    from verl_speco.trainer import checkpoint as checkpoint_module

    checkpoints = [_sealed_checkpoint(tmp_path, step, "e" * 32) for step in range(1, 4)]
    original = checkpoint_module.verify_standalone_checkpoint_integrity
    seen = 0
    def verify(path, **kwargs):
        nonlocal seen
        if Path(path).name == "draft_step_2":
            seen += 1
            if seen == 2:
                (Path(path) / "pytorch_model.bin").write_bytes(b"changed")
        return original(path, **kwargs)
    monkeypatch.setattr(checkpoint_module, "verify_standalone_checkpoint_integrity", verify)
    with pytest.raises(RuntimeError, match="revalidation failed"):
        checkpoint_module.retire_standalone_checkpoints(tmp_path, current_step=3, keep_last=2, delete=True)
    assert checkpoints[0].exists() and checkpoints[1].exists() and checkpoints[2].exists()


def test_checkpoint_integrity_requires_matching_replay_progress_when_declared(tmp_path):
    from verl_speco.trainer.checkpoint import write_standalone_integrity_manifest, verify_standalone_checkpoint_integrity

    checkpoint = _sealed_checkpoint(tmp_path, 1, "f" * 32)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["complete"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="matching replay progress"):
        write_standalone_integrity_manifest(checkpoint, run_id="f" * 32, require_resume=True)
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source)
    write_standalone_integrity_manifest(checkpoint, run_id="f" * 32, require_resume=True)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert verify_standalone_checkpoint_integrity(checkpoint)
    (checkpoint / "standalone_resume.json").unlink()
    assert not verify_standalone_checkpoint_integrity(checkpoint)


def test_checkpoint_rollback_uses_same_run_and_updates_launcher_before_producer(tmp_path, monkeypatch):
    from verl_speco.trainer.checkpoint import resolve_verified_standalone_checkpoint
    from verl_speco.standalone_tq_training_launcher import _resolve_checkpoint_rollback, _find_override, _strip_quotes

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "checkpoints"
    first = _sealed_checkpoint(root, 1, "a" * 32, input_path=source)
    second = _sealed_checkpoint(root, 2, "a" * 32, input_path=source)
    requested = _sealed_checkpoint(root, 3, "a" * 32, input_path=source)
    _sealed_checkpoint(root, 4, "b" * 32, input_path=source)
    (requested / "pytorch_model.bin").write_bytes(b"corrupt")
    assert Path(resolve_verified_standalone_checkpoint(requested, input_path=source)) == second
    args = [f"data.train_files={source}", f"actor_rollout_ref.rollout.drafter.model_path={requested}",
            "actor_rollout_ref.rollout.drafter.training.checkpoint_auto_rollback=true"]
    result = _resolve_checkpoint_rollback(args)
    selected = _strip_quotes(_find_override(result, "actor_rollout_ref.rollout.drafter.model_path"))
    assert Path(selected) == second and first.exists()
    source.write_text('{"changed": true}\n', encoding="utf-8")
    from verl_speco.trainer import standalone_resume
    original_fingerprint = standalone_resume.build_input_fingerprint
    calls = []
    def fingerprint(*args, **kwargs):
        calls.append(args[0])
        return original_fingerprint(*args, **kwargs)
    monkeypatch.setattr(standalone_resume, "build_input_fingerprint", fingerprint)
    with pytest.raises(ValueError, match="No verified compatible checkpoint"):
        resolve_verified_standalone_checkpoint(requested, input_path=source)
    assert len(calls) == 1


def test_checkpoint_retention_never_follows_symlinks(tmp_path):
    from verl_speco.trainer.checkpoint import verify_standalone_checkpoint_integrity

    checkpoint = _sealed_checkpoint(tmp_path, 1, "d" * 32)
    outside = tmp_path / "external"
    outside.write_text("keep", encoding="utf-8")
    try:
        (checkpoint / "link").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    assert not verify_standalone_checkpoint_integrity(checkpoint)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_ray_reference_checkpoint_preserves_upstream_shuffle_resume_settings(tmp_path):
    """Both checkpoint call sites must retain main's shuffle metadata for ray_reference."""
    import ast
    from typing import Any

    import torch

    from verl_speco.standalone_tq_producer import _assert_resume_shuffle_alignment

    source = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_standalone_producer_shuffle", "_save_resume_sidecar"}]
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    namespace = {"Any": Any, "torch": torch, "save_standalone_resume": save_standalone_resume,
                 "producer_data_recipe": producer_data_recipe}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    settings = namespace["_standalone_producer_shuffle"]({
        "speco": {"standalone_tq_producer": {"shuffle": True, "shuffle_seed": 73, "source_proportional_field": "source"}},
    })
    loop = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_run_standalone_draft_training_async")
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_save_standalone_checkpoint_across_ranks"]
    assert len(calls) == 2
    for call in calls:
        expression = next(keyword.value for keyword in call.keywords if keyword.arg == "producer_shuffle")
        assert eval(compile(ast.Expression(expression), str(source), "eval"), {
            "feature_store_type": "ray_reference", "standalone_producer_shuffle": settings,
        }) == settings
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"prompt":"q","response":"a"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    namespace["_save_resume_sidecar"](str(checkpoint), torch.tensor([1, 5]), step=7,
                                       input_path=str(input_path), producer_shuffle=settings)
    consumed, metadata = load_standalone_resume(checkpoint, input_path=input_path)
    assert metadata["producer_shuffle"] == settings
    _assert_resume_shuffle_alignment(metadata, shuffle=True, shuffle_seed=73,
                                     consumed_sequence_nos=consumed, source_proportional_field="source",
                                     data_recipe=settings["data_recipe"])


def test_producer_data_recipe_ignores_runtime_capacity_and_keeps_legacy_compatibility():
    from verl_speco.standalone_tq_producer import _assert_resume_shuffle_alignment
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    first = {"max_feature_length": 256, "max_inflight_requests": 2, "publish_workers": 1}
    second = {**first, "max_inflight_requests": 64, "publish_workers": 4}
    assert producer_data_recipe(first) == producer_data_recipe(second)
    legacy = {"producer_shuffle": {"shuffle": False, "shuffle_seed": 42}}
    _assert_resume_shuffle_alignment(legacy, shuffle=False, shuffle_seed=42,
                                     consumed_sequence_nos={1}, data_recipe=producer_data_recipe(second))


def test_resume_requires_same_implicit_source_identity_setting():
    from verl_speco.standalone_tq_producer import _assert_resume_shuffle_alignment
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    legacy = producer_data_recipe({})
    assert legacy == producer_data_recipe({"stable_source_ids": False})
    stable = producer_data_recipe({"stable_source_ids": True})
    metadata = {"producer_shuffle": {"shuffle": True, "shuffle_seed": 42, "data_recipe": stable}}
    _assert_resume_shuffle_alignment(metadata, shuffle=True, shuffle_seed=42,
                                     consumed_sequence_nos={1}, data_recipe=stable)
    with pytest.raises(ValueError, match="recipe"):
        _assert_resume_shuffle_alignment(metadata, shuffle=True, shuffle_seed=42,
                                         consumed_sequence_nos={1}, data_recipe=legacy)


def test_resume_binds_optimizer_step_and_training_plan(tmp_path):
    from verl_speco.trainer.draft_training_loop import _validate_resume_training_plan
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    plan = {"world_size": 2, "batch_size_per_gpu": 4, "seed": 42,
            "deterministic_step_rng": True, "rng_device_type": "cpu"}
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=3, input_path=source, training_plan=plan)
    _, metadata = load_standalone_resume(checkpoint, input_path=source)
    _validate_resume_training_plan(metadata, 3, plan)
    with pytest.raises(ValueError, match="optimizer step"):
        _validate_resume_training_plan(metadata, 2, plan)
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 3, {**plan, "world_size": 4})
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 3, {**plan, "rng_device_type": "cuda"})
    del metadata["training_plan"]
    _validate_resume_training_plan(metadata, 3, plan)  # legacy sidecar


def test_resume_requires_same_deterministic_publication_mode(tmp_path):
    from verl_speco.standalone_tq_producer import _assert_resume_shuffle_alignment
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source,
                           producer_shuffle={"shuffle": False, "shuffle_seed": 42, "deterministic_publication": True})
    consumed, metadata = load_standalone_resume(checkpoint, input_path=source)
    assert metadata["producer_shuffle"]["deterministic_publication"] is True
    _assert_resume_shuffle_alignment(metadata, shuffle=False, shuffle_seed=42,
                                     consumed_sequence_nos=consumed, deterministic_publication=True)
    with pytest.raises(ValueError, match="deterministic publication setting changed"):
        _assert_resume_shuffle_alignment(metadata, shuffle=False, shuffle_seed=42,
                                         consumed_sequence_nos=consumed, deterministic_publication=False)


def test_legacy_training_plan_implies_drop_tail_policy():
    from verl_speco.trainer.draft_training_loop import _validate_resume_training_plan
    old_plan = {"world_size": 2}
    metadata = {"optimizer_step": 3, "training_plan": old_plan}
    _validate_resume_training_plan(metadata, 3, {**old_plan, "tail_policy": "drop"})
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 3, {**old_plan, "tail_policy": "partial"})

@pytest.mark.parametrize("values", [[], [0], [5, 1, 5, 3], [0, 1, 2**63 - 1]])
def test_compressed_resume_round_trip(tmp_path, monkeypatch, values):
    import verl_speco.trainer.standalone_resume as resume

    monkeypatch.setattr(resume, "_CHUNK_ENTRIES", 2)
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, values, optimizer_step=3, input_path=source, storage_format="gzip")
    consumed, metadata = load_standalone_resume(checkpoint, input_path=source)
    assert consumed == set(values)
    assert metadata["schema_version"] == 3


@pytest.mark.parametrize("payload", [b"\x00" * 16, b"\xff" * 8, b"\x01" * 9])
def test_compressed_resume_rejects_invalid_decoded_data(tmp_path, payload):
    import gzip
    import hashlib

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1, 2], optimizer_step=1, input_path=source, storage_format="gzip")
    meta_path = checkpoint / "standalone_resume.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data_path = checkpoint / meta["consumed_sequence_file"]
    data_path.write_bytes(gzip.compress(payload))
    meta["consumed_sequence_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    meta["consumed_sequence_decoded_sha256"] = hashlib.sha256(payload).hexdigest()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        load_standalone_resume(checkpoint)


def test_compressed_resume_rejects_size_and_checksum_corruption(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source, storage_format="gzip")
    meta_path = checkpoint / "standalone_resume.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data_path = checkpoint / meta["consumed_sequence_file"]
    original = data_path.read_bytes()
    data_path.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_standalone_resume(checkpoint)
    data_path.write_bytes(b"x" * 5000)
    with pytest.raises(ValueError, match="byte limit"):
        load_standalone_resume(checkpoint)
    data_path.write_bytes(original)
    meta["consumed_count"] = 100_000_001
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor"):
        load_standalone_resume(checkpoint)


def test_compressed_resume_integrates_with_checkpoint_integrity(tmp_path):
    from verl_speco.trainer.checkpoint import write_standalone_integrity_manifest, verify_standalone_checkpoint_integrity

    checkpoint = _sealed_checkpoint(tmp_path, 1, "f" * 32)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["complete"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    save_standalone_resume(checkpoint, [1], optimizer_step=1, input_path=source, storage_format="gzip")
    write_standalone_integrity_manifest(checkpoint, run_id="f" * 32, require_resume=True)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert verify_standalone_checkpoint_integrity(checkpoint)
    resume = json.loads((checkpoint / "standalone_resume.json").read_text(encoding="utf-8"))
    (checkpoint / resume["consumed_sequence_file"]).write_bytes(b"corrupt")
    assert not verify_standalone_checkpoint_integrity(checkpoint)


def test_resume_recipe_binds_training_semantics_but_allows_operational_changes():
    from copy import deepcopy
    from verl_speco.trainer.draft_training_loop import _standalone_recipe, _validate_resume_training_plan

    config = {
        "lr": 1e-5, "optimizer": "adamw", "adamw_betas": [0.9, 0.95],
        "packing": {"enable": True, "max_packed_len": 2048},
        "dspark_num_anchors": 128, "dspark_max_window": 512,
        "feature_store": {"type": "ray_reference", "max_seq_len": 4096},
        "transfer_queue": {"run_id": "run-a", "expected_feature": {"target_revision": "rev-1"}},
        "save_interval_steps": 100, "reference_streaming_prepare": False,
    }
    recipe = _standalone_recipe(config, "/logical-target")
    metadata = {"optimizer_step": 7, "training_plan": {"world_size": 2, "recipe": recipe}}
    changed = deepcopy(config)
    changed["save_interval_steps"] = 20
    changed["reference_streaming_prepare"] = True
    changed["transfer_queue"]["run_id"] = "new-run"
    assert _standalone_recipe(changed, "/logical-target") == recipe
    for mutate in (
        lambda c: c.update(lr=2e-5),
        lambda c: c["packing"].update(max_packed_len=1024),
        lambda c: c.update(dspark_num_anchors=256),
        lambda c: c["feature_store"].update(max_seq_len=1024),
        lambda c: c["transfer_queue"]["expected_feature"].update(target_revision="rev-2"),
    ):
        changed = deepcopy(config)
        mutate(changed)
        with pytest.raises(ValueError, match="training plan changed"):
            _validate_resume_training_plan(metadata, 7, {"world_size": 2, "recipe": _standalone_recipe(changed, "/logical-target")})
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 7, {"world_size": 2, "recipe": _standalone_recipe(config, "/different-target")})
    _validate_resume_training_plan({"optimizer_step": 7, "training_plan": {"world_size": 2}},
                                   7, {"world_size": 2, "recipe": recipe})


def test_resume_recipe_resolves_omegaconf_values():
    from omegaconf import OmegaConf
    from verl_speco.trainer.draft_training_loop import _standalone_recipe

    config = OmegaConf.create({"base_lr": 0.002, "lr": "${base_lr}",
                               "adamw_betas": [0.9, 0.95], "unrelated_missing": "???"})
    recipe = _standalone_recipe(config, "target")
    assert recipe["settings"]["lr"] == 0.002
    assert recipe["settings"]["adamw_betas"] == [0.9, 0.95]


def test_resume_recipe_detects_changed_target_config(tmp_path):
    from verl_speco.trainer.draft_training_loop import _standalone_recipe, _validate_resume_training_plan

    config_file = tmp_path / "config.json"
    config_file.write_text('{"hidden_size": 4}', encoding="utf-8")
    before = _standalone_recipe({}, str(tmp_path))
    metadata = {"optimizer_step": 2, "training_plan": {"recipe": before}}
    config_file.write_text('{"hidden_size": 8}', encoding="utf-8")
    after = _standalone_recipe({}, str(tmp_path))
    assert before["settings"]["target_config_sha256"] != after["settings"]["target_config_sha256"]
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 2, {"recipe": after})


def test_verified_model_cache_identity_is_path_independent_and_content_bound(tmp_path):
    import os
    import shutil
    from verl_speco.trainer.draft_training_loop import _standalone_recipe, _validate_resume_training_plan

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (first / "config.json").write_text('{"hidden_size":4}', encoding="utf-8")
    (first / "weights.safetensors").write_bytes(b"original-weights")
    shutil.copytree(first, second)
    cfg = {"target_model_cache_content_identity": True, "target_model_logical_id": "model@revision"}
    original = _standalone_recipe(cfg, str(first))
    copied = _standalone_recipe(cfg, str(second))
    assert original == copied
    assert original["settings"]["target_model_path"] == "model@revision"
    plan = {"recipe": original}
    _validate_resume_training_plan({"optimizer_step": 1, "training_plan": plan}, 1, {"recipe": copied})
    weights = second / "weights.safetensors"
    stat = weights.stat()
    weights.write_bytes(b"modified-weights")  # Same size, restored mtime.
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = _standalone_recipe(cfg, str(second))
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan({"optimizer_step": 1, "training_plan": plan}, 1, {"recipe": changed})


def test_verified_model_cache_normalizes_expected_local_target_path(tmp_path):
    import shutil
    from verl_speco.trainer.draft_training_loop import _standalone_recipe

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (first / "config.json").write_text("{}", encoding="utf-8")
    (first / "weights.bin").write_bytes(b"weights")
    shutil.copytree(first, second)
    def cfg(path):
        return {"target_model_cache_content_identity": True, "target_model_logical_id": "model@revision",
                "transfer_queue": {"expected_feature": {"target_model_path": str(path)}}}
    assert _standalone_recipe(cfg(first), str(first)) == _standalone_recipe(cfg(second), str(second))


def test_model_cache_rejects_config_change_during_hash(tmp_path, monkeypatch):
    import verl_speco.trainer.draft_training_loop as loop

    config_file = tmp_path / "config.json"
    config_file.write_text('{"v":1}', encoding="utf-8")
    (tmp_path / "weights.bin").write_bytes(b"weights")
    original = loop.model_cache_content_sha256
    def fingerprint(path):
        result = original(path)
        config_file.write_text('{"v":2}', encoding="utf-8")
        return result
    monkeypatch.setattr(loop, "model_cache_content_sha256", fingerprint)
    with pytest.raises(ValueError, match="changed while fingerprinting"):
        loop._standalone_recipe({"target_model_cache_content_identity": True,
                                 "target_model_logical_id": "model"}, str(tmp_path))


@pytest.mark.parametrize("config,path", [
    ({"target_model_cache_content_identity": "true"}, "missing"),
    ({"target_model_cache_content_identity": True}, "missing"),
    ({"target_model_cache_content_identity": True, "target_model_logical_id": "model"}, "missing"),
])
def test_verified_model_cache_rejects_invalid_configuration(config, path):
    from verl_speco.trainer.draft_training_loop import _standalone_recipe
    with pytest.raises(ValueError):
        _standalone_recipe(config, path)


def _run_two_rank(worker, *args):
    """Run Gloo workers with bounded cleanup even when a child fails."""
    import multiprocessing
    import socket
    import torch.distributed as dist

    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=worker, args=(rank, port, *args, results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        output = [results.get(timeout=55) for _ in processes]
        assert {item[0] for item in output} == {0, 1}
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        return output
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


def _step_rng_resume_worker(rank, port, directory, results):
    import random
    from datetime import timedelta
    import numpy as np
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _seed_standalone_step
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(123)
        model = torch.nn.Linear(4, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        def step(model, optimizer, index):
            _seed_standalone_step(17, index, rank)
            scale = random.random() + float(np.random.random())
            loss = (model(torch.rand(3, 4)) * scale).square().mean()
            optimizer.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
            optimizer.step()
        for index in range(2):
            step(model, optimizer, index)
        checkpoint = Path(directory) / f"rng-rank-{rank}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        for index in range(2, 4):
            step(model, optimizer, index)
        expected = [parameter.detach().clone() for parameter in model.parameters()]
        torch.rand(23)  # Restart-time random work must not change the next step.
        resumed = torch.nn.Linear(4, 1)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        resumed.load_state_dict(state["model"])
        resumed_optimizer.load_state_dict(state["optimizer"])
        for index in range(2, 4):
            step(resumed, resumed_optimizer, index)
        results.put((rank, all(torch.equal(actual, wanted) for actual, wanted in zip(resumed.parameters(), expected, strict=True))))
    finally:
        dist.destroy_process_group()


def test_two_rank_step_rng_matches_training_after_checkpoint_resume(tmp_path):
    assert all(equal for _, equal in _run_two_rank(_step_rng_resume_worker, str(tmp_path)))


def _model_cache_recipe_worker(rank, port, paths, mode, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _standalone_recipe_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        try:
            recipe = _standalone_recipe_across_ranks(
                {"target_model_cache_content_identity": True, "target_model_logical_id": "target@rev",
                 "trainable_snapshot_interval_steps": 1 if mode == "interval" and rank else 0,
                 "trainable_snapshot_max_bytes": 1024 if mode == "budget" and rank else 0},
                paths[rank], 2)
            results.put((rank, recipe["sha256"]))
        except ValueError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["matching", "mismatch", "missing", "interval", "budget"])
def test_model_cache_identity_checked_on_every_rank(tmp_path, case):
    import shutil

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (first / "config.json").write_text("{}", encoding="utf-8")
    (first / "weights.bin").write_bytes(b"weights-a")
    shutil.copytree(first, second)
    if case == "mismatch":
        (second / "weights.bin").write_bytes(b"weights-b")
    if case == "missing":
        second = tmp_path / "missing"
    outputs = _run_two_rank(_model_cache_recipe_worker, [str(first), str(second)], case)
    if case == "matching":
        assert outputs[0][1] == outputs[1][1] and len(outputs[0][1]) == 64
    else:
        assert all("recipe differs or failed" in value for _, value in outputs)


def test_trainable_snapshot_excludes_frozen_parameters_and_checks_integrity(tmp_path):
    from types import SimpleNamespace
    import torch
    from verl_speco.trainer.checkpoint import read_standalone_trainable_snapshot
    from verl_speco.trainer.draft_training_loop import _save_trainable_snapshot_across_ranks

    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    trainer = SimpleNamespace(model=model, checkpoint_dir=str(tmp_path),
                              _standalone_training_plan={"recipe": "target@rev"})
    path = Path(_save_trainable_snapshot_across_ranks(trainer, 7, device=torch.device("cpu")))
    weights, manifest = read_standalone_trainable_snapshot(path, expected_training_plan=trainer._standalone_training_plan)
    assert set(weights) == {"weight"}
    torch.testing.assert_close(weights["weight"], model.weight)
    assert manifest["optimizer_step"] == 7 and manifest["kind"] == "trainable_parameters_only"
    assert not (path / "optimizer").exists()
    with pytest.raises(ValueError, match="identity differs"):
        read_standalone_trainable_snapshot(path, expected_training_plan={"recipe": "different"})
    with (path / "trainable.safetensors").open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_standalone_trainable_snapshot(path)


def test_trainable_snapshot_rejects_conflicting_tied_weights_before_copy(tmp_path):
    from types import SimpleNamespace
    import torch
    from verl_speco.trainer.checkpoint import write_standalone_trainable_snapshot
    from verl_speco.trainer.draft_training_loop import _load_trainable_snapshot_across_ranks

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor(0.0))
            self.second = self.first
    model = Model()
    trainer = SimpleNamespace(model=model, _standalone_training_plan={"recipe": "same"})
    path = write_standalone_trainable_snapshot(tmp_path / "snapshot", step=1,
        tensors={"first": torch.tensor(1.0), "second": torch.tensor(2.0)},
        training_plan=trainer._standalone_training_plan)
    with pytest.raises(RuntimeError, match="initialization failed"):
        _load_trainable_snapshot_across_ranks(trainer, path, device=torch.device("cpu"))
    assert model.first.item() == 0.0 and model.first is model.second


def test_trainable_snapshot_byte_limit_applies_before_gather_and_copy(tmp_path):
    from types import SimpleNamespace
    import torch
    from verl_speco.trainer.draft_training_loop import (
        _load_trainable_snapshot_across_ranks, _save_trainable_snapshot_across_ranks,
    )

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trainer = SimpleNamespace(model=model, checkpoint_dir=str(tmp_path), _standalone_training_plan={})
    with pytest.raises(RuntimeError, match="collection failed"):
        _save_trainable_snapshot_across_ranks(trainer, 1, device=torch.device("cpu"), max_bytes=11)
    assert not list(tmp_path.iterdir())
    path = _save_trainable_snapshot_across_ranks(trainer, 1, device=torch.device("cpu"), max_bytes=12)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    with pytest.raises(RuntimeError, match="initialization failed"):
        _load_trainable_snapshot_across_ranks(trainer, path, device=torch.device("cpu"), max_bytes=11)
    assert torch.count_nonzero(model.weight).item() == 0 and torch.count_nonzero(model.bias).item() == 0
    _load_trainable_snapshot_across_ranks(trainer, path, device=torch.device("cpu"), max_bytes=12)
    assert torch.count_nonzero(model.weight).item() > 0


def _trainable_snapshot_initialize_worker(rank, port, paths, mismatch, results):
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _load_trainable_snapshot_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(7)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
        if rank == 1 and mismatch:
            model.module.bias.requires_grad_(False)
        before = {name: parameter.detach().clone() for name, parameter in model.module.named_parameters()}
        trainer = SimpleNamespace(model=model, _standalone_training_plan={"recipe": "same"})
        try:
            metadata = _load_trainable_snapshot_across_ranks(trainer, paths[rank], device=torch.device("cpu"))
            results.put((rank, "loaded", model.module.weight.detach().tolist(), metadata["optimizer_step"]))
        except RuntimeError as exc:
            unchanged = all(torch.equal(before[name], parameter) for name, parameter in model.module.named_parameters())
            results.put((rank, str(exc), unchanged, None))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["valid", "mismatch", "different_payload"])
def test_trainable_snapshot_initialization_validates_all_ranks_before_copy(tmp_path, case):
    import torch
    from verl_speco.trainer.checkpoint import write_standalone_trainable_snapshot

    plan = {"recipe": "same"}
    weights = {"weight": torch.full((1, 2), 3.0), "bias": torch.full((1,), 4.0)}
    first = write_standalone_trainable_snapshot(tmp_path / "first", step=5, tensors=weights, training_plan=plan)
    paths = [first, first]
    if case == "different_payload":
        paths[1] = write_standalone_trainable_snapshot(tmp_path / "second", step=5,
            tensors={"weight": torch.full((1, 2), 9.0), "bias": weights["bias"]}, training_plan=plan)
    output = _run_two_rank(_trainable_snapshot_initialize_worker, paths, case == "mismatch")
    if case == "valid":
        assert all(item[1:] == ("loaded", [[3.0, 3.0]], 5) for item in output)
    else:
        assert all("initialization failed" in item[1] and item[2] for item in output)


def _trainable_snapshot_worker(rank, port, directory, mismatch, results):
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _save_trainable_snapshot_across_ranks

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(123)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
        if rank == 1 and mismatch:
            model.module.bias.requires_grad_(False)
        trainer = SimpleNamespace(model=model, checkpoint_dir=directory,
                                  _standalone_training_plan={"recipe": "same"})
        try:
            result = _save_trainable_snapshot_across_ranks(trainer, 5, device=torch.device("cpu"))
            results.put((rank, result))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["valid", "mismatch", "write_failure"])
def test_trainable_snapshot_gloo_writer_and_schema_agreement(tmp_path, case):
    from verl_speco.trainer.checkpoint import read_standalone_trainable_snapshot

    directory = tmp_path
    if case == "write_failure":
        directory = tmp_path / "not_a_directory"
        directory.write_text("x", encoding="utf-8")
    output = _run_two_rank(_trainable_snapshot_worker, str(directory), case == "mismatch")
    if case == "mismatch":
        assert all("collection failed" in value for _, value in output)
        assert not list(tmp_path.glob("trainable_step_*"))
    elif case == "write_failure":
        assert all("publication failed" in value for _, value in output)
    else:
        assert output[0][1] == output[1][1]
        weights, manifest = read_standalone_trainable_snapshot(output[0][1])
        assert set(weights) == {"weight", "bias"} and manifest["optimizer_step"] == 5
        assert len(list(tmp_path.glob("trainable_step_*"))) == 1


def _trainable_dtensor_worker(rank, port, directory, shard_dim, rows, results):
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from verl_speco.trainer.draft_training_loop import (
        _load_trainable_snapshot_across_ranks, _save_trainable_snapshot_across_ranks,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=35))
    try:
        mesh = init_device_mesh("cpu", (2,))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(distribute_tensor(torch.arange(float(rows * 3)).reshape(rows, 3), mesh, [Shard(shard_dim)]))
                self.replicated = torch.nn.Parameter(distribute_tensor(torch.tensor([42.0]), mesh, [Replicate()]))
        model = Model()
        trainer = SimpleNamespace(model=model, checkpoint_dir=directory,
                                  _standalone_training_plan={"world_size": 2, "recipe": "same"})
        expected = {name: value.to_local().detach().clone() for name, value in model.named_parameters()}
        budget = rows * 3 * 4 + 4
        path = _save_trainable_snapshot_across_ranks(trainer, 2, device=torch.device("cpu"), max_bytes=budget)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.to_local().zero_()
        try:
            _load_trainable_snapshot_across_ranks(trainer, path, device=torch.device("cpu"), max_bytes=budget)
            matched = all(torch.equal(value.to_local(), expected[name]) for name, value in model.named_parameters())
            results.put((rank, "loaded", matched, path))
        except RuntimeError as exc:
            unchanged = all(not torch.count_nonzero(value.to_local()).item() for value in model.parameters())
            results.put((rank, str(exc), unchanged, path))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim,rows", [(0, 5), (0, 1), (1, 5)])
def test_trainable_snapshot_real_cpu_dtensor_shards(tmp_path, shard_dim, rows):
    from verl_speco.trainer.checkpoint import read_standalone_trainable_snapshot

    output = _run_two_rank(_trainable_dtensor_worker, str(tmp_path), shard_dim, rows)
    assert output[0][3] == output[1][3]
    if shard_dim == 0:
        assert all(item[1] == "loaded" and item[2] for item in output)
        weights, _ = read_standalone_trainable_snapshot(output[0][3])
        assert weights["weights"].tolist() == [[float(i * 3 + j) for j in range(3)] for i in range(rows)]
    else:
        assert all("initialization failed" in item[1] and item[2] for item in output)


def _trainable_fsdp_worker(rank, port, directory, version, results):
    from datetime import timedelta
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, fully_shard
    from verl_speco.trainer.draft_training_loop import (
        _load_trainable_snapshot_across_ranks, _save_trainable_snapshot_across_ranks,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=35))
    try:
        torch.manual_seed(11)
        base = torch.nn.Linear(2, 3)
        base.bias.requires_grad_(False)
        if version == 1:
            expected_weight, expected_bias = base.weight.detach().clone(), base.bias.detach().clone()
            model = FSDP(base, device_id=torch.device("cpu"), use_orig_params=True)
        else:
            model = base
            fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
            expected_weight = model.weight.to_local().detach().clone()
            expected_bias = model.bias.to_local().detach().clone()
        trainer = SimpleNamespace(model=model, checkpoint_dir=directory,
                                  _standalone_training_plan={"world_size": 2, "recipe": "same"})
        path = _save_trainable_snapshot_across_ranks(trainer, 2, device=torch.device("cpu"), max_bytes=24)
        if version == 1:
            with FSDP.summon_full_params(model, recurse=True, writeback=True, rank0_only=False):
                with torch.no_grad():
                    model.module.weight.zero_()
        else:
            with torch.no_grad():
                model.weight.to_local().zero_()
        _load_trainable_snapshot_across_ranks(trainer, path, device=torch.device("cpu"), max_bytes=24)
        if version == 1:
            with FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=False):
                good = torch.equal(model.module.weight, expected_weight) and torch.equal(model.module.bias, expected_bias)
        else:
            good = torch.equal(model.weight.to_local(), expected_weight) and torch.equal(model.bias.to_local(), expected_bias)
        results.put((rank, good, path))
    except Exception as exc:
        results.put((rank, f"{type(exc).__name__}: {exc}", None))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("version", [1, 2])
def test_trainable_snapshot_real_cpu_fsdp(tmp_path, version):
    from verl_speco.trainer.checkpoint import read_standalone_trainable_snapshot

    output = _run_two_rank(_trainable_fsdp_worker, str(tmp_path), version)
    assert all(item[1] is True for item in output), output
    assert output[0][2] == output[1][2]
    weights, _ = read_standalone_trainable_snapshot(output[0][2])
    assert set(weights) == {"weight"}


def _gloo_recipe_resume_worker(rank, port, checkpoint, source, mismatch, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _load_resume_across_ranks, _standalone_recipe

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        config = {"lr": 0.02 if mismatch and rank == 1 else 0.01,
                  "packing": {"enable": True}, "dspark_num_anchors": 16}
        plan = {"world_size": 2, "recipe": _standalone_recipe(config, None)}
        try:
            consumed, _ = _load_resume_across_ranks(checkpoint, source, 1, plan, 2)
            results.put((rank, sorted(consumed)))
        except ValueError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


def _gloo_retention_resume_worker(rank, port, root, checkpoint, mismatch, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _retention_resume_identity_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        metadata = None if mismatch and rank == 1 else {"optimizer_step": 2}
        try:
            results.put((rank, _retention_resume_identity_across_ranks(root, checkpoint, 2, metadata, 2)))
        except ValueError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


def _gloo_shared_file_probe_worker(rank, port, root, other_root, mode, results):
    import asyncio
    import sys
    from datetime import timedelta
    from types import SimpleNamespace
    import torch.distributed as dist
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    from verl_speco.trainer.draft_training_loop import _verify_shared_file_root_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        selected_root = other_root if mode == "root_mismatch" and rank == 1 else root
        hidden = FileHiddenStatesStore(selected_root)
        if mode == "not_visible" and rank == 1:
            def missing(reference):
                raise FileNotFoundError("injected nonshared mount")
            hidden.validate_reference = missing
        config = {"speco": {"standalone_tq_producer": {"hidden_states_store": {"file_root": root}}}}
        actor = StandaloneProducerActor(config)
        handle = SimpleNamespace(
            create_shared_file_probe=SimpleNamespace(remote=lambda: asyncio.run(actor.create_shared_file_probe())),
            remove_shared_file_probe=SimpleNamespace(remote=lambda path, token: asyncio.run(actor.remove_shared_file_probe(path, token))),
        )
        sys.modules["ray"] = SimpleNamespace(get=lambda value, timeout: value)
        try:
            _verify_shared_file_root_across_ranks(SimpleNamespace(hidden_states_store=hidden), handle,
                                                  rank=rank, device=torch.device("cpu"), timeout=5)
            results.put((rank, "ok"))
        except RuntimeError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["shared", "not_visible", "root_mismatch"])
def test_real_two_rank_shared_file_witness(tmp_path, mode):
    root, other = tmp_path / "shared", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    outputs = _run_two_rank(_gloo_shared_file_probe_worker, str(root), str(other), mode)
    assert all((result == "ok") if mode == "shared" else ("Shared file root" in result) for _, result in outputs)
    assert not list(root.glob(".speco-shared-probe-*"))


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_two_rank_retention_resume_identity(tmp_path, mismatch):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "checkpoints"
    checkpoint = _sealed_checkpoint(root, 2, "c" * 32, input_path=source)
    outputs = _run_two_rank(_gloo_retention_resume_worker, str(root), str(checkpoint), mismatch)
    if mismatch:
        assert all("inheritance failed across ranks" in value for _, value in outputs)
    else:
        assert all(value == "c" * 32 for _, value in outputs)


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("mismatch", [False, True])
def test_real_two_rank_resume_checks_recipe_consensus(tmp_path, mismatch):
    from verl_speco.trainer.draft_training_loop import _standalone_recipe

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    config = {"lr": 0.01, "packing": {"enable": True}, "dspark_num_anchors": 16}
    save_standalone_resume(checkpoint, [0, 3], optimizer_step=1, input_path=source,
                           training_plan={"world_size": 2, "recipe": _standalone_recipe(config, None)})
    outputs = _run_two_rank(_gloo_recipe_resume_worker, str(checkpoint), str(source), mismatch)
    if mismatch:
        assert all("differs or failed across ranks" in value for _, value in outputs)
    else:
        assert all(value == [0, 3] for _, value in outputs)
