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
"""Bounded, JSON-only proposal records. No rollout supervision tensors."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

PROPOSAL_KEY = "_specopd_proposals"
PLAN_KEY = "specopd_plan"
TRACE_KEY = "specopd_proposals"
SCHEMA = 1


def token_digest(ids):
    return hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ProposalRecord:
    branch_id: str
    prefix_len: int
    proposal_ids: tuple[int, ...]
    prefix_digest: str
    round_id: int
    accepted_count: int
    verified_count: int | None = None

    def validate(self):
        if (
            not isinstance(self.branch_id, str)
            or not self.branch_id
            or type(self.prefix_len) is not int
            or self.prefix_len < 1
        ):
            raise ValueError("invalid SpecOPD branch identity/prefix")
        if not self.proposal_ids or any(
            type(i) is not int or i < 0 for i in self.proposal_ids
        ):
            raise ValueError("proposal must contain valid nonnegative token IDs")
        if (
            type(self.accepted_count) is not int
            or type(self.round_id) is not int
            or not 0 <= self.accepted_count <= len(self.proposal_ids)
            or self.round_id < 0
            or (
                self.verified_count is not None
                and (
                    type(self.verified_count) is not int
                    or not self.accepted_count
                    <= self.verified_count
                    <= len(self.proposal_ids)
                )
            )
        ):
            raise ValueError("invalid verification diagnostics")
        if (
            not isinstance(self.prefix_digest, str)
            or len(self.prefix_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.prefix_digest)
        ):
            raise ValueError("missing exact prefix digest")


def encode_bundle(bundle, max_bytes):
    wire = json.dumps(bundle, separators=(",", ":"), allow_nan=False)
    if len(wire.encode()) > max_bytes:
        raise ValueError("SpecOPD proposal wire exceeds max_wire_bytes")
    return wire


def decode_bundle(wire, max_bytes=1048576):
    if not isinstance(wire, str) or not wire or len(wire.encode()) > max_bytes:
        raise ValueError("invalid or oversized SpecOPD wire")
    bundle = json.loads(wire)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != SCHEMA
        or not isinstance(bundle.get("records"), list)
    ):
        raise ValueError("unsupported SpecOPD schema/records")
    if not isinstance(bundle.get("request_id"), str) or not bundle["request_id"]:
        raise ValueError("invalid SpecOPD request identity")
    for key in ("actor_revision", "drafter_revision"):
        if type(bundle.get(key)) is not int or bundle[key] < 0:
            raise ValueError("invalid SpecOPD revision")
    temperature = bundle.get("proposal_temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid SpecOPD sampling temperature")
    ids = set()
    for raw in bundle["records"]:
        record = ProposalRecord(**{**raw, "proposal_ids": tuple(raw["proposal_ids"])})
        record.validate()
        if record.branch_id in ids:
            raise ValueError("duplicate proposal branch")
        ids.add(record.branch_id)
    return bundle


def record_dict(record):
    record.validate()
    return asdict(record)
