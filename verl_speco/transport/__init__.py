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
"""Transport protocols shared by standalone SPECO producers and consumers."""

from verl_speco.transport.drafter_sample_protocol import (
    DRAFTER_TQ_PARTITION,
    PROTOCOL_SCHEMA_VERSION,
    ExpectedFeatureConfig,
    SampleMetadata,
    decode_sample,
    encode_sample,
    is_ready_sample_tag,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
    parse_ready_tag,
)

__all__ = [
    "DRAFTER_TQ_PARTITION",
    "PROTOCOL_SCHEMA_VERSION",
    "ExpectedFeatureConfig",
    "SampleMetadata",
    "decode_sample",
    "encode_sample",
    "is_ready_sample_tag",
    "make_eos_record",
    "make_ready_tag",
    "make_sample_key",
    "parse_ready_tag",
]
