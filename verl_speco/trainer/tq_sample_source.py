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
"""Distributed streaming sample source for a TQ-backed standalone Consumer."""

from verl_speco.trainer.reference_sample_source import (
    ScheduledFeatureDataLoader as TQFeatureDataLoader,
)
from verl_speco.trainer.standalone_data import (
    StandaloneLocalBatch as TQLocalBatch,
    _entry_from_wire,
    _entry_to_wire,
    build_assignments,
)
from verl_speco.trainer.standalone_prefetch import (
    StandalonePrefetchIterator as TQPrefetchIterator,
    _prepared_host_bytes,
)

__all__ = [
    "TQFeatureDataLoader",
    "TQLocalBatch",
    "TQPrefetchIterator",
    "_entry_from_wire",
    "_entry_to_wire",
    "_prepared_host_bytes",
    "build_assignments",
]
