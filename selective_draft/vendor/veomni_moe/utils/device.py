# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
#
# Modified for dOPSD: reduced to the two helpers the vendored group-GEMM kernels use (CUDA only; the NPU branch of
# the original veomni/utils/device.py is dropped).

from typing import Any

import torch


def get_torch_device() -> Any:
    """torch.cuda (the original also supports torch.npu)."""
    return torch.cuda


def get_device_name() -> str:
    """Real device name, e.g. NVIDIA A100-PCIE-40GB."""
    return get_torch_device().get_device_name()
