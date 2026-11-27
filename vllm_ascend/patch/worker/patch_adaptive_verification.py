# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Extend AV's existing TP agreement to the PCP replicas on Ascend."""

from dataclasses import dataclass

from vllm.distributed import get_pcp_group, get_tp_group
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.worker.gpu.spec_decode import adaptive_verification


@dataclass(frozen=True)
class AdaptiveVerificationGroup:
    tp_group: GroupCoordinator
    pcp_group: GroupCoordinator

    def broadcast_object(self, obj=None, src: int = 0):
        obj = self.tp_group.broadcast_object(obj, src=src)
        return self.pcp_group.broadcast_object(obj, src=src)

    def broadcast(self, tensor, src: int = 0):
        self.tp_group.broadcast(tensor, src=src)
        return self.pcp_group.broadcast(tensor, src=src)


def get_adaptive_verification_group() -> GroupCoordinator | AdaptiveVerificationGroup:
    tp_group, pcp_group = get_tp_group(), get_pcp_group()
    if pcp_group.world_size == 1:
        return tp_group
    return AdaptiveVerificationGroup(tp_group, pcp_group)


# Only AV's module-local agreement dependency is rebound. Its constructor,
# budget kernel, request compaction and admitted-count D2H lifecycle stay
# upstream-owned; other TP collectives retain their original groups.
# Remove once the upstream AV manager agrees across PCP as well as TP.
adaptive_verification.get_tp_group = get_adaptive_verification_group
