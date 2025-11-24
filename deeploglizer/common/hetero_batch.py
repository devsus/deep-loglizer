import math
import torch
from torch.utils.data import Sampler
from typing import Iterable, List, Sequence

#het seq
def get_capacity_ratios(
        world_size: int,
        method: str = "mixed",
        normaize: bool = True
):
    capacities = []
    for dev_idx in range(world_size):
        props = torch.cuda.get_device_properties(dev_idx)

        if method == "memory":
            cap = props.total_memory / (1024 ** 3)
        elif method == "compute":
            cap = props.multi_processor_count * props.clock_rate
        elif method == "mixed":
            mem = props.total_memory / (1024 ** 3)
            compute = props.multi_processor_count * props.clock_rate
            cap = 0.5 * mem + 0.5 * compute
        capacities.append(float(cap))

    if not normaize:
        return capacities

    mean_cap = sum(capacities) / len(capacities)
    ratios = [c / mean_cap for c in capacities]

    print(
        "Capacity ratios (%s): capacities=%s, normalized ratios=%s",
        method, capacities, ratios
    )
    return ratios

def compute_per_rank_batch_sizes(
        global_batch_size: int,
        world_size: int,
        ratios: Sequence[float],
) -> List[int]:
    assert world_size > 0
    assert len(ratios) == world_size
    assert global_batch_size >= world_size

    ratios =[float(r) for r in ratios]
    total = sum(ratios)
    assert total > 0

    sizes = [max(1, int(round(global_batch_size * r / total))) for r in ratios]

    # fix rounding so that sum == global_batch_size
    diff = global_batch_size - sum(sizes)
    i = 0
    while diff != 0:
        if diff > 0:
            sizes[i] += 1
            diff -= 1
        else:  # diff < 0
            if sizes[i] > 1:
                sizes[i] -= 1
                diff += 1
        i = (i + 1) % world_size

    return sizes

class HeteroBatchSampler(Sampler[List[int]]):
    def __init__(
            self,
            dataset_size: int,
            rank: int,
            world_size: int,
            per_rank_batch_sizes: Sequence[int],
            seed: int = 0,
    ) -> None:
        super().__init__(None)
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size

        assert len(per_rank_batch_sizes) == world_size
        self.per_rank_batch_sizes = [int(b) for b in per_rank_batch_sizes]
        assert all(b > 0 for b in self.per_rank_batch_sizes)
        assert 0 <= self.rank < self.world_size

        self.local_batch_size = self.per_rank_batch_sizes[self.rank]
        self.global_batch_size = sum(self.per_rank_batch_sizes)

        assert self.dataset_size > 0
        assert self.global_batch_size > 0

        # same num steps on all ranks
        self.num_steps = math.ceil(self.dataset_size / self.global_batch_size)
        self.seed = int(seed)

        self.epoch = 0

        prefix = [0]
        for b in self.per_rank_batch_sizes:
            prefix.append(prefix[-1] + b)
        self._offset_within_global = prefix[self.rank]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterable[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.dataset_size, generator=g).tolist()

        total_needed = self.num_steps * self.global_batch_size

        if total_needed <= len(perm):
            full = perm[:total_needed]
        else:
            k = total_needed // len(perm) + 1
            full = (perm * k)[:total_needed]

        offset = self._offset_within_global
        gbs = self.global_batch_size
        lbs = self.local_batch_size

        for step in range(self.num_steps):
            base = step * gbs + offset
            batch = full[base: base + lbs]
            yield batch