import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel


class BalancedDataParallel(DataParallel):
    """
    平衡的数据并行处理，允许为不同GPU分配不同的batch size
    """

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """
    将输入数据按照指定的chunk_sizes分散到不同的GPU上
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return obj.split(chunk_sizes, dim)
        if isinstance(obj, (list, tuple)):
            return [scatter_map(o) for o in obj]
        if isinstance(obj, dict):
            return {k: scatter_map(v) for k, v in obj.items()}
        return [obj for _ in target_gpus]

    # After scattering: len(inputs) == len(target_gpus)
    scattered_inputs = scatter_map(inputs)
    scattered_kwargs = scatter_map(kwargs)

    if len(scattered_inputs) < len(target_gpus):
        scattered_inputs.extend(
            [() for _ in range(len(target_gpus) - len(scattered_inputs))]
        )
    if len(scattered_kwargs) < len(target_gpus):
        scattered_kwargs.extend(
            [{} for _ in range(len(target_gpus) - len(scattered_kwargs))]
        )

    # Move tensors to their respective devices
    scattered_inputs = tuple(
        tuple(
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs_per_gpu
        )
        for inputs_per_gpu, device in zip(scattered_inputs, target_gpus)
    )

    scattered_kwargs = tuple(
        {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs_per_gpu.items()
        }
        for kwargs_per_gpu, device in zip(scattered_kwargs, target_gpus)
    )

    return scattered_inputs, scattered_kwargs
