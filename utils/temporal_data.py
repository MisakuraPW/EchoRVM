"""Deterministic audit clips and stochastic training clips with padding metadata."""

from pathlib import Path
import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import _as_video_tensor
from echo_aug_validation.io_utils import load_echonet_filelist, find_echonet_video
from .echo_input import INPUT_PROTOCOLS, read_echo_input


class TemporalEchoDataset(Dataset):
    def __init__(self, root, split, frames, img_size=112, channels=3, sampling_rate=1,
                 limit=None, seed=42, two_views=False, random_start=None, input_protocol='rgb'):
        self.root = Path(root)
        self.df = load_echonet_filelist(self.root, split)
        # A seeded permutation prevents prefix selection from depending on file ordering.
        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if limit is not None:
            self.df = self.df.iloc[:int(limit)]
        self.frames, self.img_size, self.channels = int(frames), int(img_size), int(channels)
        self.input_protocol = input_protocol
        if input_protocol not in INPUT_PROTOCOLS or (input_protocol == 'gray_repeat3' and self.channels != 3):
            raise ValueError('gray_repeat3 keeps the three-channel model; use channels=3.')
        self.stride, self.seed, self.two_views = int(sampling_rate), int(seed), bool(two_views)
        self.epoch = multiprocessing.Value('i', 0)
        self.random_start = split.lower() == 'train' if random_start is None else random_start
        if len(self.df) == 0 or self.frames < 1 or self.stride < 1:
            raise ValueError('Empty split or invalid sampling configuration.')
        self.ids = [Path(str(x)).stem for x in self.df.FileName]

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch):
        self.epoch.value = int(epoch)

    def sample(self, raw, index, view=0):
        required = (self.frames - 1) * self.stride + 1
        if len(raw) < 2:
            raise ValueError('A temporal experiment requires at least two real frames.')
        maximum = max(0, len(raw) - required)
        epoch = self.epoch.value if self.random_start else 0
        rng = np.random.RandomState((self.seed + index * 997 + view * 7919 + epoch * 104729) % (2**32))
        start = int(rng.randint(maximum + 1))
        indices = start + np.arange(self.frames) * self.stride
        valid = indices < len(raw)
        clip = np.zeros((self.frames, *raw.shape[1:]), dtype=raw.dtype)
        clip[valid] = raw[indices[valid]]
        tensor = _as_video_tensor(clip, self.frames, self.img_size, channels=self.channels)
        return tensor, torch.from_numpy(valid), torch.from_numpy(np.where(valid, indices, -1))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = find_echonet_video(self.root, str(row.FileName))
        if path is None:
            raise FileNotFoundError(row.FileName)
        raw = read_echo_input(path, self.input_protocol)
        video, valid, indices = self.sample(raw, index)
        sample = dict(video=video, frame_valid=valid, frame_indices=indices, id=self.ids[index],
                      target=torch.tensor(float(row.EF)), dataset='echonet', source_path=str(path),
                      input_protocol=self.input_protocol)
        if self.two_views:
            sample['video_view2'], _, _ = self.sample(raw, index, 1)
        return sample
