"""Real, consecutive prefix+window sampling for optional stage-two EF fine-tuning."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.temporal_data import TemporalEchoDataset
from utils.echo_input import read_echo_input
from utils.datasets import _as_video_tensor
from echo_aug_validation.io_utils import find_echonet_video
from echo_aug_validation.augment_recipes import augment_video


class CompleteHistoryEchoDataset(TemporalEchoDataset):
    def __init__(self, *args, aug_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_cfg=aug_cfg
        self.excluded=[]
        keep=[]
        for i,case in enumerate(tqdm(self.ids,desc='Check real EF history')):
            path=find_echonet_video(self.root,case)
            if path is None:
                raise FileNotFoundError(case)
            raw=read_echo_input(path,self.input_protocol)
            length=len(raw)
            if isinstance(raw,np.memmap):
                raw._mmap.close()
            if length>=self.frames:
                keep.append(i)
            else:
                self.excluded.append(dict(id=case,frames=length,required=self.frames))
        self.df=self.df.iloc[keep].reset_index(drop=True)
        self.ids=[Path(str(x)).stem for x in self.df.FileName]
        if not len(self.df):
            raise ValueError('No complete-history EF cases; never repeat frames as historical evidence')

    def __getitem__(self,index):
        row=super().__getitem__(index)
        if not bool(row['frame_valid'].all()):
            raise RuntimeError('Video changed since complete-history filtering')
        if self.aug_cfg is not None:
            # Gray caches remain gray; replicate channels only after augmentation.
            video=row['video'][:,0].numpy() if self.input_protocol=='gray_repeat3' else row['video'].permute(0,2,3,1).numpy()
            video=augment_video(video,self.aug_cfg,self.seed+index*997+self.epoch.value*104729,False)
            row['video']=_as_video_tensor(video,self.frames,self.img_size,channels=self.channels)
        return row
