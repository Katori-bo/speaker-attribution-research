import torch
from torch.utils.data import DataLoader
from typing import List

from src.neural.schemas import NovelSequence

def novel_collate_fn(batch: List[NovelSequence]) -> List[NovelSequence]:
    # Since batch_size is 1, batch will contain exactly one NovelSequence
    # We do not pad initially as per instructions.
    return batch

def get_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=novel_collate_fn
    )
