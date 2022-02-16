import json

import torch
import os

from base import BaseDataLoader
from torch.utils.data import DataLoader

from dataloader.dataset import SentenceDataset
from utils import Path, ensure_dir, read_json


class SentenceDataLoader(BaseDataLoader):
    r"""
    Sentence Data dataloader
    """

    def create_dataloader(
            self,
            root_path: str,
            data_file: str,
            label_file: str = None,
            phase: str = None
    ):
        data_path = Path(f"{root_path}/{data_file}")
        label_path = Path(f"{root_path}/{label_file}")

        dir_exists = ensure_dir(root_path)
        if not dir_exists:
            print(f"Creating directory {root_path}. If you expected directory to exist please verify config. ")

        data = torch.load(os.path.join(data_path))
        # ds = SentenceDataset(data)

        return DataLoader(
            dataset=SentenceDataset(data),
            **self.init_param_kwargs
        )


# method

def build_vocab(root_path: str, emb_size: int, load_embeddings: bool = False):

    word_map = load_word_map(root_path)
    # size of vocabulary
    vocab_size = len(word_map)

    label_map, _ = get_label_maps()

    # number of classes
    n_classes = len(label_map)

    # word embeddings
    if load_embeddings:
        raise NotImplementedError("Failed to load embeddings")
        # load Glove as pre-trained word embeddings for words in the word map
        # emb_path = os.path.join(config.emb_folder, config.emb_filename)
        # embeddings, emb_size = load_embeddings(
        #    emb_file=os.path.join(config.emb_folder, config.emb_filename),
        #    word_map=word_map,
        #    output_folder=config.output_path
        # )
    # or initialize embedding weights randomly
    else:
        embeddings = None
        emb_size = emb_size

    return embeddings, emb_size, word_map, n_classes, vocab_size


def load_word_map(root_path):
    # load word2index map
    word_map_path = Path(f"{root_path}/word_map.json")

    word_map = read_json(word_map_path)

    return word_map


def get_label_maps():
    agnews_classes = [
        'World',
        'Sports',
        'Business',
        'Sci / Tech'
    ]
    label_map = {k: v for v, k in enumerate(agnews_classes)}
    rev_label_map = {v: k for k, v in label_map.items()}
    return label_map, rev_label_map
