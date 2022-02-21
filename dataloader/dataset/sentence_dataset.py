
from torch.utils.data import Dataset
from base import BaseDataset
import torch



class SentenceDataset(BaseDataset):

    def __init__(self, data):
        self.data = data


    def __getitem__(self, i: int):
        return torch.LongTensor(self.data['sents'][i]), \
               torch.LongTensor([self.data['labels'][i]])

    def __len__(self) -> int:
        return len(self.data['labels'])