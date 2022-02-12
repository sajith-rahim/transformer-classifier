from dataclasses import dataclass

from config import Paths, Files, Params


@dataclass
class ClfConfig:
    paths: Paths
    files: Files
    params: Params
