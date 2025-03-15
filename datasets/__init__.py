from .mind import MINDDataset
from .ml_100k import ML100kDataset
from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .steam import SteamDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    ML100kDataset.code(): ML100kDataset,
    MINDDataset.code(): MINDDataset,
    SteamDataset.code(): SteamDataset
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
