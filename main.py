import torch
import numpy as np
import random
from BERT import BertModel
from HFDataset import HFDataSet


def label_to_language(label: str) -> str:
    if label == "LABEL_0":
        return "French"
    elif label == "LABEL_1":
        return "Norwegian"
    elif label == "LABEL_2":
        return "Russian"
    return label


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    hf = HFDataSet()
    model = BertModel(hf, is_trained=True)
    # model.train_model()
    # model.evaluate_model()
    # model.predict_sentence("Hello my name is Anna and I'm from Norway")