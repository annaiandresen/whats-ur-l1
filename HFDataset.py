import datasets
import pandas as pd
import numpy as np
from CreateDataset import Dataset
from transformers import BertTokenizerFast
import os.path

PATH = "/content/gdrive/MyDrive/model/hf"


class HFDataSet:
    """
    Either loads a preexisting Hugging Face Dataset from path or creates a new dataset from a pandas dataframe.
    Splits the Pandas dataframe into train-val-test sets.
    Uses a BertTokenizerFast to tokenize the datasets.
    Creates a DatasetDict that stores each set.
    """

    def __init__(self, path: str = PATH):
        self.path = path
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        if os.path.exists(self.path):
            print("Loading dataset from " + self.path)
            self.df_dict = datasets.DatasetDict.load_from_disk(self.path)
        else:
            self.df = Dataset()
            self.df_dict = self.build()

    def train_test_val(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Shuffles a Pandas dataframe, then splits it into train, validation and test set with ratios 60/20/20.
        """
        train, validate, test = np.split(self.df.sample(frac=1), [int(.6 * len(self.df)), int(.8 * len(self.df))])
        return train, validate, test

    def build(self, save_to_disk: bool = True) -> datasets.DatasetDict:
        """
        Separates a dataframe into train, validation and test sets.
        Tokenizes each of these sets.
        :return: a DatasetDict with each tokenized set
        """
        train, val, test = self.train_test_val()
        train_ds, val_ds, test_ds = datasets.Dataset.from_pandas(train), datasets.Dataset.from_pandas(
            val), datasets.Dataset.from_pandas(
            test)
        train_ds_tok = self.tokenize_element(train_ds)
        val_ds_tok = self.tokenize_element(val_ds)
        test_ds_tok = self.tokenize_element(test_ds)
        dict = datasets.DatasetDict(
            {
                "train": train_ds_tok,
                "val": val_ds_tok,
                "test": test_ds_tok
            }
        )

        if save_to_disk:
            dict.save_to_disk(self.path)
            print("Dataset saved to " + self.path)
        return dict

    def tokenize_element(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Tokenizes each sentence in dataset
        :param dataset: a Dataset object.
        :return: a tokenized dataset.
        """
        dataset_tok = dataset.map(
            lambda ds: self.tokenizer(ds['text'], truncation=True, padding='max_length', max_length=512),
            batched=True).remove_columns(['text', '__index_level_0__', 'l1'])
        dataset_tok.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dataset_tok.rename_column("label", "labels")
        return dataset_tok

    def get_dataset(self):
        return self.df_dict

    def get_tokenizer(self):
        return self.tokenizer
