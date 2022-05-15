import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, \
    BertForSequenceClassification, pipeline
from HFDataset import HFDataSet

PATH: str = "/content/gdrive/MyDrive/model"

def label_to_language(label: str) -> str:
    if label == "LABEL_0":
        return "French"
    elif label == "LABEL_1":
        return "Norwegian"
    elif label == "LABEL_2":
        return "Russian"
    return label


class BertModel:
    def __init__(self, hf: HFDataSet, path: str = PATH, is_trained: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = path
        self.ds = hf.get_dataset()
        print('Using device:', self.device)
        self.tokenizer = hf.get_tokenizer()
        if is_trained:
            self.model = BertForSequenceClassification.from_pretrained(self.path).to(self.device)
        else:
            self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3).to(self.device)
        self.trainer = self.build_trainer()
        self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0)

    @staticmethod
    def get_training_arguments(learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, epochs=1):
        return TrainingArguments(
            output_dir=PATH,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )

    def build_trainer(self):
        train_args = self.get_training_arguments()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return Trainer(
            model=self.model,  # the instantiated Transformers model to be trained
            args=train_args,  # training arguments, defined above
            train_dataset=self.ds["train"],  # training dataset
            eval_dataset=self.ds["val"],  # evaluation dataset
            compute_metrics=self.compute_metrics,  # the callback that computes metrics of interest
            data_collator=data_collator
        )

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)
        r = recall_score(labels, preds)
        p = precision_score(labels, preds)
        return {
            'accuracy': acc,
        }

    def train_model(self):
        self.trainer.train()
        self.trainer.save_model(PATH)

    def evaluate_model(self, with_test_set: bool = True):
        if with_test_set:
            metrics = self.trainer.predict(self.ds["test"])
        else:
            metrics = self.trainer.evaluate()
        print(metrics)
        return metrics

    def resume_training(self, checkpoint_path: str):
        self.trainer.train(resume_from_checkpoint=checkpoint_path)
        self.trainer.save_model(PATH)

    def predict_sentence(self, sentence: str):
        res = self.pipe(sentence)
        label = res[0]['label']
        print("Predicting L1 of the author of the sentence: ", sentence)
        print(label_to_language(label))
        print("Score: ", res[0]['score'])
        return res


def label_to_language(label: str) -> str:
    if label == "LABEL_0":
        return "French"
    elif label == "LABEL_1":
        return "Norwegian"
    elif label == "LABEL_2":
        return "Russian"
    return label
