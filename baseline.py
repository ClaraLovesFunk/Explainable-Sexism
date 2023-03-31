import pandas as pd
import numpy as np
import evaluate
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

class model:
    def __init__(self, df, num_labels, model_id='bert-base-cased', ckpt_dir='ckpts', num_epochs=3):
        """
        General Model Class for task

        Args:
            df (pd.DataFrame): {"text":str, "label":int}
            num_labels (int): number of labels in the task
            model_id (str, optional): model_id of HuggingFace model to use. Defaults to "bert-base-cased".
            ckpt_dir (str, optional): checkpoint dir to store the checkpoints. Defaults to 'ckpts'.
            num_epochs (int, optional): number of epochs to train the model. Defaults to 3.
        """
        self.model_id = model_id
        self.df = df
        self.num_labels = num_labels
        self.ckpt_dir = ckpt_dir
        self.num_epochs = num_epochs
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.training_args = TrainingArguments(
            output_dir=self.ckpt_dir, 
            evaluation_strategy="epoch", 
            num_train_epochs=self.num_epochs,
        )
        self.metric = evaluate.load("f1")

    def tokenize_function(self, data):
        return self.tokenizer(data["text"], padding="max_length", truncation=True, return_tensors='pt')

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels, average="macro")

    def train(self):
        X_train, X_eval, y_train, y_eval = train_test_split(self.df['text'], self.df['label'], test_size=0.1, random_state=42)
        train_data = Dataset.from_dict({
            'text': X_train.values,
            'label': y_train.values
        })
        eval_data = Dataset.from_dict({
            'text': X_eval.values,
            'label': y_eval.values
        })
        tokenized_train_dataset = train_data.map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = eval_data.map(self.tokenize_function, batched=True)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=self.compute_metrics,
        )   
        result = trainer.train()
        return result

def binary_baseline(df):
    df.drop(['rewire_id', 'label_category', 'label_vector'], axis=1, inplace=True)    
    df['label_sexist'].replace({'not sexist':0, 'sexist':1}, inplace=True)
    df.rename(columns={'label_sexist':'label'}, inplace=True)
    print(df.head(10))
    obj = model(df, 2)
    result = obj.train()
    

if __name__ == '__main__':
    df = pd.read_csv('data/train_all_tasks.csv')
    binary_baseline(df)
    """
    This is file is no longer maintained in favour of the ipynb file
    """
