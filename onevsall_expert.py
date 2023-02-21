import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassF1Score
from pytorch_lightning.loggers import TensorBoardLogger


def preproc_df(df, label, num_classes):
    new_map = {i: 0 for i in range(num_classes)}
    new_map[label] = 1
    binary_df = df.copy(deep=True)
    binary_df['label'].replace(new_map, inplace=True)
    return binary_df


def train_experts(df_train, df_dev, df_test, config, doTrain=False, doTest=False):
    expert_models = []
    for label in range(config['num_classes']):
        proc_df_train = preproc_df(df_train, label, config['num_classes'])
        proc_df_dev = preproc_df(df_dev, label, config['num_classes'])
        proc_df_test = preproc_df(df_test, label, config['num_classes'])

        dm = ExpertDataModule(
            proc_df_train,
            proc_df_dev,
            proc_df_test,
            batch_size=config['batch_size'],
            balance=config['balance'],
            model_name=config['hf_model_name']
        )
        dm.setup()

        config['train_size'] = len(dm.train_dataloader())
        config['model_filename'] = f"{config['expert_type']}{label}_bal{config['balance']}_b{config['batch_size']}_e{config['n_epochs']}"

        expert_model = ExpertClassifier(config)

        logger = TensorBoardLogger(save_dir=config['log_dir'], name=config['model_filename'])
        trainer = pl.Trainer(
            max_epochs=config['n_epochs'],
            accelerator='gpu',
            devices=1,
            num_sanity_val_steps=15,
            default_root_dir=config['root_dir'],
            logger=logger,
            enable_checkpointing=False,
        )

        if doTrain:
            trainer.fit(expert_model, dm)
            torch.save(expert_model.state_dict(), f"{config['save_dir']}/{config['model_filename']}.pt")

        ckpt = torch.load(f"{config['save_dir']}/{config['model_filename']}.pt")
        expert_model.load_state_dict(ckpt)
        expert_models.append(expert_model)

        if doTest:
            expert_model.eval()
            test_dataloader = dm.test_dataloader()
            trainer.test(expert_model, dataloaders=test_dataloader)

    return expert_models


class ExpertDataset(Dataset):

    def __init__(self, df, tokenizer, max_token_len: int = 128, balance=False):
        self.data = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.balance = balance
        self._prepare_data()

    def _prepare_data(self):
        if self.balance:
            binary_class = self.data[self.data['label'] == 1]
            binary_notClass = self.data[self.data['label'] == 0]

            if len(binary_class) > len(binary_notClass):
                self.data = pd.concat([binary_notClass, binary_class.sample(len(binary_notClass), random_state=0)])
            else:
                self.data = pd.concat([binary_class, binary_notClass.sample(len(binary_class), random_state=0)])

        self.data['class'] = np.where(self.data['label'] == 1, 1, 0)
        self.data['notClass'] = np.where(self.data['label'] == 0, 1, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item.text)
        labels = torch.FloatTensor([item['class'], item['notClass']])
        tokens = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_len,
                                            return_attention_mask=True
                                            )
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}


class ExpertDataModule(pl.LightningDataModule):

    def __init__(self, df_train, df_dev, df_test, batch_size, max_token_length: int = 128, model_name='roberta-base', balance=False):
        super().__init__()
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.balance = balance

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = ExpertDataset(self.df_train, tokenizer=self.tokenizer, balance=self.balance)
            self.dev_dataset = ExpertDataset(self.df_dev, tokenizer=self.tokenizer)
            self.test_dataset = ExpertDataset(self.df_test, tokenizer=self.tokenizer)
        if stage == 'predict':
            self.test_dataset = ExpertDataset(self.df_test, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


class ExpertClassifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['hf_model_name'], return_dict=True)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        self.soft = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout()

        self.loss_func = nn.BCELoss()
        self.f1_func = MulticlassF1Score(num_classes=self.config['n_labels'])

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = self.soft(logits)

        # calculate loss and f1
        loss = 0
        f1 = 0
        if labels is not None:
            loss = self.loss_func(outputs, labels)
            f1 = self.f1_func(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
        return loss, f1, outputs

    def training_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("train f1 ", f1, prog_bar=True, logger=True)
        self.log("train loss ", loss, prog_bar=True, logger=True)
        return {"loss": loss, "train f1": f1, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("val f1", f1, prog_bar=True, logger=True)
        self.log("val loss ", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "val f1": f1, "predictions": outputs, "labels": batch["labels"]}

    def test_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("test f1", f1, prog_bar=True, logger=True)
        self.log("test loss ", loss, prog_bar=True, logger=True)
        return {"loss": loss, "test f1": f1, "predictions": outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        _, _, outputs = self(**batch)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size'] / self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
