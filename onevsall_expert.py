import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import F1Score

class ExpertDataset(Dataset):

    def __init__(self, df, tokenizer, max_token_len: int = 128, sample = 5000):
        self.data = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.sample = sample
        self._prepare_data()

    def _prepare_data(self):
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
                                            return_attention_mask = True
                                            )
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}


class ExpertDataModule(pl.LightningDataModule):

    def __init__(self, train_df, val_df, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = ExpertDataset(self.train_df, tokenizer=self.tokenizer)
            self.val_dataset = ExpertDataset(self.val_df, tokenizer=self.tokenizer, sample=None)
        if stage == 'predict':
            self.val_dataset = ExpertDataset(self.val_df, tokenizer=self.tokenizer, sample=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)


class ExpertClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
        self.hidden = torch.nn.Linear(
                self.pretrained_model.config.hidden_size,
                self.pretrained_model.config.hidden_size
                )
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout()

        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.f1_func = F1Score(task='binary', average='macro')
        
    def forward(self, input_ids, attention_mask, labels=None):
        # roberta layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # final logits
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss and f1
        loss = 0
        f1 = 0
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
            f1 = self.f1_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
        return loss, f1, logits
    
    def training_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("train f1 ", f1, prog_bar = True, logger=True)
        self.log("train loss ", loss, prog_bar = True, logger=True)
        return {"loss":loss, "train f1":f1, "predictions":outputs, "labels": batch["labels"]}
    
    def validation_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("val f1", f1, prog_bar = True, logger=True)
        self.log("val loss ", loss, prog_bar = True, logger=True)
        return {"val_loss": loss, "val f1":f1, "predictions":outputs, "labels": batch["labels"]}
    
    def predict_step(self, batch, batch_index):
        _, _, outputs = self(**batch)
        return outputs
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

