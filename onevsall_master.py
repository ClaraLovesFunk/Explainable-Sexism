import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassF1Score

class MasterDataset(Dataset):

    def __init__(self, df, tokenizer, max_token_len: int = 128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self._prepare_data()

    def _prepare_data(self):
        for i in range(5):
            self.data[f'{i}'] = np.where(self.data['label'] == i, 1, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item.text)
        labels = torch.FloatTensor([item[f'{i}'] for i in range(5)])
        tokens = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_len,
                                            return_attention_mask = True
                                            )
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}


class MasterDataModule(pl.LightningDataModule):

    def __init__(self, train_df, val_df, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = MasterDataset(self.train_df, tokenizer=self.tokenizer)
            self.val_dataset = MasterDataset(self.val_df, tokenizer=self.tokenizer)
        if stage == 'predict':
            self.val_dataset = MasterDataset(self.val_df, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)


class MasterClassifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        device = torch.device("cuda")
        self.experts = self.config['experts']
        for exp in self.experts:
            exp.to(device)
            exp.eval()

        self.hidden = torch.nn.Linear(
                self.experts[0].config.hidden_size*self.config['n_labels'],
                512
                )
        self.classifier = torch.nn.Linear(512, self.config['n_labels'])
        self.soft = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout()

        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.f1_func = MulticlassF1Score(num_classes=self.config['n_labels'])
        print('init done')
        
    def forward(self, input_ids, attention_mask, labels=None):
        output0 = self.experts[0](input_ids=input_ids, attention_mask=attention_mask)
        pooled_output0 = torch.mean(output0.last_hidden_state, 1)

        output1 = self.experts[1](input_ids=input_ids, attention_mask=attention_mask)
        pooled_output1 = torch.mean(output1.last_hidden_state, 1)

        output2 = self.experts[2](input_ids=input_ids, attention_mask=attention_mask)
        pooled_output2 = torch.mean(output2.last_hidden_state, 1)

        output3 = self.experts[3](input_ids=input_ids, attention_mask=attention_mask)
        pooled_output3 = torch.mean(output3.last_hidden_state, 1)

        output4 = self.experts[4](input_ids=input_ids, attention_mask=attention_mask)
        pooled_output4 = torch.mean(output4.last_hidden_state, 1)

        pooled_output = torch.cat((pooled_output0, pooled_output1, pooled_output2, pooled_output3, pooled_output4), 1)

        # final logits
        output = self.dropout(pooled_output)
        output = self.hidden(output)
        output = F.relu(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.soft(logits)

        # calculate loss and f1
        loss = 0
        f1 = 0
        if labels is not None:
            loss = self.loss_func(logits, labels)
            f1 = self.f1_func(logits, labels)
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
    
    def test_step(self, batch, batch_index):
        loss, f1, outputs = self(**batch)
        self.log("test f1", f1, prog_bar = True, logger=True)
        self.log("test loss ", loss, prog_bar = True, logger=True)
        return {"loss":loss, "test f1":f1, "predictions":outputs, "labels": batch["labels"]}

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
