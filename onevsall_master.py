import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassF1Score
from pytorch_lightning.loggers import TensorBoardLogger


def train_master(df_train, df_dev, df_test, config, doTrain=False, doTest=False):
    dm = MasterDataModule(
        df_train,
        df_dev,
        df_test,
        batch_size=config['batch_size'],
        model_name=config['hf_model_name']
    )
    dm.setup()

    config['train_size'] = len(dm.train_dataloader())
    config['model_filename'] = f"master_b{config['batch_size']}_e{config['n_epochs']}"

    model = MasterClassifier(config)

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

    if not doTrain:
        ckpt = torch.load(f"{config['save_dir']}/{config['model_filename']}.pt")
        model.load_state_dict(ckpt)
    else:
        trainer.fit(model, dm)
        torch.save(model.state_dict(), f"{config['save_dir']}/{config['model_filename']}.pt")

    if doTest:
        model.eval()
        test_dataloader = dm.test_dataloader()
        trainer.test(model, dataloaders=test_dataloader)

    return model


class MasterDataset(Dataset):

    def __init__(self, df, tokenizer, max_token_len: int = 128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self._prepare_data()

    def _prepare_data(self):
        for i in range(4):
            self.data[f'{i}'] = np.where(self.data['label'] == i, 1, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item.text)
        labels = torch.FloatTensor([item[f'{i}'] for i in range(4)])
        tokens = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_len,
                                            return_attention_mask=True
                                            )
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}


class MasterDataModule(pl.LightningDataModule):

    def __init__(self, df_train, df_dev, df_test, batch_size, max_token_length: int = 128, model_name='roberta-base'):
        super().__init__()
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test
        self.batch_size = batch_size
        self.max_token_length = max_token_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = MasterDataset(self.df_train, tokenizer=self.tokenizer)
            self.dev_dataset = MasterDataset(self.df_dev, tokenizer=self.tokenizer)
            self.test_dataset = MasterDataset(self.df_test, tokenizer=self.tokenizer)
        if stage == 'predict':
            self.test_dataset = MasterDataset(self.df_test, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


class MasterClassifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        device = torch.device("cuda")
        self.experts = self.config['experts']
        for exp in self.experts:
            exp.to(device)
            # exp.eval()
            for param in exp.parameters():
                param.requires_grad = False

        self.hidden = torch.nn.Linear(
            self.experts[0].config.hidden_size * self.config['n_labels'],
            self.experts[0].config.hidden_size,
        )
        self.hidden2 = torch.nn.Linear(
            self.experts[0].config.hidden_size,
            self.experts[0].config.hidden_size,
        )
        self.hidden3 = torch.nn.Linear(
            self.experts[0].config.hidden_size,
            512,
        )
        self.classifier = torch.nn.Linear(
            512,
            self.config['n_labels']
        )
        self.soft = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(p=0.2)

        self.loss_func = nn.CrossEntropyLoss()
        self.f1_func = MulticlassF1Score(num_classes=self.config['n_labels'])
        self.f1_func_none = MulticlassF1Score(num_classes=self.config['n_labels'], average=None)
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

        pooled_output = torch.cat((pooled_output0, pooled_output1, pooled_output2, pooled_output3), 1)

        # final logits
        output = self.hidden(pooled_output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.hidden2(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.hidden3(output)
        output = F.relu(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        outputs = self.soft(logits)

        # calculate loss and f1
        loss = 0
        f1 = 0
        f1_none = 0
        if labels is not None:
            loss = self.loss_func(outputs, labels)
            f1 = self.f1_func(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            f1_none = self.f1_func_none(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
        return loss, f1, f1_none, outputs

    def training_step(self, batch, batch_index):
        loss, f1, _, outputs = self(**batch)
        self.log("train f1 ", f1, prog_bar=True, logger=True)
        self.log("train loss ", loss, prog_bar=True, logger=True)
        return {"loss": loss, "train f1": f1, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, f1, _, outputs = self(**batch)
        self.log("val f1", f1, prog_bar=True, logger=True)
        self.log("val loss ", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "val f1": f1, "predictions": outputs, "labels": batch["labels"]}

    def test_step(self, batch, batch_index):
        loss, f1, f1_none, outputs = self(**batch)
        # f1_none = f1_none.cpu()

        self.log("Harm f1", f1_none[0], prog_bar=True, logger=True)
        self.log("Derogation f1", f1_none[1], prog_bar=True, logger=True)
        self.log("Animosity f1", f1_none[2], prog_bar=True, logger=True)
        self.log("Prejudice f1", f1_none[3], prog_bar=True, logger=True)
        self.log("test f1", f1, prog_bar=True, logger=True)
        self.log("test loss ", loss, prog_bar=True, logger=True)
        return {"loss": loss, "test f1": f1, "test f1 none": f1_none, "predictions": outputs, "labels": batch["labels"]}

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
