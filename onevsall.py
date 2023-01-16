import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

class ExpertDataset(Dataset):

  def __init__(self, df, tokenizer, labels, max_token_len: int = 128, sample = 5000):
    self.data = df
    self.tokenizer = tokenizer
    self.labels = labels
    self.max_token_len = max_token_len
    self.sample = sample

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data.iloc[index]
    comment = str(item.comment)
    labels = torch.FloatTensor(item[self.labels])
    tokens = self.tokenizer.encode_plus(comment,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_token_len,
                                        return_attention_mask = True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}


class ExpertDataModule(pl.LightningDataModule):

  def __init__(self, train_df, val_df, labels, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
    super().__init__()
    self.train_df = train_df
    self.val_df = val_df
    self.labels = labels
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = ExpertDataset(self.train_df, labels=self.labels, tokenizer=self.tokenizer)
      self.val_dataset = ExpertDataset(self.val_df, labels=self.labels, tokenizer=self.tokenizer, sample=None)
    if stage == 'predict':
      self.val_dataset = ExpertDataset(self.val_df, labels=self.labels, tokenizer=self.tokenizer, sample=None)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

if __name__ == "__main__":
    #device = 0 if torch.cuda.is_available() else -1
    #model = AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")
    #tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
    #pipe = pipeline(
    #        'feature-extraction',
    #        model = model,
    #        tokenizer = tokenizer,
    #        device = device,
    #)
    #res = pipe("Test sentence")
    #print(res)


    df = pd.read_csv('data/train_all_tasks.csv')

    label_map = {
            'none':0,
            '1. threats, plans to harm and incitement' : 1,
            '2. derogation': 2,
            '3. animosity': 3,
            '4. prejudiced discussions': 4
            }
    df.drop(['rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)    
    df['label_category'].replace(label_map, inplace=True)
    df.rename(columns={'label_category':'label'}, inplace=True)

    grouped_dfs = df.groupby(['label'])

    for (_, df) in grouped_dfs:
        X_train, X_val, Y_train, Y_val = train_test_split(df['text'], df['label'], test_size=0.2)
        #datamodule  = ExpertDataModule(X)



