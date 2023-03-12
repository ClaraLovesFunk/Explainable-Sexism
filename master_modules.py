import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
import torch.nn.functional as F 
from torchmetrics.classification import MulticlassF1Score
from transformers import set_seed 



class Master_Dataset(Dataset): 

  def __init__(self, data, tokenizer, attributes, max_token_len: int = 128, sample=False): 
    self.data = data
    self.tokenizer = tokenizer
    self.attributes = attributes
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self): 

    if self.sample:                          

      label_derogation = self.data.loc[self.data['1. threats, plans to harm and incitement']==1] 
      label_animosity = self.data.loc[self.data['2. derogation']==1]
      label_threats = self.data.loc[self.data['3. animosity']==1]
      label_prejudice = self.data.loc[self.data['4. prejudiced discussions']==1]

      # figure out smallest class
      class_sizes = [len(label_derogation), len(label_animosity), len(label_threats), len(label_prejudice)]
      sample_size = min(class_sizes)

      self.data = pd.concat([
        label_derogation.sample(sample_size, random_state=0), 
        label_animosity.sample(sample_size, random_state=0), 
        label_threats.sample(sample_size, random_state=0), 
        label_prejudice.sample(sample_size, random_state=0)
        ])
    
  def __len__(self): 
    return len(self.data)

  def __getitem__(self, index):  
    item = self.data.iloc[index]
    comment = str(item.text)             
    attributes = torch.FloatTensor(item[self.attributes])
    tokens = self.tokenizer.encode_plus(comment,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_token_len,
                                        return_attention_mask = True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}







class Master_DataModule(pl.LightningDataModule):

  def __init__(self, model_id, X_train, X_test, attributes, batch_size: int = 16, max_token_length: int = 128, sample=True, seed = 0):     
    super().__init__() 
    
    self.seed = seed       
    set_seed(seed)    
         
    self.X_train = X_train
    self.X_test = X_test
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_id = model_id
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.sample = sample 

  def setup(self, stage = None): 
    if stage in (None, "fit"): 
      self.train_dataset = Master_Dataset(self.X_train, attributes=self.attributes, tokenizer=self.tokenizer, sample = self.sample) 
      self.val_dataset = Master_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer) 
    if stage == 'test':
      self.test_dataset = Master_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer) 
    if stage == 'predict': 
      self.val_dataset = Master_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer) 

  def train_dataloader(self): 
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True) # CAN WE SHUFFLE AND STILL USE SEED

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)




class Master_Classifier(pl.LightningModule):

  def __init__(self, config_master: dict, config_expert:dict):
    super().__init__()
    self.config_master = config_master
    self.config_expert = config_expert

    device = torch.device("cuda")
    self.expert = self.config_master['experts'][0]       
    self.expert1 = self.config_master['experts'][1]

    self.expert.to(device)
    self.expert1.to(device)

    self.expert.eval()
    self.expert1.eval()

    self.hidden = torch.nn.Linear(self.expert.config.hidden_size+self.expert1.config.hidden_size, 512) 
    self.classifier = torch.nn.Linear(512, self.config_master['n_labels']) 
    self.soft = torch.nn.Softmax(dim=1)
    torch.nn.init.xavier_uniform_(self.classifier.weight) 
    self.loss_func = nn.BCEWithLogitsLoss(reduction='mean') 
    self.dropout = nn.Dropout()
    self.f1_func = MulticlassF1Score(num_classes = self.config_master['n_labels']) 
    
  def forward(self, input_ids, attention_mask, labels=None):
    # roberta layer
    output0 = self.expert(input_ids=input_ids, attention_mask=attention_mask)  
    pooled_output0 = torch.mean(output0.last_hidden_state, 1)

    output1 = self.expert1(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output1 = torch.mean(output1.last_hidden_state, 1)

    pooled_output = torch.cat((pooled_output0, pooled_output1), 1)

    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    logits = self.soft(logits)
    # calculate loss and f1
    loss = 0
    f1 = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config_master['n_labels']), labels.view(-1, self.config_master['n_labels'])) 
      f1 = self.f1_func(logits.view(-1, self.config_master['n_labels']), labels.view(-1, self.config_master['n_labels']))      
    return loss, f1, logits

  def training_step(self, batch, batch_index):
    loss, f1, outputs = self(**batch)
    self.log("train f1", f1, prog_bar = True, logger=True)
    self.log("train loss", loss, prog_bar = True, logger=True)
    return {"loss":loss, "train f1":f1, "predictions":outputs, "labels": batch["labels"]}
  
  def validation_step(self, batch, batch_index):  
    loss, f1, outputs = self(**batch)
    self.log("val f1", f1, prog_bar = True, logger=True)
    self.log("val loss", loss, prog_bar = True, logger=True)
    return {"val_loss": loss, "val f1":f1, "predictions":outputs, "labels": batch["labels"]}

  def test_step(self, batch, batch_index):
    loss, f1, outputs = self(**batch)
    self.log("test f1", f1, prog_bar = True, logger=True)
    self.log("test loss", loss, prog_bar = True, logger=True)
    return {"test_loss": loss, "test f1":f1, "predictions":outputs, "labels": batch["labels"]}
  
  def predict_step(self, batch, batch_index):
    _, _, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config_master['lr'], weight_decay=self.config_master['weight_decay'])
    total_steps = self.config_master['train_size']/self.config_master['batch_size']
    warmup_steps = math.floor(total_steps * self.config_master['warmup'])
    warmup_steps = math.floor(total_steps * self.config_master['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]