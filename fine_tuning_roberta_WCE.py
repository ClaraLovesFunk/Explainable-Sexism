%env TOKENIZERS_PARALLELISM=True

#!nvidia-smi

%%capture ##########    WHATS THAT??????
#!pip install transformers 
!pip install pytorch-lightning

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F 
from torchmetrics.classification import F1Score

model_name =  'GroNLP/hateBERT' #'distilroberta-base' 

# PREPARE AND INSPECT DATA

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'data/train_all_tasks.csv'
data = pd.read_csv(data_path)

# represent label_sexist in a seperate binary column
data['sexist'] = np.where(data['label_sexist'] == 'sexist', 1, 0)

########## MAKE CLEAN COLUMN HERE AND NOT IN DATASET MODULE

# represent label_category in seperate binary columns
labels = ['1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions']     

for k in labels:
    data[k] = np.where(data['label_category'] == k, 1, 0)

attributes = labels ######## ATTRIBUTES SHOULD BE CALLED LABELS

# plot data
data[attributes].sum().plot.bar()

# measure class sizes
binary_labels = ['sexist']
no_sexist = data[binary_labels].sum()
no_no_sexist = len(data)-no_sexist
label_count = data[attributes].sum()

# split data
#X_train, X_test, y_train, y_test = train_test_split(
#    data, data['label_category'], test_size=0.33, random_state=42)

X_train, X_val_test, y_train, y_val_test = train_test_split(data, data['label_category'], test_size=0.3, random_state=1) # 70 % train, 15 % val, 15 % test data

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1) 

# inspect data in table
data.head(5)

from torch.utils.data import Dataset

class UCC_Dataset(Dataset): ########## WHY DO WE NEED FANCY DATASET MODULE?

  def __init__(self, data, tokenizer, attributes, max_token_len: int = 128, sample = None): ######## DEFINE HOW TO TRUNCATE SAMPLES,  THINK HOW MANY MAX TOKENS WE HAVE
    self.data = data
    self.tokenizer = tokenizer
    self.attributes = attributes
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self): 
    #data = pd.read_csv(self.data_path)
    #data['sexist'] = np.where(data['label_sexist'] == 'sexist', 1, 0)
    #labels = ['none','1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions'] ##### TURN THIS into an input variable of the function!!!
    #for k in labels:
    #  data[k] = np.where(train_data['label_category'] == k, 1, 0)

    if self.sample is not None:                            
      sexist = data.loc[data['sexist']==1]
      clean = data.loc[data['sexist']==0]
      self.data = pd.concat([sexist.sample(self.sample, random_state=7), clean.sample(self.sample, random_state=7)])
    #else:
    #  self.data = data
    
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


import torch
from transformers import AutoTokenizer

attributes = labels
#model_name = 'roberta-base' ################### connect with config
tokenizer = AutoTokenizer.from_pretrained(model_name) #,TOKENIZERS_PARALLELISM=True
ucc_ds = UCC_Dataset(X_train, tokenizer, attributes=attributes, sample = 300) 
ucc_ds_val = UCC_Dataset(X_test, tokenizer, attributes=attributes) 

# for testing:
#len(ucc_ds)
#ucc_ds.__len__()


import pytorch_lightning as pl

class UCC_Data_Module(pl.LightningDataModule):

  def __init__(self,model_name, X_train, X_test, attributes, batch_size: int = 16, max_token_length: int = 128): #model_name='roberta-base' ######ADDEDCONFIG
    super().__init__()
    #self.config = config   ####### ADDED BY ME         
    self.X_train = X_train
    self.X_test = X_test
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name) #model_name

  def setup(self, stage = None):
    if stage in (None, "fit"): 
      self.train_dataset = UCC_Dataset(self.X_train, attributes=self.attributes, tokenizer=self.tokenizer, sample=3000) ####### ADD SAMPLE PARAMETER
      self.val_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) ###### REMOVE SAMPLE PARAMETER
    if stage == 'test':
      self.test_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) 
    if stage == 'predict': ######### CAN WE DISTINGUISH BETWEEN PREDICT FOR VAL AND PREDICT FOR TEST????????????
      self.val_dataset = UCC_Dataset(self.X_test, attributes=self.attributes, tokenizer=self.tokenizer, sample=None) 

  def train_dataloader(self): ####### HERE ITS NICELY SEPERATED IN TRAIN, VAL, TEST -- WHY DIDNT WE DO THAT ABOVE IN SETUP????????
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True) 

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)


from torch.utils.data import DataLoader

ucc_data_module = UCC_Data_Module(model_name, X_train, X_test, attributes=attributes, batch_size=1) ######## ADDED CONFIG
ucc_data_module.setup()
ucc_data_module.train_dataloader()
len(ucc_data_module.train_dataloader()) ###### NUMBER OF BATCHES IN OUR TRAIN DATASET
len(X_train)

class UCC_Comment_Classifier(pl.LightningModule):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.classifier.weight) # makes quicker
    self.loss_func = nn.BCEWithLogitsLoss(reduction='mean') ####### WE JUST WANT CROSSENTROPY??????????
    self.dropout = nn.Dropout()
    self.f1_func = F1Score(task='multiclass', average='macro')
    
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
    # calculate loss
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels'])) ##### WHY AND HOW DO WE NEED TO MAKE SURE THAT WHAT IS OF THE SAME SHAPE??? THE LOGITS?
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

  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)


from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F

config = {
    'model_name': model_name,
    'n_labels': len(attributes),
    'batch_size': 1,                 ######## CHANGE
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 1           ####### CHANGE HIGHER - before 100
}

model = UCC_Comment_Classifier(config)


idx=0
input_ids = ucc_ds.__getitem__(idx)['input_ids']
attention_mask = ucc_ds.__getitem__(idx)['attention_mask']
labels = ucc_ds.__getitem__(idx)['labels']
model.cpu()
loss, output = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), labels.unsqueeze(dim=0))
print(labels.shape, output.shape, output)


# TRAIN MODEL

# datamodule
ucc_data_module = UCC_Data_Module(model_name, X_train, X_test, attributes=attributes, batch_size=config['batch_size'])
ucc_data_module.setup()

# model
model = UCC_Comment_Classifier(config)

# trainer and fit
trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
#trainer.fit(model, ucc_data_module)



%load_ext tensorboard
%tensorboard --logdir ./lightning_logs/



# PREDICT WITH MODEL
# method to convert list of comments into predictions for each comment
def classify_raw_comments(model, dm):
  predictions = trainer.predict(model, datamodule=dm)
  flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
  return flattened_predictions
  
y_pred = classify_raw_comments(model, ucc_data_module)
y_true =X_test.iloc[:,-4:]


y_pred



y_pred_int = np.argmax(np.max(y_pred, axis=1))
y_pred_int



#val_data = pd.read_csv(val_path)
#val_data['unhealthy'] = np.where(val_data['healthy'] == 1, 0, 1)
#val_data=X_test

#true_labels = attributes #np.array(val_data[attributes])



'''from sklearn import metrics
plt.figure(figsize=(15, 8))
for i, attribute in enumerate(attributes):
  fpr, tpr, _ = metrics.roc_curve(
      true_labels[:,i].astype(int), predictions[:, i])
  auc = metrics.roc_auc_score(
      true_labels[:,i].astype(int), predictions[:, i])
  plt.plot(fpr, tpr, label='%s %g' % (attribute, auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('RoBERTa Trained on UCC Datatset - AUC ROC')'''