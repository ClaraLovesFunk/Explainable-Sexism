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
from torchmetrics.classification import MulticlassF1Score


model_name =  'GroNLP/hateBERT' #'distilroberta-base' 





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





tokenizer = AutoTokenizer.from_pretrained(model_name) #,TOKENIZERS_PARALLELISM=True
ucc_ds = UCC_Dataset(X_train, tokenizer, attributes=attributes, sample = 3) 
ucc_ds_val = UCC_Dataset(X_test, tokenizer, attributes=attributes) 





ucc_data_module = UCC_Data_Module(model_name, X_train, X_test, attributes=attributes, batch_size=1) ######## ADDED CONFIG
ucc_data_module.setup()
ucc_data_module.train_dataloader()
len(ucc_data_module.train_dataloader()) ###### NUMBER OF BATCHES IN OUR TRAIN DATASET
len(X_train)





config = {
    'model_name': model_name,
    'n_labels': len(attributes), ########l
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
loss, f1, output = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), labels.unsqueeze(dim=0))
print(labels.shape, output.shape, output)


# TRAIN MODEL

# datamodule
ucc_data_module = UCC_Data_Module(model_name, X_train, X_test, attributes=attributes, batch_size=config['batch_size'])
ucc_data_module.setup()

# model
model = UCC_Comment_Classifier(config)

# trainer and fit
trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
trainer.fit(model, ucc_data_module)



#%load_ext tensorboard ######## OPEN ANOTHER TERMINAL
#%tensorboard --logdir ./lightning_logs/



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