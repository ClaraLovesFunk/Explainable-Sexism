

import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

from EDA import *
from experts_modules import *


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, len(num_layers_to_keep)):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    return copyOfModel


config = {
'model_name': 'bert-base-uncased',
'n_labels': 4,#len(attributes), 
'batch_size': 2,                 
'lr': 1.5e-3,           #######1.5e-6
'warmup': 0.2, 
'train_size': 10,#len(full_expert_dm.train_dataloader()),
'weight_decay': 0.001,
'n_epochs': 1      
}


full_expert = Expert_Classifier(config)  
full_expert.load_state_dict(torch.load(f'experts_by_pretraining_models/BERT_base_uncased_bal_True.pt'))
#full_expert.eval()

copyOfModel = deleteEncodingLayers(full_expert, 11)





'''    full_experts, configs = train_experts(df, model_name, doTrain=True, doTest=True)

    experts = [AutoModel.from_pretrained(model_name) for _ in range(len(label_map))]

    for i in range(len(label_map)):
        finetuned_dict = full_experts[i].state_dict()
        model_dict = experts[i].state_dict()
        
        # 1. filter out unnecessary keys
        finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(finetuned_dict) 
        # 3. load the new state dict
        experts[i].load_state_dict(model_dict)
        # freeze the weights for the master model
        experts[i].eval()'''