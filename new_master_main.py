from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

from EDA import *
from new_master_modules import *
from experts_modules import *



if __name__ == "__main__":

  train_master_flag = True
  test_master_flag = True
  exp_bal_train = True

  expert_info = {
    'GroNLP/hateBERT': 'hateBERT', 
    'bert-base-uncased': 'BERT_base_uncased',
    }
  
  data_path = 'data/train_all_tasks.csv'
  expert_model_path = 'expert_models'
  master_model_path = 'master_models'

  # PREPARE DATA
  expert_id = list(expert_info.keys())#['GroNLP/hateBERT','bert-base-uncased'] 
  expert_name = list(expert_info.values()) #[expert_info[expert_id[0]], expert_info[expert_id[0]]]

  data, attributes = load_arrange_data(data_path)

  X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
  
  master_dm = Master_DataModule(expert_id[0], X_train, X_test, attributes=attributes, sample = exp_bal_train) ###### MAKE EXPERT MODULE1 AND 2
  master_dm.setup()






  # LOADING EXPERTS AND CUTTING OFF HIDDEN LAYER AND CLASSIFICATION HEAD

  config_expert = {
    'model_name': expert_id[0],
    'model_name1': expert_id[1],
    'experts': expert_id,
    'n_labels': len(attributes), 
    'batch_size': 2,                 
    'lr': 1.5e-3,           #######1.5e-6
    'warmup': 0.2, 
    'train_size': len(master_dm.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 1      
  }

  
  full_experts = [] #[Expert_Classifier(config_expert), Expert_Classifier(config_expert)]
  experts = []
  finetuned_dict = []
  model_dict = []

  for i in range(2):
    full_experts.append(Expert_Classifier(config_expert))
    full_experts[i] = Expert_Classifier(config_expert) 
    full_experts[i].load_state_dict(torch.load(f'{expert_model_path}/{expert_name[i]}_bal_{exp_bal_train}.pt'))

    experts.append(AutoModel.from_pretrained(expert_id[i]))

    finetuned_dict.append(full_experts[i].state_dict())

    model_dict.append(experts[i].state_dict())

    finetuned_dict[i] = {k: v for k, v in finetuned_dict[i].items() if k in model_dict[i]}

    model_dict[i].update(finetuned_dict[i])

    experts[i].load_state_dict(model_dict[i])

    experts[i].eval()


  #full_experts[0] = Expert_Classifier(config_expert) 
  #full_experts[1] = Expert_Classifier(config_expert) 
                                        
  #full_experts[0].load_state_dict(torch.load(f'{expert_model_path}/{expert_name[0]}_bal_{exp_bal_train}.pt'))
  #full_experts[1].load_state_dict(torch.load(f'{expert_model_path}/{expert_name[1]}_bal_{exp_bal_train}.pt')) 
 
  #full_experts, configs = train_experts(df, model_name, doTrain=True, doTest=True)

  #expert = AutoModel.from_pretrained(expert_id[0]) 
  #expert1 = AutoModel.from_pretrained(expert_id[1]) 

  #finetuned_dict = full_experts[0].state_dict()
  #finetuned_dict1 = full_experts[1].state_dict()

  #expert_info = expert.state_dict()
  #model_dict1 = expert1.state_dict()
  
  # 1. filter out unnecessary keys
  #finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in expert_info}
  #finetuned_dict1 = {k: v for k, v in finetuned_dict1.items() if k in model_dict1}

  # 2. overwrite entries in the existing state dict
  #expert_info.update(finetuned_dict) 
  #model_dict1.update(finetuned_dict1) 

  # 3. load the new state dict
  #expert.load_state_dict(expert_info)
  #expert1.load_state_dict(model_dict1)

  # freeze the weights for the master model
  #expert.eval()
  #expert1.eval()

  #experts = [expert,expert1]




  
  # PREPARE MODELS
  config_master = {
    'experts': experts,
    'n_labels': len(attributes),
    'batch_size': 2,                 
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(master_dm.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 1          #######20     
  }

  checkpoint_callback = ModelCheckpoint(
    dirpath=expert_model_path,
    save_top_k=1,
    monitor="val loss",
    filename=f'master',
    )

  trainer = pl.Trainer(
    max_epochs=config_master['n_epochs'], 
    gpus=1, 
    num_sanity_val_steps=50,
    callbacks=[checkpoint_callback]
    )
  

















  # TRAINING
  if train_master_flag == True: 
    master_clf = Master_Classifier(config_master,config_expert)
    trainer.fit(master_clf, master_dm)
    torch.save(master_clf.state_dict(),f'{master_model_path}/master-{expert_name[0]}-{expert_name[1]}-bal_{exp_bal_train}.pt')
  

  # TESTING
  if test_master_flag == True:   
    master_clf = Master_Classifier(config_master,config_expert)                                          
    master_clf.load_state_dict(torch.load(f'{master_model_path}/master-{expert_name[0]}-{expert_name[1]}-bal_{exp_bal_train}.pt')) 
    master_clf.eval()

    # get predictions and turn to array
    y_pred_tensor = trainer.predict(master_clf, master_dm)
    y_pred_arr = []
    for tensor in y_pred_tensor:
      y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
    y_pred = y_pred_arr

    # compute performance
    perf_metrics = {
      'f1-macro_avrg': f1_score(y_test, y_pred, average="macro"),
      'f1-no_avrg': f1_score(y_test, y_pred, average=None),
      'acc': accuracy_score(y_test, y_pred), 
      }

    np.save('results.npy', perf_metrics) 
    results = np.load('results.npy',allow_pickle='TRUE').item()
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print(f'f1-macro_avrg: {results["f1-macro_avrg"]}')
    print(f'f1-no_avrg: {results["f1-no_avrg"]}')
    print(f'acc: {results["acc"]}')