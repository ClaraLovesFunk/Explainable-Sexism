from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

from EDA import *
from experts_modules import *
from new_master_modules import *



if __name__ == "__main__":

  #######################################################################################
  #####################################   FLAGS   #######################################
  #######################################################################################
  
  train_expert_flag = True

  #######################################################################################
  ############################   VALUES TO ITERATE OVER   ###############################
  #######################################################################################

  model_dict = {
    'GroNLP/hateBERT': 'hateBERT', 
    'bert-base-uncased': 'BERT_base_uncased',
    }
  train_balanced = [True, False]      

  #######################################################################################
  #####################################   HYPS   ########################################
  #######################################################################################

  data_path = 'data/train_all_tasks.csv'
  model_path = 'expert_models'

  results_by_fullExpert = {}
  for model_id in model_dict:
    
    results_by_balancing = {}
    for b in train_balanced: 
      
      # PREPARE DATA
      model_name = model_dict[model_id]

      data, attributes = load_arrange_data(data_path)

      X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
      
      full_expert_dm = Expert_DataModule(model_id, X_train, X_test, attributes=attributes, sample = b) 
      full_expert_dm.setup()
      
      # PREPARE MODELS
      config = {
        'model_name': model_id,
        'n_labels': len(attributes), 
        'batch_size': 2,                 
        'lr': 1.5e-6,     ########1.5e-6      
        'warmup': 0.2, 
        'train_size': len(full_expert_dm.train_dataloader()),
        'weight_decay': 0.001,
        'n_epochs': 1 ############20      
      }

      checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_top_k=1,
        monitor="val loss",
        filename=f'{model_name}_bal_{b}',
        )

      trainer = pl.Trainer(
        max_epochs=config['n_epochs'],                                  
        gpus=1, 
        num_sanity_val_steps=50,
        callbacks=[checkpoint_callback]
        )

      
      # TRAINING
      if train_expert_flag == True: 
        
        full_expert = Expert_Classifier(config)
        trainer.fit(full_expert, full_expert_dm)
        torch.save(full_expert.state_dict(),f'{model_path}/{model_name}_bal_{b}.pt')
      

      # TESTING

      full_expert = Expert_Classifier(config)  
      full_expert.load_state_dict(torch.load(f'{model_path}/{model_name}_bal_{b}.pt'))
      full_expert.eval()

      y_pred_tensor = trainer.predict(full_expert, full_expert_dm)
      y_pred_arr = []
      for tensor in y_pred_tensor:
        y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
      y_pred = y_pred_arr

      perf_metrics = {
        'f1-macro_avrg': f1_score(y_test, y_pred, average="macro"),
        'f1-no_avrg': f1_score(y_test, y_pred, average=None),
        'acc': accuracy_score(y_test, y_pred), 
        }

      results_by_balancing[b] = perf_metrics
    results_by_fullExpert[model_dict[model_id]] = results_by_balancing
  np.save('results.npy', results_by_fullExpert) 
  results = np.load('results.npy',allow_pickle='TRUE').item()
  
  for model_id in model_dict: 
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print(f'     {model_name}')
    print(f'\n')
    for b in train_balanced:
      print('-----------------------------------------------------------')
      print(f'bal-{b}:')
      print(results[model_dict[model_id]][b]['f1-macro_avrg']) #f'f1: ', 
      print(results[model_dict[model_id]][b]['f1-no_avrg']) #f'f1: ', 
      print(results[model_dict[model_id]][b]['acc']) #f'acc: ', 
      print('\n')

  




  ########################################################################
  ######################## SLAPPING THE MASTER MAIN IN####################
  ########################################################################




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

  expert_id = list(expert_info.keys()) 
  expert_name = list(expert_info.values()) 

  data, attributes = load_arrange_data(data_path)

  X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
  
  master_dm = Master_DataModule(expert_id[0], X_train, X_test, attributes=attributes, sample = exp_bal_train) ###### MAKE EXPERT MODULE1 AND 2
  master_dm.setup()



  # LOADING EXPERTS AND CUTTING OFF HIDDEN LAYER AND CLASSIFICATION HEAD

  config_expert = {
    'model_name': expert_id[0],            ###### CURRENTLY STILL NECESSARY FOR EXPERT MODULE
    'model_name1': expert_id[1],
    'experts': expert_id,
    'n_labels': len(attributes), 
    'batch_size': 2,                 
    'lr': 1.5e-6,           
    'warmup': 0.2, 
    'train_size': len(master_dm.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 20      
  }

  full_experts = [] 
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


  
  # PREPARE MODELS
  config_master = {
    'experts': experts,
    'n_labels': len(attributes),
    'batch_size': 2,                 
    'lr': 1.5e-1, ########1.5e-6
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

    np.save('results_master.npy', perf_metrics) 
    results = np.load('results_master.npy',allow_pickle='TRUE').item()
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print(f'f1-macro_avrg: {results["f1-macro_avrg"]}')
    print(f'f1-no_avrg: {results["f1-no_avrg"]}')
    print(f'acc: {results["acc"]}')