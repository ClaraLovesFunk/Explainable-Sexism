from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import set_seed 

from EDA import *
from pretraining_master_modules import *
from pretraining_experts_modules import *



if __name__ == "__main__":

  train_master_flag = True
  test_master_flag = True
  exp_bal_train = True

  expert_info = {
    'GroNLP/hateBERT': 'hateBERT', 
    'bert-base-uncased': 'BERT_base_uncased',
    }
  
  data_path = 'data/edos_labelled_individual_annotations.csv'        
  expert_model_path = 'expert_models'
  master_model_path = 'master_models'
  seed_master = 0
  seed_experts = [0,0]

  for m in range(2):

    
    if m == 0:
      expert_id = ['GroNLP/hateBERT', 'GroNLP/hateBERT']
      expert_name = ['hateBERT', 'hateBERT'] 

    else:
      expert_id = ['bert-base-uncased', 'bert-base-uncased']
      expert_name = ['BERT_base_uncased', 'BERT_base_uncased'] 

    # PREPARE DATA
    data, attributes = load_arrange_data(data_path)

    X_train = data.loc[(data['split'] != 'test')]
    y_train = data.loc[(data['split'] != 'test')]['label_category']

    X_test = data.loc[(data['split'] == 'test')]
    y_test = data.loc[(data['split'] == 'test')]['label_category']

    master_dm = Master_DataModule(expert_id[0], X_train, X_test, attributes=attributes, sample = exp_bal_train) 
    master_dm.setup()



    # LOADING EXPERTS AND CUTTING OFF HIDDEN LAYER AND CLASSIFICATION HEAD

    config_expert = {
      'model_name': expert_id[0],            
      'model_name1': expert_id[1],
      'experts': expert_id,
      'n_labels': len(attributes), 
      'batch_size': 1,                 
      'lr': 1.5e-6,        
      'warmup': 0.2, 
      'train_size': len(master_dm.train_dataloader()),
      'weight_decay': 0.001,
      'n_epochs': 50      
    }

    full_experts = [] 
    experts = []
    finetuned_dict = []
    model_dict = []

    for i in range(2):
      full_experts.append(Expert_Classifier(config_expert))
      full_experts[i] = Expert_Classifier(config_expert) 

      full_experts[i].load_state_dict(torch.load(f'{expert_model_path}/{expert_name[i]}_bal_{exp_bal_train}_seed-{i}.pt')) 

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
      'batch_size': 1,                 
      'lr': 1.5e-6,
      'warmup': 0.2, 
      'train_size': len(master_dm.train_dataloader()),
      'weight_decay': 0.001,
      'n_epochs': 50     
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
      )
    


    # TRAINING
    if train_master_flag == True: 
      master_clf = Master_Classifier(config_master,config_expert)
      trainer.fit(master_clf, master_dm)
      torch.save(master_clf.state_dict(),f'{master_model_path}/master-{expert_name[0]}_seed-{seed_experts[0]}-{expert_name[1]}_seed-{seed_experts[1]}-bal_{exp_bal_train}_seed-{seed_master}.pt')
    


    # TESTING
    if test_master_flag == True:   
      master_clf = Master_Classifier(config_master,config_expert)                                          
      master_clf.load_state_dict(torch.load(f'{master_model_path}/master-{expert_name[0]}_seed-{seed_experts[0]}-{expert_name[1]}_seed-{seed_experts[1]}-bal_{exp_bal_train}_seed-{seed_master}.pt')) 
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
      
      if m == 0:
        results_name = 'hh'
      else:
        results_name = 'bb'
      np.save(f'results_master_{results_name}.npy', perf_metrics) 
      results = np.load(f'results_master_{results_name}.npy',allow_pickle='TRUE').item()
      print('-----------------------------------------------------------')
      print('-----------------------------------------------------------')
      print(f'f1-macro_avrg: {results["f1-macro_avrg"]}')
      print(f'f1-no_avrg: {results["f1-no_avrg"]}')
      print(f'acc: {results["acc"]}')