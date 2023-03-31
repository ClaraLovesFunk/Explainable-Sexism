from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import set_seed 

from EDA import *
from pretraining_experts_modules import *
from pretraining_master_modules import * 



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
  train_balanced = [True] 
  seeds = [0,1]      

  #######################################################################################
  #####################################   HYPS   ########################################
  #######################################################################################

  data_path = 'data/edos_labelled_individual_annotations.csv'
  model_path = 'expert_models'

  results_by_fullExpert = {}
  for model_id in model_dict:
    
    results_by_balancing = {}
    for b in train_balanced: 

      results_by_seed = {}
      for s in seeds: 
        
        # PREPARE DATA
        model_name = model_dict[model_id]

        data, attributes = load_arrange_data(data_path)
        
        X_train = data.loc[(data['split'] != 'test')]
        y_train = data.loc[(data['split'] != 'test')]['label_category']

        X_test = data.loc[(data['split'] == 'test')]
        y_test = data.loc[(data['split'] == 'test')]['label_category']

        full_expert_dm = Expert_DataModule(model_id, X_train, X_test, attributes=attributes, sample = b) 
        full_expert_dm.setup()
        
        # PREPARE MODELS
        config = {
          'model_name': model_id,
          'n_labels': len(attributes), 
          'batch_size': 1,                 
          'lr': 1.5e-6,          
          'warmup': 0.2, 
          'train_size': len(full_expert_dm.train_dataloader()),
          'weight_decay': 0.001,
          'n_epochs': 50      
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
          )

        
        # TRAINING
        if train_expert_flag == True: 
          
          full_expert = Expert_Classifier(config, seed = s)
          trainer.fit(full_expert, full_expert_dm)
          torch.save(full_expert.state_dict(),f'{model_path}/{model_name}_bal_{b}_seed-{s}.pt')
        

        # TESTING

        full_expert = Expert_Classifier(config)  
        full_expert.load_state_dict(torch.load(f'{model_path}/{model_name}_bal_{b}_seed-{s}.pt'))
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
        results_by_seed[s] = perf_metrics
      results_by_balancing[b] = results_by_seed
    results_by_fullExpert[model_dict[model_id]] = results_by_balancing
  np.save('results_experts.npy', results_by_fullExpert) 
  results = np.load('results_experts.npy',allow_pickle='TRUE').item()
  
  for model_id in model_dict: 
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print(f'     {model_dict[model_id]}')
    print(f'\n')
    for b in train_balanced:
      print(f'bal-{b}:')

      for s in seeds:
        print('-----------------------------------------------------------')
        print(f'seed-{s}:')
        print(results[model_dict[model_id]][b][s]['f1-macro_avrg'])  
        print(results[model_dict[model_id]][b][s]['f1-no_avrg'])  
        print(results[model_dict[model_id]][b][s]['acc']) 
        print('\n')