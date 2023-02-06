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
        'batch_size': 1,                 
        'lr': 1.5e-6,          
        'warmup': 0.2, 
        'train_size': len(full_expert_dm.train_dataloader()),
        'weight_decay': 0.001,
        'n_epochs': 20      
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