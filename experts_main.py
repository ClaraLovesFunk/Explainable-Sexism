from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

from EDA import *
from experts_modules import *



if __name__ == "__main__":

  #######################################################################################
  #####################################   FLAGS   #######################################
  #######################################################################################
  
  train_flag = False

  #######################################################################################
  ############################   VALUES TO ITERATE OVER   ###############################
  #######################################################################################

  model_dict = {
    'GroNLP/hateBERT': 'hateBERT', 
    'bert-base-uncased': 'BERT_base_uncased',
    }
  use_trained_model = [True, False]
  train_balanced = [True, False]

  #######################################################################################
  #####################################   HYPS   ########################################
  #######################################################################################

  data_path = 'data/train_all_tasks.csv'
  model_path = 'experts_by_pretraining_models'

 

  results_by_model = {}
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
      if train_flag == True: 
        full_expert = Expert_Classifier(config)
        trainer.fit(full_expert, full_expert_dm)
        torch.save(full_expert.state_dict(),f'{model_path}/{model_name}_bal_{b}.pt')
      

      # TESTING

      # test trained vs. untrained models
      results_by_training = {}
      for t in use_trained_model:

        full_expert = Expert_Classifier(config)        ############### DOE WE NEED EVAL HERE ALSO????                  
        
        if t == True:                                                    
          full_expert.load_state_dict(torch.load(f'{model_path}/{model_name}_bal_{b}.pt'))
          full_expert.eval()

        # get predictions and turn to array
        y_pred_tensor = trainer.predict(full_expert, full_expert_dm)
        y_pred_arr = []
        for tensor in y_pred_tensor:
          y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
        y_pred = y_pred_arr

        # compute performance
        perf_metrics = {
          'f1': f1_score(y_test, y_pred, average="macro"),
          'acc': accuracy_score(y_test, y_pred), 
          }

        # store performance
        results_by_training[t] = perf_metrics
      results_by_balancing[b] = results_by_training
    results_by_model[model_dict[model_id]] = results_by_balancing

  np.save('results.npy', results_by_model) 

  results = np.load('results.npy',allow_pickle='TRUE').item()
  
  for model_id in model_dict: 
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print(f'     {model_name}')
    print(f'\n')
    for b in train_balanced:
      for t in use_trained_model:

        print('-----------------------------------------------------------')
        print(f'bal-{b}_trained-{t}:')
        print(results[model_dict[model_id]][b][t]['f1']) #f'f1: ', 
        print(results[model_dict[model_id]][b][t]['acc']) #f'acc: ', 
        print('\n')