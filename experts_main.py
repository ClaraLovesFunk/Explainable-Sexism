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
  
  train_expert_flag = False
  test_submission_flag = True

  #######################################################################################
  ############################   VALUES TO ITERATE OVER   ###############################
  #######################################################################################

  model_dict = {
    'GroNLP/hateBERT': 'hateBERT', 
    #'bert-base-uncased': 'BERT_base_uncased',
    }
  use_trained_model = [True] 
  train_balanced = [True]      ###################### CHANGE

  #######################################################################################
  #####################################   HYPS   ########################################
  #######################################################################################

  data_path = 'data/test_task_b_entries.csv'#'data/train_all_tasks.csv'
  test_data_path = 'data/test_task_b_entries.csv'
  model_path = 'experts_by_pretraining_models'

  results_by_fullExpert = {}
  for model_id in model_dict:
    
    results_by_balancing = {}
    for b in train_balanced: 
      
      # PREPARE DATA
      model_name = model_dict[model_id]

      data, attributes = load_arrange_data(data_path,test_submission_flag)

      if test_submission_flag == True:
        X_train = data
        X_test = data
        y_test = data['label_category']
        y_train = data['label_category']
      else:
        X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
      
      full_expert_dm = Expert_DataModule(model_id, X_train, X_test, attributes=attributes, sample = False) #######REPLACE FALSE WITH B
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
        'n_epochs': 10 #10     
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


        '''# compute performance
        perf_metrics = {
          'f1': f1_score(y_test, y_pred, average="macro"),
          'acc': accuracy_score(y_test, y_pred), 
          }

        # store performance
        results_by_training[t] = perf_metrics
      results_by_balancing[b] = results_by_training
    results_by_fullExpert[model_dict[model_id]] = results_by_balancing

  np.save('results.npy', results_by_fullExpert) 

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
        print('\n')'''

  # MAKE SEMEVAL SUBMISSION FILE
  
  X_test['label_pred_int'] =  y_pred

  label_map = {                                  
    0: "1. threats, plans to harm and incitement",
    1: "2. derogation",
    2: "3. animosity",
    3: "4. prejudiced discussions",
    }
  #X_test['label_pred'].replace(label_map, inplace=True) 
  '''X_test['label_pred'] = X_test['rewire_id']
  for x in range(len(X_test)):
    for i in range(4):
      X_test['label_pred'][x]=label_map[i]
  '''
  X_test.drop(['text', 'label_category'], inplace=True, axis=1) #, 'label_pred_int'
  X_test.drop(attributes, inplace=True, axis=1)

  X_test.to_csv('data/test_task_b_entries_pred.csv',index=False)
  pred = pd.read_csv('data/test_task_b_entries_pred.csv')
  #display(pred)