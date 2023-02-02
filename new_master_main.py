from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

from EDA import *
from new_master_modules import *



if __name__ == "__main__":

  #######################################################################################
  #####################################   FLAGS   #######################################
  #######################################################################################
  
  train_master_flag = True
  test_master_flag = True
  
  train_expert_flag = False
  balance_classes = True

  #######################################################################################
  ############################   VALUES TO ITERATE OVER   ###############################
  #######################################################################################

  model_dict = {
    'GroNLP/hateBERT': 'hateBERT', 
    'bert-base-uncased': 'BERT_base_uncased',
    }
  
  train_balanced = [True]

  #######################################################################################
  #####################################   HYPS   ########################################
  #######################################################################################

  data_path = 'data/train_all_tasks.csv'
  model_path = 'experts_by_pretraining_models'

  # PREPARE DATA
  expert0_id = list(model_dict.keys())[0]
  expert1_id = list(model_dict.keys())[1]
  expert0_name = model_dict[expert0_id]
  expert1_name = model_dict[expert1_id]

  data, attributes = load_arrange_data(data_path)

  X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
  
  master_dm = Master_DataModule(expert0_id, X_train, X_test, attributes=attributes, sample = balance_classes) ###### MAKE EXPERT MODULE1 AND 2
  master_dm.setup()
  
  # PREPARE MODELS
  config = {
    'model_name': expert0_id, ##### THE GENERIC MODEL IS LOADED FOR FINETUNING -- SUBSITUTE WITH OUR OWN FINETUNED MODELS
    'n_labels': len(attributes), 
    'batch_size': 2,                 
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(master_dm.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 1          #######20     
  }

  checkpoint_callback = ModelCheckpoint(
    dirpath=model_path,
    save_top_k=1,
    monitor="val loss",
    filename=f'master',
    )

  trainer = pl.Trainer(
    max_epochs=config['n_epochs'], 
    gpus=1, 
    num_sanity_val_steps=50,
    callbacks=[checkpoint_callback]
    )
  
  # TRAINING
  if train_master_flag == True: 
    master_clf = Master_Classifier(config)
    trainer.fit(master_clf, master_dm)
    torch.save(master_clf.state_dict(),f'{model_path}/master.pt')
  

  # TESTING
  if test_master_flag == True:   
    master_clf = Master_Classifier(config)                                          
    master_clf.load_state_dict(torch.load(f'{model_path}/master.pt')) 
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