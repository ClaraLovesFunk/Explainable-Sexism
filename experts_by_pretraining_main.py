from sklearn.model_selection import train_test_split 
from EDA import *
from experts_by_pretraining import *
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint




if __name__ == "__main__":

  data_path = 'data/train_all_tasks.csv'
  model_path = 'experts_by_pretraining_models'

  # store dict model_dict determining which pretrained lm to use
  model_dict = {
    'bert-base-uncased': 'BERT_base_uncased',
    'GroNLP/hateBERT': 'hateBERT', 
    #'roberta-large' : 'RoBERTa_large'#,
    #'microsoft/deberta-large': 'DeBERTa_large',
    }
  results = {}

  # LOOPING THROUGH MODELS
  results_by_model = {}
  for model_id in model_dict:
    

    # store variable bal determining whether training is class balanced or not
    # DO FOR BALANCING VS. NOT BALANCING
    bal = [True, False]
    results_by_balancing = {}
    for b in bal:
      

      model_name = model_dict[model_id]

      data, attributes = load_arrange_data(data_path)

      X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size = 0.2, random_state = 0) 
      
      full_expert_dm = Expert_DataModule(model_id, X_train, X_test, attributes=attributes, sample = b) ##### ADD SAMPLIGN PARAMETER HERE, IF NOT NONE AND ADD TO DATAMODULE
      full_expert_dm.setup()
      
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

      # define 
      full_expert = Expert_Classifier(config)

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
      
      # train 
      #trainer.fit(full_expert, full_expert_dm)

      # save 
      #torch.save(full_expert.state_dict(),f'{model_path}/{model_name}_bal_{b}.pt')
      
      # reload
      #full_expert = Expert_Classifier(config)
      #full_expert.eval()



      # DO FOR TRAINING VS. NO TRAINING
      training = [True, False]
      results_by_training = {}
      for t in training:

        # if we want to test a trained model, load our trained model
        if t == True:
          full_expert.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))
          full_expert.eval()

        # get predictions and turn to array
        y_pred_tensor = trainer.predict(full_expert, full_expert_dm)
        y_pred_arr = []
        for tensor in y_pred_tensor:
          y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
        y_pred = y_pred_arr

        # store results
        results_by_training[t] = [f1_score(y_test, y_pred, average="macro"), accuracy_score(y_test, y_pred)}']

      results_by_balancing[b] = results_by_training[t]
    results_by_model[model_dict[model_id]] = results_by_balancing[b]
        #print(f'untrained {model_id}')
        #print(f'f1 score {f1_score(y_test, y_pred, average="macro")}')
        #print(f'accuracy {accuracy_score(y_test, y_pred)}') 


'''

    ####################### BEFORE TRAINING
    y_pred_tensor = trainer.predict(full_expert, full_expert_dm)

    y_pred_arr = []
    for tensor in y_pred_tensor:
      y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
    
    y_pred = y_pred_arr

    label_map = {
            'none':0,
            '1. threats, plans to harm and incitement' : 1,
            '2. derogation': 2,
            '3. animosity': 3,
            '4. prejudiced discussions': 4
            }
    
    y_test.replace(label_map, inplace=True)
    print(f'untrained {model_id}')
    print(f'f1 score {f1_score(y_test, y_pred, average="macro")}')
    print(f'accuracy {accuracy_score(y_test, y_pred)}')



    ####################### AFTER TRAINING
    full_expert = Expert_Classifier(config)
    full_expert.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))
    full_expert.eval()
  
    # test 
    
    #trainer.test(model, ucc_data_module) ###### HOW DID WE DEFINE GOLD LABELS?
    
    #predict
    #y_pred = np.argmax(trainer.predict(model, ucc_data_module))

    y_pred_tensor = trainer.predict(full_expert, full_expert_dm)

    y_pred_arr = []
    for tensor in y_pred_tensor:
      y_pred_arr.extend(np.argmax(tensor.numpy(), axis = 1))
    
    y_pred = y_pred_arr

    print(f'trained {model_id}')
    print(f'f1 score {f1_score(y_test, y_pred, average="macro")}')
    print(f'accuracy {accuracy_score(y_test, y_pred)}')
    print(f'unqiue values: {np.unique(y_pred)}')
'''