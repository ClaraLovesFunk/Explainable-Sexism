from sklearn.model_selection import train_test_split 
from EDA import *
from experts_by_pretraining import *
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score



if __name__ == "__main__":

  data_path = 'data/train_all_tasks.csv'
  model_path = 'experts_by_pretraining_models'
  model_dict = {
    #'bert-base-uncased': 'BERT_base_uncased',
    'GroNLP/hateBERT': 'hateBERT', 
    #'roberta-large' : 'RoBERTa_large'#,
    #'microsoft/deberta-large': 'DeBERTa_large',
    }

  for model_id in model_dict:

    model_name = model_dict[model_id]

    data, attributes = load_arrange_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size=0.2, random_state=0) 
    
    ucc_data_module = Expert_DataModule(model_id, X_train, X_test, attributes=attributes) ##### ADD SAMPLIGN PARAMETER HERE, IF NOT NONE AND ADD TO DATAMODULE
    ucc_data_module.setup()
    
    config = {
      'model_name': model_id,
      'n_labels': len(attributes), 
      'batch_size': 2,                 
      'lr': 1.5e-6,
      'warmup': 0.2, 
      'train_size': len(ucc_data_module.train_dataloader()),
      'weight_decay': 0.001,
      'n_epochs': 25     
    }

    # define 
    #model = Expert_Classifier(config)
    trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
    
    # train 
    #trainer.fit(model, ucc_data_module)

    # save 
    #torch.save(model.state_dict(),f'{model_path}/{model_name}.pt')
    
    # reload

    model = Expert_Classifier(config)
    model.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))
    model.eval()
  
    # test 
    
    #trainer.test(model, ucc_data_module) ###### HOW DID WE DEFINE GOLD LABELS?
    
    #predict
    #y_pred = np.argmax(trainer.predict(model, ucc_data_module))

    y_pred_tensor = trainer.predict(model, ucc_data_module)
    #print(y_pred_tensor)

    
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
    print(f'trained {model_id}')
    print(f'f1 score {f1_score(y_test, y_pred, average="macro")}')
    print(f'accuracy {accuracy_score(y_test, y_pred)}')
