
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from EDA import *
from experts_by_pretraining import *#UCC_Dataset, UCC_Comment_Classifier



if __name__ == "__main__":

  data_path = 'data/train_all_tasks.csv'
  model_name =  'GroNLP/hateBERT' ####['GroNLP/hateBERT', 'distilroberta-base', 'microsoft/deberta-large', 'bert-base-cased'] ####### WHICH BERT ARCHITECTURE IS HATEBERT USING?????
  model_path = 'experts_by_pretraining_models'
  whatevername = 'Klaus' ###########!!!!!

  data, attributes = load_arrange_data(data_path)
  X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size=0.01, random_state=0) #########0.2
  
  ucc_data_module = Expert_DataModule(model_name, X_train, X_test, attributes=attributes) ######## ADDED CONFIG , batch_size=1
  ucc_data_module.setup()
  
  config = {
    'model_name': model_name,
    'n_labels': len(attributes), ########l
    'batch_size': 2,                 ######## CHANGE
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 1      ###########25     
  }

  # define 
  model = Expert_Classifier(config)
  trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
  
  # train 
  #trainer.fit(model, ucc_data_module)
  
  # test 
  trainer.test(model, ucc_data_module)

  # save 
  torch.save(model.state_dict(),f'{model_path}/{whatevername}.pt')
  
  # reload
  model = Expert_Classifier(config)
  model.load_state_dict(torch.load(f'{model_path}/{whatevername}.pt'))

