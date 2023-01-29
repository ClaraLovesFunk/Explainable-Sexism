from sklearn.model_selection import train_test_split ######### sample method!!!!!!
from EDA import *
from experts_by_pretraining import *



if __name__ == "__main__":

  data_path = 'data/train_all_tasks.csv'
  model_path = 'experts_by_pretraining_models'
  model_dict = {
    'bert-base-uncased': 'BERT_base_uncased',
    'GroNLP/hateBERT': 'hateBERT',
    #'roberta-large' : 'RoBERTa_large',
    #'microsoft/deberta-large': 'DeBERTa_large',
    }

  for model_id in model_dict:

    model_name = model_dict[model_id]

    data, attributes = load_arrange_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data, data['label_category'], test_size=0.01, random_state=0) ######### test_size 0.2
    
    ucc_data_module = Expert_DataModule(model_id, X_train, X_test, attributes=attributes) ######## attributes
    ucc_data_module.setup()
    
    config = {
      'model_name': model_id,
      'n_labels': len(attributes), ########l
      'batch_size': 1,                 ######## CHANGE
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
    trainer.fit(model, ucc_data_module)
    
    # test 
    trainer.test(model, ucc_data_module)

    # save 
    torch.save(model.state_dict(),f'{model_path}/{model_name}.pt')
    
    # reload
    model = Expert_Classifier(config)
    model.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))