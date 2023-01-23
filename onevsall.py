import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers import AutoModel
from onevsall_expert import ExpertDataModule, ExpertClassifier

def train_experts(df, isTrain=True):
    expert_config = {}
    for label in range(len(label_map)):
        new_map = {i:0 for i in range(len(label_map))}
        new_map[label] = 1
        binary_df = df.copy(deep=True)
        binary_df['label'].replace(new_map, inplace=True)
        
        df_train, df_test = train_test_split(df, test_size=0.2)
        dm = ExpertDataModule(df_train, df_test)
        dm.setup()

        expert_config = {
            'model_name': "microsoft/deberta-large",
            'n_labels': 2,
            'batch_size': 128,
            'lr': 1.5e-6,
            'warmup': 0.2, 
            'train_size': len(dm.train_dataloader()),
            'weight_decay': 0.001,
            'n_epochs': 1
            }
        if not isTrain:
            return expert_config
        else:
            expert_model = ExpertClassifier(expert_config)

            trainer = pl.Trainer(
                    max_epochs=expert_config['n_epochs'],
                    accelerator='gpu',
                    devices=1,
                    default_root_dir="project/Explainable-Sexism/"
                    )
            trainer.fit(expert_model, dm)
            torch.save(expert_model.pretrained_model.state_dict(), "project/Explainable-Sexism/model.pt")
            return expert_model
        break

    return expert_config


if __name__ == "__main__":
    df = pd.read_csv('project/Explainable-Sexism/data/train_all_tasks.csv')
    label_map = {
            'none':0,
            '1. threats, plans to harm and incitement' : 1,
            '2. derogation': 2,
            '3. animosity': 3,
            '4. prejudiced discussions': 4
            }
    df.drop(['rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)    
    df['label_category'].replace(label_map, inplace=True)
    df.rename(columns={'label_category':'label'}, inplace=True)

    expert_model = train_experts(df, isTrain=True)
    expert_dict = expert_model.pretrained_model.state_dict()
    expert = expert_dict['encoder.layer.23.output.dense.weight'].numpy()
    

    new_model = AutoModel.from_pretrained("microsoft/deberta-large", return_dict = True)
    model_dict = new_model.state_dict() #new model keys

    new_model.load_state_dict(torch.load("project/Explainable-Sexism/model.pt"))


    pretrained_dict = torch.load("project/Explainable-Sexism/model.pt")

    before = model_dict['encoder.layer.23.output.dense.weight'].numpy()
    after = pretrained_dict['encoder.layer.23.output.dense.weight'].numpy()
    print(np.array_equal(before, expert))
    print(before == expert)
    







