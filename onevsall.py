import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from onevsall_expert import ExpertDataModule, ExpertClassifier

def train_experts(df, model_name, num_classes=5, doTrain=False, doTest=False):
    expert_configs = []
    expert_models = []
    for label in range(num_classes):
        new_map = {i:0 for i in range(num_classes)}
        new_map[label] = 1
        binary_df = df.copy(deep=True)
        binary_df['label'].replace(new_map, inplace=True)
        
        df_train, df_test = train_test_split(binary_df, test_size=0.2)
        dm = ExpertDataModule(df_train, df_test, balance=True)
        dm.setup()
        
        expert_config = {
            'model_name': model_name,
            'n_labels': 2,
            'batch_size': 128,
            'lr': 1.5e-6,
            'warmup': 0.2, 
            'train_size': len(dm.train_dataloader()),
            'weight_decay': 0.001,
            'n_epochs': 1,
            }
        expert_configs.append(expert_config)
        expert_model = ExpertClassifier(expert_config)
        
        checkpoint_callback = ModelCheckpoint(
                dirpath="project/Explainable-Sexism/onevsall_models",
                save_top_k=1,
                monitor="val f1",
                filename=f"balanced_expert_{label}",
                )
        trainer = pl.Trainer(
                max_epochs=expert_config['n_epochs'],
                accelerator='gpu',
                devices=1,
                num_sanity_val_steps=15,
                default_root_dir="project/Explainable-Sexism/",
                callbacks=[checkpoint_callback],
                )

        if not doTrain:
            ckpt = torch.load(
                  f"project/Explainable-Sexism/onevsall_models/balanced_expert_{label}.ckpt"
                    )
            expert_model.load_state_dict(ckpt['state_dict'])
            expert_models.append(expert_model)
        else:
            expert_models.append(expert_model)
            trainer.fit(expert_model, dm)

        if doTest:
            expert_model.eval()
            trainer.test(expert_model, dm)

    return expert_models, expert_configs

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

    print(df['label'].value_counts())

    model_name = "microsoft/deberta-large"

    full_experts, configs = train_experts(df, model_name, doTrain=True, doTest=False)

    experts = [ExpertClassifier(config) for config in configs]

    for i in range(len(label_map)):
        finetuned_dict = full_experts[i].state_dict()
        model_dict = experts[i].state_dict()
        
        # 1. filter out unnecessary keys
        finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(finetuned_dict) 
        # 3. load the new state dict
        experts[i].load_state_dict(model_dict)
        # freeze the weights for the master model
        experts[i].eval()
        






