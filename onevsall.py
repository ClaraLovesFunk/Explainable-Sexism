import torch
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from onevsall_expert import ExpertDataModule, ExpertClassifier
from onevsall_master import MasterDataModule, MasterClassifier


def train_master(df, model_name, experts, num_classes=5, doTrain=False, doTest=False):
    save_dir = "project/Explainable-Sexism/onevsall_models"
    model_filename="master"

    df_train, df_test = train_test_split(df, test_size=0.2)
    dm = MasterDataModule(df_train, df_test, model_name=model_name)
    dm.setup()
        
    config = {
            'model_name': model_name,
            'n_labels': num_classes,
            'experts': experts,
            'batch_size': 8,
            'lr': 1e-3,
            'warmup': 0.2, 
            'train_size': len(dm.train_dataloader()),
            'weight_decay': 0.001,
            'n_epochs': 15,
        }
    model = MasterClassifier(config)
        
    checkpoint_callback = ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=1,
                monitor="val f1",
                filename=model_filename
                )
    trainer = pl.Trainer(
                max_epochs=config['n_epochs'],
                accelerator='gpu',
                devices=1,
                num_sanity_val_steps=15,
                default_root_dir="project/Explainable-Sexism/",
                callbacks=[checkpoint_callback],
                )

    if not doTrain:
        ckpt = torch.load(
                  f"{save_dir}/{model_filename}.ckpt"
                    )
        model.load_state_dict(ckpt['state_dict'])
    else:
        trainer.fit(model, dm)
        torch.save(
                model.state_dict(),
                f"{save_dir}/{model_filename}.pt"
                )

    if doTest:
        model.eval()
        test_dataloader = dm.test_dataloader()
        trainer.test(model, dataloaders=test_dataloader)

    return model

def train_experts(df, model_name, num_classes=5, doTrain=False, doTest=False):
    expert_configs = []
    expert_models = []
    save_dir = "project/Explainable-Sexism/onevsall_models"
    for label in range(num_classes):
        model_filename=f"unbalanced_expert_{label}_batch8"

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
            'batch_size': 16,
            'lr': 1e-3,
            'warmup': 0.2, 
            'train_size': len(dm.train_dataloader()),
            'weight_decay': 0.001,
            'n_epochs': 3,
            }
        expert_configs.append(expert_config)
        expert_model = ExpertClassifier(expert_config)
        
        checkpoint_callback = ModelCheckpoint(
                dirpath=save_dir,
                save_top_k=1,
                monitor="val f1",
                filename=model_filename
                )

        if not doTrain:
            ckpt = torch.load(
                  f"{save_dir}/{model_filename}.ckpt"
                    )
            expert_model.load_state_dict(ckpt['state_dict'])
            expert_models.append(expert_model)
        else:
            trainer = pl.Trainer(
                max_epochs=expert_config['n_epochs'],
                accelerator='gpu',
                devices=1,
                num_sanity_val_steps=15,
                default_root_dir="project/Explainable-Sexism/",
                callbacks=[checkpoint_callback],
                )
            trainer.fit(expert_model, dm)
            torch.save(
                expert_model.state_dict(),
                f"{save_dir}/{model_filename}.pt"
                )
            expert_models.append(expert_model)

        if doTest:
            expert_model.eval()
            test_dataloader = dm.test_dataloader()
            trainer.test(expert_model, dataloaders=test_dataloader)

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

    full_experts, configs = train_experts(df, model_name, doTrain=False, doTest=False)

    experts = [AutoModel.from_pretrained(model_name) for _ in range(len(label_map))]

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

    master = train_master(df, model_name, experts, doTrain=True, doTest=True)
        






