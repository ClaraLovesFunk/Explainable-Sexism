import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModel
from onevsall_expert import train_experts
from onevsall_master import train_master, MasterDataModule


def get_preds(model, model_name, df_train, df_test):
    df_copy = df_test.copy(deep=True)
    df_copy.drop(['rewire_id'], axis=1, inplace=True)
    dm = MasterDataModule(df_train, df_test, model_name=model_name)
    dm.setup()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        num_sanity_val_steps=15,
        default_root_dir=os.getcwd(),
    )
    model.eval()
    test_dataloader = dm.predict_dataloader()
    y_pred_tensor = trainer.predict(model, dataloaders=test_dataloader)

    label_map = {
        0: '1. threats, plans to harm and incitement',
        1: '2. derogation',
        2: '3. animosity',
        3: '4. prejudiced discussions'
    }

    y_pred = []
    for tensor in y_pred_tensor:
        y_pred.extend(np.argmax(tensor.numpy(), axis=1))

    print(np.unique(y_pred))
    df_test.drop(['label'], axis=1, inplace=True)
    df_test['label'] = y_pred
    df_test['label'].replace(label_map, inplace=True)
    df_test.drop(['text', '0', '1', '2', '3'], inplace=True, axis=1)
    return df_test


def preproc_data(data_path, label_map):
    df = pd.read_csv(data_path)
    df.drop(df.loc[df['label_category'] == 'none'].index, axis=0, inplace=True)
    label_map = {
        '1. threats, plans to harm and incitement': 0,
        '2. derogation': 1,
        '3. animosity': 2,
        '4. prejudiced discussions': 3
    }
    df.drop(['rewire_id', 'label_sexist', 'label_vector'], axis=1, inplace=True)
    df['label_category'].replace(label_map, inplace=True)
    df.rename(columns={'label_category': 'label'}, inplace=True)

    df_train = df[df['split'] == 'train']
    df_dev = df[df['split'] == 'dev']
    df_test = df[df['split'] == 'test']

    return df_train, df_dev, df_test


if __name__ == "__main__":
    data_path = 'data/edos_labelled_aggregated.csv'
    label_map = {
        '1. threats, plans to harm and incitement': 0,
        '2. derogation': 1,
        '3. animosity': 2,
        '4. prejudiced discussions': 3
    }

    df_train, df_dev, df_test = preproc_data(data_path, label_map)
    num_classes = len(np.unique(df_train['label']))

    expert_config = {
        # 'hf_model_name': "microsoft/deberta-large",
        'hf_model_name': "GroNLP/hateBERT",
        'expert_type': 'hatebert',
        'num_classes': num_classes,
        'n_labels': 2,
        'batch_size': 16,
        'balance': 1,
        'lr': 3e-4,
        'warmup': 0.2,
        'weight_decay': 0.001,
        'n_epochs': 25,
        'root_dir': os.getcwd(),
        'save_dir': "onevsall_models",
        'log_dir': 'lightning_logs/'
    }

    full_experts = train_experts(df_train, df_dev, df_test, expert_config, doTrain=False, doTest=False)

    experts = [AutoModel.from_pretrained(expert_config['hf_model_name']) for _ in range(len(label_map))]

    for i in range(len(label_map)):
        finetuned_dict = full_experts[i].state_dict()
        model_dict = experts[i].state_dict()

        # 1. filter out unnecessary keys
        finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dictuu
        model_dict.update(finetuned_dict)
        # 3. load the new state dict
        experts[i].load_state_dict(model_dict)
        # freeze the weights for the master model
        experts[i].eval()

    master_config = {
        # 'hf_model_name': "microsoft/deberta-large",
        'hf_model_name': "GroNLP/hateBERT",
        'n_labels': num_classes,
        'experts': experts,
        'batch_size': 16,
        'lr': 3e-4,
        'warmup': 0.2,
        'weight_decay': 0.001,
        'n_epochs': 100,
        'root_dir': os.getcwd(),
        'save_dir': "onevsall_models",
        'log_dir': 'lightning_logs/'
    }
    master = train_master(df_train, df_dev, df_test, master_config, doTrain=True, doTest=True)

    # #dev
    # test1 = pd.read_csv('project/Explainable-Sexism/data/dev_task_b_entries.csv')
    # test1['label'] = np.zeros(len(test1))
    # df = get_preds(master, model_name, df, test1)
    # print(df.head())
    # df.to_csv('project/Explainable-Sexism/data/dev_task_b_result.csv', index=False)

    # #test
    # test2 = pd.read_csv('project/Explainable-Sexism/data/test_task_b_entries.csv')
    # test2['label'] = np.zeros(len(test2))
    # df = get_preds(master, model_name, df, test2)
    # df.drop(['text', '0', '1', '2', '3'], inplace=True, axis=1)
    # print(df.head())
    # df.to_csv('project/Explainable-Sexism/data/test_task_b_result.csv', index=False)
