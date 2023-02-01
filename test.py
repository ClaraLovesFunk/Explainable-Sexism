import numpy as np
import pandas as pd
from IPython.display import display

label_pred = pd.read_csv('data/test_task_b_entries_pred.csv')
display(label_pred)

        full_expert = Expert_Classifier(config)
        trainer.fit(full_expert, full_expert_dm)
        torch.save(full_expert.state_dict(),f'{model_path}/{model_name}_bal_{b}.pt')