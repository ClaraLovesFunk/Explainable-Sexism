import pandas as pd
from IPython.display import display

data_path = 'data/train_all_tasks.csv'
data = pd.read_csv(data_path)
display(data)
