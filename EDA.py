import numpy as np
import pandas as pd
from IPython.display import display



def load_arrange_data(data_path):
  
  data = pd.read_csv(data_path)

  labels = ['1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions']    

  # make columns for each label
  for k in labels:
    data[k] = np.where(data['label_category'] == k, 1, 0)
    
  label_map = {                                  
    'none': 0,
    '1. threats, plans to harm and incitement' : 1,
    '2. derogation': 2,
    '3. animosity': 3,
    '4. prejudiced discussions': 4
    }
  
  data['label_category'].replace(label_map, inplace=True) 
  
  # drop instances with no sexism in train data since no sexism is not regarded as a class in the semeval task
  data.drop(data.loc[data['label_category']==0].index, inplace=True)
  
  return data, labels

#data, attributes = load_arrange_data('data/test_task_b_entries.csv', True)
#data.drop(data['text'], inplace=True, axis=1)
#display(data)