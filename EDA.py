import numpy as np
import pandas as pd
from IPython.display import display



def load_arrange_data(data_path,test_submission_flag):
  
  data = pd.read_csv(data_path)

  labels = ['1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions']    
  
  # if we use test submission data or not
  if test_submission_flag == True:
    data['label_category'] = np.zeros(len(data))
    
    # make columns for each label
    for k in labels:
      data[k] = np.where(data['label_category'] == k, 1, 0)
    
  else:
    label_map = {                                  
      'none': 0,
      '1. threats, plans to harm and incitement' : 1,
      '2. derogation': 2,
      '3. animosity': 3,
      '4. prejudiced discussions': 4
      }
    
    data['label_category'].replace(label_map, inplace=True) 
    # drop instances with no sexism in train data
    data.drop(data.loc[data['label_category']==0].index, inplace=True)

    # make columns for each label
    for k in labels:
      data[k] = np.where(data['label_category'] == k, 1, 0)
    
  return data, labels

#data, attributes = load_arrange_data('data/test_task_b_entries.csv', True)
#data.drop(data['text'], inplace=True, axis=1)
#display(data)