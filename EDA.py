import numpy as np
import pandas as pd
from IPython.display import display



def load_arrange_data(data_path):
  
  data = pd.read_csv(data_path)

  # represent label_sexist in a seperate binary column
  #data['sexist'] = np.where(data['label_sexist'] == 'sexist', 1, 0)
  #data['non_sexist'] = np.where(data['label_sexist'] == 'sexist', 0, 1) ####### UNNESSECARY, WE ONLY USE THIS FOR BALANCING BINARY CLASSES, BUT EVENTUALLY WE WANT TO BALANCE MULTIPLE CLASSES

  # represent label_category in seperate binary columns
  labels = ['1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions']    
  for k in labels:
      data[k] = np.where(data['label_category'] == k, 1, 0)

  label_map = {                                  
    'none':0,
    '1. threats, plans to harm and incitement' : 1,
    '2. derogation': 2,
    '3. animosity': 3,
    '4. prejudiced discussions': 4
    }
  data['label_category'].replace(label_map, inplace=True) 

  # drop unnessecary columns
  #data.drop(['rewire_id', 'label_sexist','label_vector'], axis=1, inplace=True)  
  data.drop(data.loc[data['label_category']==0].index, inplace=True)

  attributes = labels        


  #data[attributes].sum().plot.bar()

  return data, attributes

data, attributes = load_arrange_data('data/train_all_tasks.csv')
data.drop(data['text'], inplace=True, axis=1)
display(data)