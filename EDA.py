
import numpy as np
import pandas as pd
from IPython.display import display



def load_arrange_data(data_path):
  
  data = pd.read_csv(data_path)

  # represent label_sexist in a seperate binary column
  data['sexist'] = np.where(data['label_sexist'] == 'sexist', 1, 0)
  data['non_sexist'] = np.where(data['label_sexist'] == 'sexist', 0, 1) ####### UNNESSECARY, WE ONLY USE THIS FOR BALANCING BINARY CLASSES, BUT EVENTUALLY WE WANT TO BALANCE MULTIPLE CLASSES

  # represent label_category in seperate binary columns
  labels = ['1. threats, plans to harm and incitement','2. derogation', '3. animosity','4. prejudiced discussions']    ##### ADD NONE 
  for k in labels:
      data[k] = np.where(data['label_category'] == k, 1, 0)

  # drop unnessecary columns
  data.drop(['rewire_id', 'label_sexist','label_vector'], axis=1, inplace=True)    

  attributes = labels        ######## ATTRIBUTES SHOULD BE CALLED LABELS

  return data, attributes



def show_data(data):
    display(data)



'''
# plot data
data[attributes].sum().plot.bar()

# measure class sizes
binary_labels = ['sexist']
no_sexist = data[binary_labels].sum()
no_no_sexist = len(data)-no_sexist
label_count = data[attributes].sum()

# split data
#X_train, X_test, y_train, y_test = train_test_split(
#    data, data['label_category'], test_size=0.33, random_state=42)

X_train, X_val_test, y_train, y_val_test = train_test_split(data, data['label_category'], test_size=0.3, random_state=1) # 70 % train, 15 % val, 15 % test data

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1) 

# inspect data in table
data.head(5)

'''