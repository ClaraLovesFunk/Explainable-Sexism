

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