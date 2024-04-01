import helper

# Load data files
X_train, X_val, X_test, y_train, y_val, y_test = helper.load_data_files()
print(str(len(X_train)))
print(str(len(X_val)))
print(str(len(X_test)))