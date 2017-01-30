import pandas as pd

#ref: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

file_path = '../data/'

train_data = pd.read_csv(file_path + 'train.csv')
test_data = pd.read_csv(file_path + 'test.csv')

# print(train_data.head())
print(train_data.info())
print(test_data.info())

train_len = len(train_data)
print('train len')
print(train_len)

print('train data shape '+ str(train_data.shape))
print('test data shape ' + str(test_data.shape))

y_train = train_data['Loan_Status']
print(y_train[:5])

train_data.drop('Loan_Status', axis = 1, inplace = True)
print(train_data.shape)

all_data = pd.concat([train_data, test_data], axis = 0)
print('all data shape' + str(all_data.shape))

print(all_data.head())
print(all_data.tail())	 

print(all_data.info())