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

def process_gender():
	# print(all_data['Gender'].mode())
	all_data['Gender'].fillna('Male', inplace = True)
	# combined.Fare.fillna(combined.Fare.mean(),inplace=True)

	print('gender processed ok')

process_gender()

print(all_data.info())

def process_married():
	# print('married mode')
	# print(all_data['Married'].mode())
	all_data['Married'].fillna('Yes', inplace = True)
	print('Married processed ok')

process_married()
print(all_data.info())


def process_dependents():
	# print('dependents mode')
	# print(all_data['Dependents'].mode())
	all_data['Dependents'].fillna(0, inplace = True)
	# print('Dependents processed ok')

process_dependents()
print(all_data.info())

print(all_data.shape)
print(all_data.describe())