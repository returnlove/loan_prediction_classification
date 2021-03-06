import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# import sklearn

# print(sklearn.__version__)

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
print(all_data.info())

def process_selfemp():
	# print('Self_Employed mode')
	# print(all_data['Self_Employed'].mode())
	all_data['Self_Employed'].fillna('No', inplace = True)
	print('Self_Employed processed ok')

process_selfemp()
print(all_data.info())


def process_loanamount():
	# print('LoanAmount mode')
	# print(all_data['LoanAmount'].mode())
	# print('LoanAmount mean')

	# print(all_data['LoanAmount'].mean())

	all_data.LoanAmount.fillna(120.0, inplace=True)
	print('LoanAmount processed ok')

process_loanamount()
print(all_data.info())

def process_loanamountterm():
	# print('Loan_Amount_Term mode')
	# print(all_data['Loan_Amount_Term'].mode())
	# print('Loan_Amount_Term mean')

	# print(all_data['Loan_Amount_Term'].mean())

	all_data.Loan_Amount_Term.fillna(360, inplace=True)
	print('Loan_Amount_Term processed ok')

process_loanamountterm()
print(all_data.info())


def process_Credit_History():
	# print('Credit_History mode')
	# print(all_data['Credit_History'].mode())
	# print('Credit_History mean')

	# print(all_data['Credit_History'].mean())

	all_data.Credit_History.fillna(1, inplace=True)
	print('Credit_History processed ok')

process_Credit_History()
print(all_data.info())


cat_cols = []
for col in all_data.columns.values:
	if all_data[col].dtypes == 'object':
		cat_cols.append(col)

print('cat columns:')
print(cat_cols)
cat_cols.remove("Loan_ID")

non_cat_cols = []

for col in all_data.columns.values:
	if col not in cat_cols:
		non_cat_cols.append(col)

non_cat_cols.remove("Loan_ID")
print('non cat cols')

print(non_cat_cols)

all_data_new = pd.DataFrame()


# print("processing column" + " Gender")
# var = "Gender"+"_dummy"
# var = pd.get_dummies(all_data["Gender"], prefix = "Gender")
# all_data_new = pd.concat([all_data_new, var], axis = 1)

for col in cat_cols:
	print("processing column" + col)
	var = col+"_dummy"
	var = pd.get_dummies(all_data[col], prefix = col).iloc[:,1:]
	all_data_new = pd.concat([all_data_new, var], axis = 1)

print('all data new')
print(all_data_new.info())

all_data_new = pd.concat([all_data_new, all_data[non_cat_cols]], axis = 1)

print(all_data_new.info())

def split_train_test():
	print('before')
	global train_data
	global test_data
	print(train_data.shape)
	print(test_data.shape)
	train_data = all_data_new[:train_len]
	test_data = all_data_new[train_len:]
	print('after')
	print(train_data.shape)
	print(test_data.shape)

split_train_test()




X_train, X_test, y_train, y_test = train_test_split(train_data, y_train)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('RF score' + str(accuracy_score(y_test, clf.predict(X_test))))


clf = LogisticRegression()
clf.fit(X_train, y_train)
print('LR score' + str(accuracy_score(y_test, clf.predict(X_test))))


# clf = KNeighborsClassifier()
# clf.fit(X_train, y_train)
# print('KNN score' + str(accuracy_score(y_test, clf.predict(X_test))))

neighbors = list(range(1, 20))
max_score = 0
scores =[]
for i in neighbors:
	clf = KNeighborsClassifier(n_neighbors = i)
	clf.fit(X_train, y_train)
	# print(' K = ' + str(i))
	# print('KNN score'+ str(accuracy_score(y_test, clf.predict(X_test))))
	scores.append(accuracy_score(y_test, clf.predict(X_test)))
	if(accuracy_score(y_test, clf.predict(X_test)) > max_score):
		print('max val at' + str(i))
		max_score = accuracy_score(y_test, clf.predict(X_test))
		print('max score is' + str(max_score))


# print('max score')
print(max(scores))



# grid search cv

mylist = list(range(1,50))
neighbors = list(range(1,51,2))


parameters = [{'n_neighbors': [i for i in range(1,20)]}]
clf = GridSearchCV(KNeighborsClassifier(), parameters, cv = 10)
print(clf)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
print(dir(clf))

# http://stackoverflow.com/questions/35388647/how-to-use-gridsearchcv-output-for-a-scikit-prediction
y_true, y_pred = y_test, clf.predict(X_test)
print( "KNeighborsClassifier accuracy score" + str(accuracy_score(y_true, y_pred)))
# print(classification_report(y_true, y_pred))

# svm


clf = SVC()
print('svm object')
print(clf)
clf.fit(X_train, y_train)
print('SVM score default' + str(accuracy_score(y_test, clf.predict(X_test))))

# clf = SVC(kernel='rbf')

# clf.fit(X_train, y_train)
# print('SVM score rbf kernal' + str(accuracy_score(y_test, clf.predict(X_test))))



# param_grid = [
#   {'C': [1], 'kernel': ['linear']}
#   # {'C': [1], 'gamma': [0.001], 'kernel': ['rbf']},
#  ]
# clf = GridSearchCV(SVC(), param_grid)
# print(clf)
# clf.fit(X_train, y_train)
# print('best params')
# print(clf.best_params_)
# print('best score')
# print(clf.best_score_)



clf = GaussianNB()
clf.fit(X_train, y_train)
print('NB score' + str(accuracy_score(y_test, clf.predict(X_test))))
