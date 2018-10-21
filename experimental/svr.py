from sklearn.svm import SVR, SVC 
from sklearn import metrics
import os 
import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt

def build_model(X_train, X_val, y_train, y_val):
    print(X_val)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print('Here')

    y_rbf = svr_rbf.fit(X_val, y_val).predict(X_val)
    print('Here')
    y_lin = svr_lin.fit(X_val, y_val).predict(X_val)
    print('Here')
    y_poly = svr_poly.fit(X_val, y_val).predict(X_val)

    lw = 2
    plt.scatter(X_val, y_val, color='darkorange', label='data')
    plt.plot(X_val, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X_val, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X_val, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

def svm_model(X_train, X_test, y_train, y_test):
    print('Starting support vector machine model...')
    #Create a svm Classifier
    clf = SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def get_fivepct():
    times = sorted(dataset.index.values)
    last_5pct = times[-int(len(times)*(0.05))]
    validation_df = dataset[(dataset.index >= last_5pct)]
    train_df = dataset[(dataset.index < last_5pct)]
    return validation_df, train_df

def build_sequences(df):
    sequential_data = []
    for i in df.values:
        sequential_data.append([np.array([n for n in i[:-1]]), i[-1]])
    random.shuffle(sequential_data)
    return balance_data(sequential_data)

def balance_data(sequential_data):
    buys, sells = [], []
    for seq, target in sequential_data:
        if target == 1:
            buys.append([seq, target])
        else:
            sells.append([seq, target])

    print('The length of buys is {}, The length of sells is {}'.format(len(buys), len(sells)))
    lower = min(len(buys), len(sells))
    buys, sells = buys[:lower], sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    return sequential_data

def extract_feature_labels(seq_data):
    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

hour = False
if hour:
    filename = 'dataset_files/training_sets/hourly_indicators.csv'
else:
    filename = 'dataset_files/training_sets/minute_indicators.csv'
if os.path.exists(filename):
    print("Reading in dataset with filename {}".format(filename))
    dataset = pd.read_csv(filename, index_col=0)
    dataset = dataset.tail(9500)
    validation_df, train_df = get_fivepct()

# Building Training Data
train_seq = build_sequences(train_df)
X_train, y_train = extract_feature_labels(train_seq)

validate_seq = build_sequences(validation_df)
X_test, y_test = extract_feature_labels(validate_seq)

svm_model(X_train, X_test, y_train, y_test)

