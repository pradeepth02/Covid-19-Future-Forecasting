from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
 
 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
from sklearn.linear_model import ElasticNet
import os

# setting a seed for reproducibility
numpy.random.seed(10)
# read all stock files in directory indivisual_stocks_5yr
def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" +stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# create dataset from the dataframe
def create_preprocessed_Dataset(df):
    df.drop(df.columns.difference(['date', 'New_cases']), 1, inplace=True)
    df = df['New_cases']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY
# extract input dates and opening price value of stocks
def getData(df):
    # Create the lists / X and Y data sets
    dates = []
    prices = []

    # Get the number of rows and columns in the data set
    # df.shape

    # Get the last row of data (this will be the data that we test on)
    last_row = df.tail(1)

    # Get all of the data except for the last row
    df = df.head(len(df) - 1)
    # df

    # The new shape of the data
    # df.shape

    # Get all of the rows from the Date Column
    df_dates = df.loc[:, 'date']
    # Get all of the rows from the Open Column
    df_open = df.loc[:, 'New_cases']

    # Create the independent data set X
    for date in df_dates:
        dates.append([int(date.split('-')[2])])

    # Create the dependent data se 'y'
    for open_price in df_open:
        prices.append(float(open_price))

    # See what days were recorded
    last_date = int(((list(last_row['date']))[0]).split('-')[2])
    last_price = float((list(last_row['New_cases']))[0])
    return dates, prices, last_date, last_price


def SVR_linear(dates, prices, test_date, df):
    svr_lin = SVR(kernel='linear', C=1e3)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.33, random_state = 42)
    svr_lin.fit(X_train, y_train)
    decision_boundary = svr_lin.predict(trainX)
    y_pred = svr_lin.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = svr_lin.predict(testX)[0]

    return (decision_boundary, prediction, test_score)

def SVR_rbf(dates, prices, test_date, df):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    svr_rbf.fit(trainX, trainY)
    decision_boundary = svr_rbf.predict(trainX)
    y_pred = svr_rbf.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = svr_rbf.predict(testX)[0]
    return (decision_boundary, prediction, test_score)

def linear_regression(dates, prices, test_date, df):
    lin_reg = LinearRegression()
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    lin_reg.fit(trainX, trainY)
    decision_boundary = lin_reg.predict(trainX)
    y_pred = lin_reg.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = lin_reg.predict(testX)[0]
    return (decision_boundary, prediction, test_score)

def random_forests(dates, prices, test_date, df):
    rand_forst = RandomForestRegressor(n_estimators=10, random_state=0)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    rand_forst.fit(trainX, trainY)
    decision_boundary = rand_forst.predict(trainX)
    y_pred = rand_forst.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = rand_forst.predict(testX)[0]

    return (decision_boundary, prediction, test_score)

def KNN(dates, prices, test_date, df):
    knn = KNeighborsRegressor(n_neighbors=2)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    knn.fit(trainX, trainY)
    decision_boundary = knn.predict(trainX)
    y_pred = knn.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = knn.predict(testX)[0]

    return (decision_boundary, prediction, test_score)

def DT(dates, prices, test_date, df):
    decision_trees = tree.DecisionTreeRegressor()
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    decision_trees.fit(trainX, trainY)
    decision_boundary = decision_trees.predict(trainX)
    y_pred = decision_trees.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = decision_trees.predict(testX)[0]
    return (decision_boundary, prediction, test_score)

def elastic_net(dates, prices, test_date, df):
    regr = ElasticNet(random_state=0)
    trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
    # trainX = [item for sublist in trainX for item in sublist]
    # testX = [item for sublist in testX for item in sublist]
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    regr.fit(trainX, trainY)
    decision_boundary = regr.predict(trainX)
    y_pred = regr.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = regr.predict(testX)[0]

    return (decision_boundary, prediction, test_score)

 

    return (trainPredict, (testPredict[0])[0], test_score)