import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sys
from sklearn.model_selection import GridSearchCV

def read_data(filename):
    INPUT_DIR = Path('data').resolve()
    data = pd.read_csv(INPUT_DIR / filename)
    category = data['Category']
    message = data['Message']
    return message, category

def vectorize_data(message):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(message)
    return X

def scale_data(X):
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    return X, scaler

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_svm(X, y):
    model = SVC(kernel='linear')
    model.fit(X, y)
    return model

def train_svm_optimized(X, y):
    model = SVC(kernel='linear')
    param_grid = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    grid = GridSearchCV(model, param_grid, refit = True, verbose = 3)
    grid.fit(X, y)
    chosen = grid.best_estimator_
    return grid, chosen

def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)

if __name__ == "__main__":


    message, category = read_data("labeled_email_samples.csv")
    X = vectorize_data(message)
    X, scaler = scale_data(X)
    X_train, X_test, y_train, y_test = split_data(X, category)
    model, chosen = train_svm_optimized(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Best parameters: {chosen.get_params()}')
