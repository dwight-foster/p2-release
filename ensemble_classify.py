import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys

INPUT_DIR = Path('./data').resolve()
OUTPUT_DIR = Path('./').resolve()

#this is pretty much the same as part 1, the only difference is 

def read_data(filename):
    data = pd.read_csv(INPUT_DIR / filename)
    category = data['Category']
    message = data['Message']
    return message, category

def get_fit_vectorizer(message):
    vectorizer = CountVectorizer()
    vectorizer.fit(message)
    return vectorizer

def get_fit_scaler(X):
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    return scaler


def preprocess_data(message, category, vectorizer=None, scaler=None):
    if vectorizer is None:
        vectorizer = get_fit_vectorizer(message)
    X = vectorizer.transform(message)
    if scaler is None:
        scaler = get_fit_scaler(X)
    X = scaler.transform(X)
    y = category
    return X, y

def train_randomForest (X, y):
  forest = RandomForestClassifier(n_estimators=25)
  forest.fit(X, y)
  return forest

def evaluate(forest, X, y):
  return forest.score(X, y)

def write_output (forest, X_test, input_filename, output_filename):
  y_pred = forest.predict(X_test)
  data = pd.read_csv(INPUT_DIR / input_filename)
  data['Predicted'] = y_pred
  data.to_csv(OUTPUT_DIR / output_filename)

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python3 ensemble_classify.py train_filename test_filename train_output_filename test_output_filename")
        sys.exit(1)

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    train_output_filename = sys.argv[3]
    test_output_filename = sys.argv[4]

    train_message, train_category = read_data(train_filename)
    test_message, test_category = read_data(test_filename)
    full_message = pd.concat([train_message, test_message])
    vectorizer = get_fit_vectorizer(full_message)
    scaler = get_fit_scaler(vectorizer.transform(full_message))
    train_X, train_y = preprocess_data(train_message, train_category, vectorizer, scaler)
    test_X, test_y = preprocess_data(test_message, test_category, vectorizer, scaler)

    forest = train_randomForest(train_X, train_y)
    
    train_score = evaluate(forest, train_X, train_y)
    test_score = evaluate(forest, test_X, test_y)

    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")

    write_output(forest, test_X, test_output_filename, test_filename)
    write_output(forest, train_X, train_output_filename, train_filename)

    print(f"Output written to {train_output_filename} and {test_output_filename}")

