import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys

INPUT_DIR = Path('./data').resolve()
OUTPUT_DIR = Path('./').resolve()


def read_data(filename):
    data = pd.read_csv(INPUT_DIR / filename)
    category = data['Category']
    message = data['Message']
    return message, category

def get_fit_vectorizer(message, use_engineering=False):
    if use_engineering:  # Feature engineering for SVM
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),  # adds unigrams and bigrams (e.g., "free" and "free money")
            min_df=2,  # ignore terms that appear in less than 2 documents
            max_df=0.95,  # ignore terms that appear in more than 95% of documents
        )
    else:  # Basic settings for DT (works better)
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


def stacking_model(X, y):
    svm = SVC(  # keeping same hyperparameters for this
        kernel='sigmoid',
        C=0.8,
        gamma=0.0004,
        coef0=0,
        tol=0.001)

    decision_tree = DecisionTreeClassifier()

    stacking = StackingClassifier( estimators = [('svm', svm) ('tree', decision_tree)], final = LogisticRegression())

    stacking.fit(X, y)

    return stacking

def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)

def write_output(model, X_test, output_filename, input_filename):
    y_pred = model.predict(X_test)
    og_data = pd.read_csv(INPUT_DIR / input_filename)
    og_data['Predicted'] = y_pred
    og_data.to_csv(OUTPUT_DIR / output_filename)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 ensemble_classifier.py train_filename test_filename train_output_filename test_output_filename")
        sys.exit(1)

    model_type = sys.argv[1]
    train_filename = sys.argv[2]
    test_filename = sys.argv[3]
    train_output_filename = sys.argv[4]
    test_output_filename = sys.argv[5]

    train_message, train_category = read_data(train_filename)
    test_message, test_category = read_data(test_filename)
    full_message = pd.concat([train_message, test_message])
    vectorizer = get_fit_vectorizer(full_message, use_engineering=(model_type == 'svm'))
    scaler = get_fit_scaler(vectorizer.transform(full_message))
    train_X, train_y = preprocess_data(train_message, train_category, vectorizer, scaler)
    test_X, test_y = preprocess_data(test_message, test_category, vectorizer, scaler)

    model = stacking_model(train_X, train_y)

    train_score = evaluate_model(model, train_X, train_y)
    test_score = evaluate_model(model, test_X, test_y)

    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")

    write_output(model, test_X, test_output_filename, test_filename)
    write_output(model, train_X, train_output_filename, train_filename)

    print(f"Output written to {train_output_filename} and {test_output_filename}")
