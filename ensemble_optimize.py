import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),  # adds unigrams and bigrams (e.g., "free" and "free money")
        min_df=2,  # ignore terms that appear in less than 2 documents
        max_df=0.95,  # ignore terms that appear in more than 95% of documents
    )
    X = vectorizer.fit_transform(message)
    return X


def scale_data(X):
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    return X, scaler


def train_stacking_optimized(X, y):
    # Create just 2 SVMs
    svm1 = SVC(kernel='sigmoid')
    svm2 = SVC(kernel='linear')

    # Create stacking with just 2 SVMs
    stacking = StackingClassifier(
        estimators=[('svm1', svm1), ('svm2', svm2)],
        final_estimator=LogisticRegression()
    )

    # Parameter grid for just sigmoid and linear
    param_grid = {
        'svm1__kernel': ['sigmoid'],
        'svm1__C': [0.7, 0.8, 0.82, 0.85, 0.9],     # added some fine-tuned values
        'svm1__gamma': [0.0003, 0.00035, 0.0004, 0.00042, 0.00045],
        'svm1__coef0': [-0.1, 0, 0.001, 0.01, 0.1],
        'svm1__tol': [0.001],

        'svm2__kernel': ['linear'],
        'svm2__C': [0.9, 0.95, 1.0, 1.05, 1.1],    # fine-tuned around 1.0
        'svm2__tol': [0.001],

        'final_estimator__C': [0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(stacking, param_grid, refit=True, verbose=3)
    grid.fit(X, y)
    chosen = grid.best_estimator_
    return grid, chosen

def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)


if __name__ == "__main__":
    train_message, train_category = read_data("train_labeled_email.csv")
    test_message, test_category = read_data("test_labeled_email.csv")

    # Combine for vectorizer fitting
    full_message = pd.concat([train_message, test_message])
    X = vectorize_data(full_message)
    X, scaler = scale_data(X)

    # Split back to train/test
    train_X = X[:len(train_message)]
    test_X = X[len(train_message):]

    model, chosen = train_stacking_optimized(train_X, train_category)
    accuracy = evaluate_model(model, test_X, test_category)
    print(f'Accuracy: {accuracy}')
    print(f'Best parameters: {chosen.get_params()}')