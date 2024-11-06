import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

def read_data(filename):
    INPUT_DIR = Path('data').resolve()
    data = pd.read_csv(INPUT_DIR / filename)
    category = data['Category']
    message = data['Message']
    return message, category

def split_data(message, category):
    X_train, X_test, y_train, y_test = train_test_split(message, category, test_size=0.2)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 split_data.py filename output_filename")
        sys.exit(1)

    filename = sys.argv[1]
    output_filename = sys.argv[2]

    message, category = read_data(filename)
    X_train, X_test, y_train, y_test = split_data(message, category)
    INPUT_DIR = Path('data').resolve()

    train_data = pd.DataFrame({'Message': X_train, 'Category': y_train})
    test_data = pd.DataFrame({'Message': X_test, 'Category': y_test})

    train_data.to_csv(INPUT_DIR / f'train_{output_filename}.csv', index=False)
    test_data.to_csv(INPUT_DIR / f'test_{output_filename}.csv', index=False)