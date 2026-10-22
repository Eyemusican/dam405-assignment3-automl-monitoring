from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # flip labels so 1 = malignant, the case we must not miss
    y = (1 - data.target).rename("malignant")
    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
