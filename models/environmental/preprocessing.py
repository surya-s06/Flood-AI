import pandas as pd


def load_data(file_path):

    df = pd.read_csv(file_path)

    return df

def split_features_target(df):

    X = df.drop("FloodProbability", axis=1)

    y = df["FloodProbability"]

    return X, y

from sklearn.model_selection import train_test_split

def split_dataset(X, y):

    return train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )