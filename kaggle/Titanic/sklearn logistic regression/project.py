import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _label_data(df, category):
    """labelize categorical data in df, and return unified df
    Args:
        df : pd.DataFrame
            raw data, numerical and categorical data
        category :list(str)
            list the column name of categorical data in the df
            ex: ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket']
    """
    df_cat = df[category]

    le = LabelEncoder()
    df_encoded = df_cat.apply(le.fit_transform)
    df_rest = df.drop(category, axis=1)

    return pd.concat([df_rest, df_encoded], axis=1)


def load_training_data():
    df = pd.read_csv('./data/train.csv')
    y = df.Survived
    df.drop('Survived', axis=1, inplace=True)

    df.Age.fillna(value=df.Age.mean(), inplace=True)
    df.Cabin.fillna(value='D', inplace=True)
    df.Embarked.fillna(value='C', inplace=True)

    X = _label_data(df, ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket'])
    return X, y


def load_testing_data():
    df = pd.read_csv('./data/test.csv')

    df.Age.fillna(value=df.Age.mean(), inplace=True)
    df.Cabin.fillna(value='D', inplace=True)
    df.Fare.fillna(value=df.Fare.dropna().quantile(q=.25, interpolation='midpoint'), inplace=True)

    return _label_data(df, ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket'])
