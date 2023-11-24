import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def Stacking(model, train, y, test, n_fold):
    folds = StratifiedKFold(n_splits=n_fold, random_state=1)
    test_pred = np.empty((test.shape[0], 1), float)
    train_pred = np.empty((0, 1), float)
    for train_indices, val_indices in folds.split(train, y.values):
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model.fit(X=x_train, y=y_train)
        train_pred = np.append(train_pred, model.predict(x_val))
        test_pred = np.append(test_pred, model.predict(test))
    return test_pred.reshape(-1, 1), train_pred


def simply_ensemble(scaled_df_merge, num):
    print("------------------------")
    print(num)
    X = scaled_df_merge.iloc[:, :-1]  # Features (all columns except the last one)
    y = scaled_df_merge.iloc[:, -1]  # Target variable (last column)

    train_size = X.shape[1]
    train_ratio = float(train_size / X.shape[0])
    # Step 4: Split the data into training and testing sets

    for n in range(1, 7):
        error_set = []
        for k in range(1, 10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - n * train_ratio),
                                                                random_state=42 + k)
            model1 = tree.DecisionTreeClassifier(random_state=1)

            test_pred1, train_pred1 = Stacking(model=model1, n_fold=10, train=X_train, test=X_test, y=y_train)

            train_pred1 = pd.DataFrame(train_pred1)
            test_pred1 = pd.DataFrame(test_pred1)

            model2 = KNeighborsClassifier()

            test_pred2, train_pred2 = Stacking(model=model2, n_fold=10, train=X_train, test=X_test, y=y_train)

            train_pred2 = pd.DataFrame(train_pred2)
            test_pred2 = pd.DataFrame(test_pred2)

            df = pd.concat([train_pred1, train_pred2], axis=1)
            df_test = pd.concat([test_pred1, test_pred2], axis=1)

            model = LogisticRegression(random_state=1)
            model.fit(df, y_train)
            model.score(df_test, y_test)
