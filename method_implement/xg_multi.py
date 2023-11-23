import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from error_calculation import relative_error, error, abs_relative_error
import statistics


def multi_model_xg(scaled_df_merge, num):
    print("------------------------")
    print(num)
    X = scaled_df_merge.iloc[:, :-1]    # Feature
    y = scaled_df_merge.iloc[:, -1]   # Target variable (last column)

    train_size = X.shape[1]
    train_ratio = float(train_size/X.shape[0])
    # Step 4: Split the data into training and testing sets

    for n in range(1, 7):
        error_set = []
        for k in range(1, 10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-n*train_ratio), random_state=42+k)
            X_train_1 = X_train.iloc[:, :8]
            X_train_2 = X_train.iloc[:, 8:-1]
            X_test_1 = X_test.iloc[:, :8]
            X_test_2 = X_test.iloc[:, 8:-1]

            # Step 5: Train an XGBoost model
            model = xgb.XGBRegressor()
            model_error = xgb.XGBRegressor()
            model.fit(X_train_1, y_train)

            # Calculate the Variance of the actual target values
            # variance_y = np.var(y_test)

            # Calculate the Relative Error (RE) for each prediction
            y_pred_self = model.predict(X_train_1)
            error_list_self = relative_error(y_pred_self, y_train)
            model_error.fit(X_train_2, error_list_self)

            # Make predictions on the test set
            y_pred = model.predict(X_test_1)
            y_pred_error = model_error.predict(X_test_2)
            y_pred = y_pred / (1 + y_pred_error)
            r_error_list = abs_relative_error(y_pred, y_test)

            # Calculate the Median RSE
            mre = statistics.median(r_error_list)
            #print("Relative Mean Squared Error: {:.4f}".format(np.mean(rse)))
            error_set.append(mre)
        print(statistics.mean(error_set))
        print("_______________________")