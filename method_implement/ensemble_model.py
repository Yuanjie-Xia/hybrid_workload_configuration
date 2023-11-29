# Import necessary libraries
import statistics

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from error_calculation import relative_error


def simple_ensemble(scaled_df_merge, num):
    X = scaled_df_merge.iloc[:, :-1]  # Features (all columns except the last one)
    y = scaled_df_merge.iloc[:, -1]  # Target variable (last column)

    train_size = X.shape[1]
    train_ratio = float(train_size / X.shape[0])
    # Step 4: Split the data into training and testing sets

    for n in range(1, 7):
        error_set = []
        for k in range(1, 11):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - n * train_ratio),
                                                                random_state=42 + k)

            # Define base models
            base_models = [
                ('linear', LinearRegression()),
                ('tree', DecisionTreeRegressor()),
                ('forest', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('booster', xgb.XGBRegressor())
            ]

            # Define the meta-model
            meta_model = xgb.XGBRegressor()

            # Create the stacking model
            stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

            # Train the stacking model
            stacking_model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred_stacking = stacking_model.predict(X_test)

            # Evaluate the stacking model
            mse_stacking = relative_error(y_test, y_pred_stacking)
            mre = statistics.median(mse_stacking)
            error_set.append(mre)
        print(statistics.mean(error_set))
        print("_______________________")
