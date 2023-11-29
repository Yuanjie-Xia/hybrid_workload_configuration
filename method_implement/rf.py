from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from error_calculation import relative_error
import statistics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def rf_regression(scaled_df_merge, num):
    print("------------------------")
    print(num)
    X = scaled_df_merge.iloc[:, :-1]  # Features (all columns except the last one)
    y = scaled_df_merge.iloc[:, -1]   # Target variable (last column)

    train_size = X.shape[1]
    train_ratio = float(train_size/X.shape[0])
    # Step 4: Split the data into training and testing sets

    for n in range(1, 7):
        error_set = []
        for k in range(1, 10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-n*train_ratio), random_state=42+k)

            # Step 5: Train an XGBoost model
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate the Variance of the actual target values
            variance_y = np.var(y_test)

            # Calculate the Relative Error (RE) for each prediction
            r_error_list = relative_error(y_pred, y_test)

            # Calculate the Median RSE
            mre = statistics.median(r_error_list)
            #print("Relative Mean Squared Error: {:.4f}".format(np.mean(rse)))
            error_set.append(mre)
        print(statistics.mean(error_set))
        print("_______________________")