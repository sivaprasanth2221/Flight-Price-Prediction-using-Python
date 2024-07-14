# Flight Price Prediction

This project aims to predict flight prices using a dataset from Kaggle. The dataset includes various features such as airline, source city, destination city, departure and arrival times, number of stops, and class. The project involves data preprocessing, feature encoding, and building a regression model using RandomForestRegressor. Additionally, hyperparameter tuning is performed using GridSearchCV and RandomizedSearchCV.

## Dataset

The dataset used in this project is available on Kaggle: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

## Features

- Data Preprocessing: Handling missing values, encoding categorical features, and feature scaling.
- Regression Model: Building and training a RandomForestRegressor model.
- Model Evaluation: Evaluating the model using R2 score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- Hyperparameter Tuning: Using GridSearchCV and RandomizedSearchCV for optimizing the model.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- scipy

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/flight-price-prediction.git
    ```

2. **Navigate to the project directory**:
    ```sh
    cd flight-price-prediction
    ```

3. **Install the required packages**:
    ```sh
    pip install pandas scikit-learn matplotlib scipy
    ```

## Usage

1. **Load the dataset**:
    ```python
    import pandas as pd
    df = pd.read_csv("Clean_Dataset.csv")
    ```

2. **Data Preprocessing**:
    ```python
    df.stops = pd.factorize(df.stops)[0]
    df = df.drop("Unnamed: 0", axis=1)
    df = df.drop("flight", axis=1)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
    df = df.join(pd.get_dummies(df.airline, prefix="airline")).drop("airline", axis=1)
    df = df.join(pd.get_dummies(df.source_city, prefix="source")).drop("source_city", axis=1)
    df = df.join(pd.get_dummies(df.destination_city, prefix="dest")).drop("destination_city", axis=1)
    df = df.join(pd.get_dummies(df.arrival_time, prefix="arrival")).drop("arrival_time", axis=1)
    df = df.join(pd.get_dummies(df.departure_time, prefix="departure")).drop("departure_time", axis=1)
    ```

3. **Train the Regression Model**:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X, y = df.drop('price', axis=1), df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = RandomForestRegressor(n_jobs=-1)
    reg.fit(X_train, y_train)
    reg.score(X_test, y_test)
    ```

4. **Evaluate the Model**:
    ```python
    import math
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = reg.predict(X_test)

    print("R2: ", r2_score(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred)))

    import matplotlib.pyplot as plt

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Flight Price")
    plt.ylabel("Predicted Flight Price")
    plt.title("Prediction vs Actual Price")
    ```

5. **Feature Importance**:
    ```python
    importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 6))
    plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]])
    ```

6. **Hyperparameter Tuning**:
    ```python
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from scipy.stats import randint

    # Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    grid_search = GridSearchCV(reg, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Randomized Search
    param_grid = {
        'n_estimators': randint(100, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': [1.0, 'auto', 'sqrt']
    }

    random_search = RandomizedSearchCV(estimator=reg, param_distributions=param_grid, cv=3, n_iter=20, 
                                       scoring='neg_mean_squared_error', verbose=2, random_state=10, n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_regressor = random_search.best_estimator_
    best_regressor.score(X_test, y_test)

    y_pred = best_regressor.predict(X_test)

    print("R2: ", r2_score(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred)))

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Flight Price")
    plt.ylabel("Predicted Flight Price")
    plt.title("Prediction vs Actual Price")
    ```

## Acknowledgments

- The dataset used in this project is from Kaggle: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).
