from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from preprocessing import *

df = load_data("../../data/environmental/flood.csv")

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = split_dataset(X, y)

print(X_train.shape)

print(X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print()

print("RESULTS - ")

print(f"MSE : {mse}")

print(f"R²  : {r2}")