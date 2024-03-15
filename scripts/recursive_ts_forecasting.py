import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def create_recursive_data(data, window_size=5):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
# Fill missing values by interpolation
data["co2"] = data["co2"].interpolate()


data = create_recursive_data(data, 5)
x = data.drop(["time", "target"], axis=1)
y = data["target"]
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

reg = LinearRegression()
# reg = RandomForestRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("R2 score: {}".format(r2_score(y_test, y_predict)))
print("MAE score: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE score: {}".format(mean_squared_error(y_test, y_predict)))

fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples*train_size)], data["co2"][:int(num_samples*train_size)], label="train")
ax.plot(data["time"][int(num_samples*train_size):], data["co2"][int(num_samples*train_size):], label="test")
ax.plot(data["time"][int(num_samples*train_size):], y_predict, label="predict")
ax.set_xlabel("Time")
ax.set_ylabel("Co2")
ax.legend()
ax.grid()
plt.show()