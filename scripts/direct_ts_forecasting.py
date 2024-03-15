import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def create_direct_data(data, window_size=5, target_size=3):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["co2"].shift(-i-window_size)
        i += 1
    data = data.dropna(axis=0)
    return data


data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])
# Fill missing values by interpolation
data["co2"] = data["co2"].interpolate()

window_size = 8
target_size = 4
data = create_direct_data(data, window_size, target_size)
x = data.drop(["time"] + ["target_{}".format(i) for i in range(target_size)], axis=1)
y = data[["target_{}".format(i) for i in range(target_size)]]
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples * train_size)]
y_train = y[:int(num_samples * train_size)]
x_test = x[int(num_samples * train_size):]
y_test = y[int(num_samples * train_size):]

regs = [LinearRegression() for _ in range(target_size)]
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])
r2 = []
mae = []
mse = []
for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    r2.append(r2_score(y_test["target_{}".format(i)], y_predict))
    mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i)], y_predict))
print("R2 score {}".format(r2))
print("Mean absoluate error {}".format(mae))
print("Mean squared error {}".format(mse))
#
# fig, ax = plt.subplots()
# ax.plot(data["time"][:int(num_samples * train_size)], data["co2"][:int(num_samples * train_size)], label="train")
# ax.plot(data["time"][int(num_samples * train_size):], data["co2"][int(num_samples * train_size):], label="test")
# ax.plot(data["time"][int(num_samples * train_size):], y_predict, label="predict")
# ax.set_xlabel("Time")
# ax.set_ylabel("Co2")
# ax.legend()
# ax.grid()
# plt.show()
