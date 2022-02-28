import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def get_rmse(test, predict):
    return np.sqrt(mean_squared_error(test, predict))


x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]
x_predict = np.array(range(101, 111))  # 최종 예측시에는 테스트에도 사용하지 않은 데이터로 예측해보는것이 좋음

model = Sequential()
model.add(Dense(5, input_shape=(1,), activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

mse = model.evaluate(x_test, y_test, batch_size=1)

y_predict = model.predict(x_test)

r2_y_predict = r2_score(y_test, y_predict)

model.save('predict_num')

print(f"mse : {mse}")
print(f"rmse : {get_rmse(y_test, y_predict)}")
print(f"r2 : {r2_y_predict}")

y_predict = model.predict(x_predict)

print(f"predict : {y_predict}")