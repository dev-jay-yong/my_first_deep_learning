from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 데이터 준비
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 102, 103, 104, 105])

# 모델 구성
model = Sequential()

model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# 훈련
model.compile(loss='mse', optimizer='adam',metrics='mse')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_data=(x_val, y_val))

# 평가, 예측
y_predict = model.predict(x_test)
mse = model.evaluate(x_test, y_test, batch_size=1)

def rmse(y_yest, y_predict):
    return np.sqrt(mean_squared_error(y_yest, y_predict)) # mse에서 루트를 씌워서 rmse를 만듬

r2_y_predict = r2_score(y_test, y_predict)

print(f"RMSE : {mse}")
print(f"RMSE : {rmse(y_test, y_predict)}") # rmse는 낮을 수록 좋음
print(f"R2 : {r2_y_predict}")
print(y_predict)

"""학습데이터에서 일부를 검증셋으로 분리하여 훈련을 진행하니 훨씬 정확도가 높아짐."""