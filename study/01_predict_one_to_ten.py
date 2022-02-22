import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 딥러닝의 기본식 => y = ax + b
# h(x) = wx + b (w는 weight 가중치를 의미, h는 hypothesis 가설을 의미

tf.random.set_seed(5) # 시드를 고정하여야 항상 동일한 결과를 받을 수 있음

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.compile(
    loss='mean_squared_error',  # 손실 함수는 평균제곱법 사용
    optimizer='adam',  # 최적화 함수는 adam 옵티마이저 사용
    metrics=['accuracy']  # 방식은 정확도를 통해 판정
)

model.fit(x, y, epochs=500, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)
print(model.predict([1, 2, 3, 4, 5, 10, 11]))

print(f"loss : {loss}")
print(f"accuracy: {acc}")

# loss : 38.5
# accuracy: 0.0

loss : 0.006178527604788542
accuracy: 0.10000000149011612
