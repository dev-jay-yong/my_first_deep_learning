from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

""" 데이터 준비 단계 """
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
y_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

# tf.random.set_seed(9) # 시드를 고정하여야 항상 동일한 결과를 받을 수 있음

# while acc < 0.5:

""" 모델 구성 단계 """
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))  # input_dim에 1을 지정함으로써 한개의 input값을 의미, 출력은 5개
model.add(Dense(3))  # 출력 3개 (첫번째 이후의 라인에는 input 개수를 지정 안해주어도 됨.)
model.add(Dense(1, activation='relu'))  # 출력 1개

model.summary()  # 모델의 구성을 확인

""" summary 결과
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 5)                 10        
                                                                 
 dense_1 (Dense)             (None, 3)                 18        
                                                                 
 dense_2 (Dense)             (None, 1)                 4         
                                                                 
=================================================================
Total params: 32
Trainable params: 32
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0

이때 각 dense 별 Param 개수가
입력값 개수 * 노드 개수 즉
dense => 1 * 5 
dense_1 => 5 * 3
dense_2 => 3 * 1 이 아닌 이유는
레이어마다 bias를 추가해서 계산해주어야 하기 때문 즉

dense => (1 + 1) * 5 
dense_1 => (5 + 1) * 3
dense_2 => (3 + 1) * 1
이 맞는 계산식이다. 

"""
""" 컴파일, 훈련 단계 """
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_test, y_test))

""" 평가, 예측 단계 """
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print(f"loss: {loss}")
print(f"acc: {acc}")
print(model.predict(x_test))