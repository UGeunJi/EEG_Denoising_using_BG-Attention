import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()


# key, value list
k_l = []
v_l = []

# Access key-value pairs 
for key, value in loss_history.items():                 # np.ndarray에서 키, 벨류값 구분   
    # print(f"Key: {key}")
    k_l.append(key)                                     
    # print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.
    print()

# confirm key, value list
# print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
# print(v_l[0].keys())
# print(v_l[0].values())

mse_history = v_l[0].values()                           # mse 값 추출
# print('mse_history:', mse_history)
# print(type(mse_history))

mse_history_l = list(mse_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis = list(range(0, len(mse_history_l[0])))          # x축 설정
# print(x_axis)

plt.title('Novel CNN EOG Grads MSE history')
plt.plot(x_axis, mse_history_l[0], linestyle='-')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()  
