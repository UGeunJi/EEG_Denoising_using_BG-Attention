import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()
# val = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()


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
    # Example: value[0], value[1], etc.\
    print()

# confirm key, value list
# print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
print(k_l[1])
print(v_l[1].keys())
print(v_l[1].values())


loss_history = v_l[1].values()                         # loss_history 값 추출
# print('val_history:', val_history)
# print(type(val_history))

loss_history_l = list(loss_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis_train = list(range(0, len(loss_history_l[0])))
x_axis_val = list(range(0, len(loss_history_l[1])))          # x축 설정
# print(x_axis_train, x_axis_val)

plt.title('Novel CNN')
plt.plot(x_axis_train, loss_history_l[0], linestyle='-', label='Training loss')
plt.plot(x_axis_val, loss_history_l[1], linestyle='-', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
