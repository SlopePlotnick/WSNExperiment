import pandas as pd
import numpy as np
from math import sqrt

# maodian = [(4.86, 8.4, 2.35), (13.3, 8.4, 2.35), (21.16, 7, 2), (4.86, 0.4, 2.25), (13.3, 0.4, 2.35), (20.9, 3.1, 2)]
# ceshidian = (12.85, 4.11, 0)
#
# dist = np.array([])
# for dian in maodian:
#     dist = np.append(dist, sqrt((dian[0] - ceshidian[0]) ** 2 + (dian[1] - ceshidian[1]) ** 2 + (dian[2] - ceshidian[2]) ** 2))
# dist2id = pd.DataFrame([])
# dist2id['距离'] = dist
# id = pd.Series(['DO:7E:01:47:7E', 'D0:7E:01:00:AC:2D', 'D0:7E:01:01:A7:25', 'DO:7E:01:00:2D:02', 'D0:7E:01:00:13:D6', 'D0:7E:01:01:36:F1'])
# dist2id['传感器编号'] = id
# print(dist2id['距离'])
#
# file = open('2023年11月09日蓝牙扫描记录.txt', 'r')
# lines = file.readlines()
# print(lines)

# def log(base,x):
#     return np.log(x) / np.log(base)
#
# # 自定义函数
# def func(d):
#     return -62.933333 - 10 * 2.401 * log(10, d) - (-4.415) # 此处的A是实际计算出的1m处信号强度的均值
#
# def rev(rssi):
#     return 10 ** ((-62.933333 - (-4.415) - rssi) / (10 * 2.401))
#
# d = sqrt((2.09 - 4.86) ** 2 + (1.67 - 8.4) ** 2 + (0 - 2.35) ** 2)
#
# print(rev(-77))

# str = "2023-11-09" + ' ' +  "20:36"
# print(str[0:3])

x = 3 ** 2 + 4 ** 2 + 5 ** 2
print(x)