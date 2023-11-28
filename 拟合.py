import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from math import sqrt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# pd.set_option('display.max_rows', None) # 取消展示行数限制
# pd.set_option('display.max_columns', None) # 取消展示列数限制

# 在numpy中进行任意底数的对数运算
def log(base,x):
    return np.log(x) / np.log(base)

# 自定义函数
def func(d, n, A):
    return A - 10 * n * log(10, d) # 此处的A是实际计算出的1m处信号强度的均值

# 封装读取轩数据的函数
def read(filename):
    data = pd.read_excel(filename, names = ['时间', '信号强度', 'Unknown'])
    data = data.drop('Unknown', axis = 1)
    data = data.set_index('时间')
    data = data.resample('min').mean().to_period('min')
    for i in range(len(filename)):
        if filename[i] != '_':
            continue
        else:
            dist = filename[:i - 1]
            dist = float(dist)
            time = '2023-11-09' + ' ' + filename[i + 1 : i + 3] + ':' + filename[i + 4 : i + 6]
            time = pd.to_datetime(time)
            break
    rssi = data.loc[time]

    return dist, rssi

def read_sun(filename):
    # 注意逗号和括号是中文的
    maodian = [(4.86, 8.4, 2.35), (13.3, 8.4, 2.35), (21.16, 7, 2), (4.86, 0.4, 2.25), (13.3, 0.4, 2.35), (20.9, 3.1, 2)]
    i = 0
    while filename[i] != '，':
        i = i + 1
    j = i + 1
    while filename[j] != '）':
        j = j + 1
    # 此时i的位置在逗号，j的位置在右括号
    token1 = filename[1 : i]
    token2 = filename[i + 1 : j]
    token1 = float(token1)
    token2 = float(token2)
    ceshidian = (token1, token2, 0)
    data = pd.read_excel(filename, index_col = 0)

    rssi = np.array([])
    for i in range(data.shape[0]):
        rssi = np.append(rssi, data.iloc[i].mean())
    dist = np.array([])
    for dian in maodian:
        dist = np.append(dist, sqrt((dian[0] - ceshidian[0]) ** 2 + (dian[1] - ceshidian[1]) ** 2 + (dian[2] - ceshidian[2]) ** 2))

    return dist, rssi

xdata = np.array([])
ydata = np.array([])

# 数据处理1：轩的数据
filelist = [
    '0.5m_21_02.xls',
    '1m_21_04.xls',
    '1.5m_21_06.xls',
    '2m_21_08.xls',
    '2.5m_20_58.xls',
    '3m_21_10.xls',
    '3.5m_21_12.xls',
    '4m_21_14.xls',
    '4.5m_21_16.xls',
    '5m_21_17.xls',
]
# 读取轩的数据
for filename in filelist:
    x, y = read(filename)
    xdata = np.append(xdata, x)
    ydata = np.append(ydata, y)

# 数据处理2：读取大佬数据
data = pd.read_excel('大佬数据.xlsx')
# print(data)
data = data.drop('设备：HUAWEI MATE30', axis = 1)
data.columns = ['距离', '信号强度']
xdata = np.append(xdata, data['距离'])
ydata = np.append(ydata, data['信号强度'])

# 数据处理3：读取孙欢康数据
filelist1 = [
    '（2.1，6.79）.xls',
    '（2.09，1.67）.xls',
    '（4.87，4.3）.xls',
    '（8.0，4.3）.xls',
    '（12.87，4.5）.xls',
    '（15.3，4.3）.xls',
    '（18.38，4.02）.xls',
    '（19.4，4.4）.xls'
]
for filename in filelist1:
    x, y = read_sun(filename)
    xdata = np.append(xdata, x)
    ydata = np.append(ydata, y)

# 对数据按照x的升序进行重组
sorted_index = np.argsort(xdata)
tx = xdata[sorted_index[::1]]
xdata = tx
ty = ydata[sorted_index[::1]]
ydata = ty

all_data = pd.DataFrame([])
all_data['距离'] = xdata
all_data['rssi值'] = ydata
all_data.to_excel('汇总数据.xlsx')

# rng = np.random.default_rng() # 随机数生成对象
# y_noise = 0.2 * rng.normal(size=xdata.size)
# ydata = y + y_noise

# 拟合
popt, pcov = curve_fit(func, xdata, ydata)

# 可视化
plt.plot(xdata, ydata, 'b-', label='data')
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: n=%5.3f A=%5.3f' % tuple(popt))

plt.xlabel('距离')
plt.ylabel('rssi值')
plt.legend()
plt.show()

# 11.16 n = 2.401 X = -4.415
# 11.17 n = 2.567 X = -15.715
# 11.18 n = 2.907 X = -13.324