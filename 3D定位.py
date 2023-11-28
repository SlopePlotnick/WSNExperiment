#  Rssi 3D定位算法
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# 锚节点
class node:
    x=0
    y=0
    z=0
    D=0

# 观察目标
class target:
    x=0
    y=0
    z=0

# 两点之间距离
def Get_DIST(A,B):
    dist = ((A.x - B.x) ** 2 + (A.y - B.y) ** 2 + (A.z - B.z) ** 2)**0.5
    return dist

# 由Rssi的值计算距离d
def GetDistByRssi(rssi):
    A = -49.609
    n = 2.907
    return 10 ** ((A - rssi) / (10 * n))

xdata = np.array([])
ydata = np.array([])
for i in range(4, 7):
    # 绘制图层 锚节点个数相同的情况绘制在一张图里
    plt.figure(dpi=80, figsize = (20, 10))

    # 将当前锚节点个数加入xdata中
    xdata = np.append(xdata, i)
    # 锚节点
    Node_number = i  # 观测站个数，至少4个
    Node = []  # 存储锚节点的列表

    node1 = node()
    node1.x, node1.y, node1.z = 4.86, 8.4, 2.35
    node1.D = node1.x ** 2 + node1.y ** 2 + node1.z ** 2
    Node.append(node1)

    node2 = node()
    node2.x, node2.y, node2.z = 13.3, 8.4, 2.35
    node2.D = node2.x ** 2 + node2.y ** 2 + node2.z ** 2
    Node.append(node2)

    node3 = node()
    node3.x, node3.y, node3.z = 21.16, 7, 2
    node3.D = node3.x ** 2 + node3.y ** 2 + node3.z ** 2
    Node.append(node3)

    node4 = node()
    node4.x, node4.y, node4.z = 4.86, 0.4, 2.25
    node4.D = node4.x ** 2 + node4.y ** 2 + node4.z ** 2
    Node.append(node4)

    node5 = node()
    node5.x, node5.y, node5.z = 13.3, 0.4, 2.35
    node5.D = node5.x ** 2 + node5.y ** 2 + node5.z ** 2
    Node.append(node5)

    node6 = node()
    node6.x, node6.y, node6.z = 20.9, 3.1, 2
    node6.D = node6.x ** 2 + node6.y ** 2 + node6.z ** 2
    Node.append(node6)

    # for i in range(6):
    #     print(Node[i].x)
    #     print(Node[i].y)
    #     print(Node[i].z)
    #     print(Node[i].D)
    #     print()

    # 测试点
    filelist = [
        '（2.1，6.79）.xls',
        '（2.09，1.67）.xls',
        '（4.87，4.3）.xls',
        '（8.0，4.3）.xls',
        '（12.87，4.5）.xls',
        '（15.3，4.3）.xls',
        '（18.38，4.02）.xls',
        '（19.4，4.4）.xls'
    ]
    error = np.array([])
    for filename in filelist:
        idx = filelist.index(filename) # 求解当前文件名的下标 用来绘制子图

        # 从文件名中解析出测试点坐标
        i = 0
        while filename[i] != '，':
            i = i + 1
        j = i + 1
        while filename[j] != '）':
            j = j + 1
        token1 = filename[1: i]
        token1 = float(token1)
        token2 = filename[i + 1: j]
        token2 = float(token2)
        Target = target()
        Target.x, Target.y, Target.z = token1, token2, 0
        data = pd.read_excel(filename, index_col=0)

        # ZZ存储各锚节点信号值的均值
        ZZ = np.array([])
        for i in range(Node_number):
            ZZ = np.append(ZZ, np.mean(data.iloc[i]))

        # 根据Rssi求各锚节点的观测距离
        Zd = np.zeros(Node_number)  # 计算的距离
        for i in range(Node_number):
            Zd[i] = GetDistByRssi(ZZ[i])
            # Zd[i] = Get_DIST(Node[i], Target)
        # print(Zd)

        # 根据观测距离用最小二乘法估计目标位置
        H = np.zeros((Node_number - 1, 3))
        b = np.zeros(Node_number - 1)
        end = Node_number - 1
        for i in range(0, Node_number - 1):
            # A 矩阵 和 B 矩阵
            H[i - 1] = [(-2) * (Node[i].x - Node[end].x), (-2) * (Node[i].y - Node[end].y),
                        (-2) * (Node[i].z - Node[end].z)]
            b[i - 1] = (Zd[i] ** 2) - (Zd[end] ** 2) + Node[end].D - Node[i].D

        # Estimate=inv(H'*H)*H'*b #估计目标位置
        H_T = np.transpose(H)  # H'
        # np.dot( np.transpose(H), H ) # (H'*H)
        # np.linalg.inv(np.dot( np.transpose(H), H))  # inv(H'*H)
        HH = np.dot(H_T, H)
        HH_inv = np.linalg.inv(HH)
        HHH = np.dot(HH_inv, H_T)
        Estimate = np.dot(HHH, b)

        print(Estimate)

        Est_Target = target()
        Est_Target.x = Estimate[0]
        Est_Target.y = Estimate[1]
        Est_Target.z = 0
        # Est_Target.z = Estimate[2]

        ########画图########
        ax = plt.subplot(2, 4, idx + 1, projection='3d')  # 设置3D绘图空间
        plt.grid(linestyle="dotted")
        # 设置坐标范围
        ax.set_zlim3d(0, 3)
        # 标注坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 画锚节点
        for i in range(Node_number):
            # print(Node[i].x, Node[i].y, Node[i].z)
            a = ax.scatter(Node[i].x, Node[i].y, Node[i].z, linewidths=5)
        # 画目标节点 真实值和预测值
        b = ax.scatter(Target.x, Target.y, Target.z, marker='*', s=100)
        c = ax.scatter(Est_Target.x, Est_Target.y, Est_Target.z, marker='D', s=80)
        ax.plot([Est_Target.x, Target.x], [Est_Target.y, Target.y], [Est_Target.z, Target.z], color='r')

        plt.tick_params(labelsize=8)
        ax.legend([a, b, c], ['观测站', '目标位置', '估计位置'], prop={"size": 10})
        Error_Dist = Get_DIST(Est_Target, Target)  # 计算误差
        Error_2D = sqrt((Target.x - Est_Target.x) ** 2 + (Target.y - Est_Target.y) ** 2)
        plt.title(
            '真实坐标：' + '(' + str(Target.x) + ',' + str(Target.y) + ',' + str(Target.z) + ')' + '\n锚节点数：' + str(
                Node_number) + '\nerror=' + str(Error_Dist) + 'm', fontdict={"size": 10})
        # plt.title('锚节点数：' + str(Node_number) + '\n3d error=' + str(Error_Dist) + 'm' + '\n2d error=' + str(Error_2D) + 'm', fontdict={"size": 15})

        error = np.append(error, Error_Dist)

    # 每一轮结束之后 将8个误差的均值加入ydata中
    ydata = np.append(ydata, np.mean(error))

    # 展示当前锚节点个数下绘制的左右子图
    plt.show()

# 绘制平均误差随锚节点个数变化的曲线
plt.figure()
plt.plot(xdata, ydata, 'b-')
plt.xticks([4, 5, 6])
plt.xlabel('锚节点个数')
plt.ylabel('平均误差')
for i in range(len(xdata)):
    plt.text(xdata[i], ydata[i], '(' + str(int(xdata[i])) + ',' + str('%.3f'%ydata[i]) + ')', fontsize = 8)
plt.show()