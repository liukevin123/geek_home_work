import time

import numpy as np
import pandas as pd


def operate():
    #生成0或1，100000行1列的矩阵
    x = np.random.randint(2, size=(100000, 1))
    # 生成0或1，100000行1列的矩阵
    y = np.random.randint(2, size=(100000, 1))
    #转换成DataFrame
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start = time.time()
    #调用cython方法
    res = home_work.target_mean(data, 'y', 'x')

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    operate()
    list = []
    
