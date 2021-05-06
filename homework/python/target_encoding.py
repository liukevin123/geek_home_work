# coding = 'utf-8'
import time

import numpy as np
import pandas as pd

def target_mean(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    print("0/1求和结果："+str(value_dict))
    print("0/1条数："+str(count_dict))
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)

    return result


def main():
    # 生成0或1，100000行1列的矩阵
    y = np.random.randint(2, size=(100000, 1))
    x = np.random.randint(2, size=(100000, 1))
    # 转换成DataFrame
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    target_mean(data, 'y', 'x')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("python版本运行时长：" + str(end - start))
