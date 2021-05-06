import time

import numpy as np
import pandas as pd


def operate():
    x = np.random.randint(2, size=(100000, 1))
    y = np.random.randint(2, size=(100000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start = time.time()
    res = home_work.target_mean(data, 'y', 'x')

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    operate()
    list = []
    
