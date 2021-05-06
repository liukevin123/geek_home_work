# distutils: language=c++
import numpy as np
cimport numpy as np
import pandas as pd
cimport pandas as pd

cpdef target_mean(data,y_name,x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow),dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(np.zeros(y_name),dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(np.zeros(x_name),dtype=np.float64)

    target_mean_impl(result, y, x, nrow)
    return result


cdef void target_mean_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)