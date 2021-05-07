# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
cpdef target_mean(data,y_name,x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow),dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name],dtype=np.int32)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name],dtype=np.int32)

    target_mean_impl(result, y, x, nrow)
    return result


cdef void target_mean_impl(double[:] result, double[:] y, double[:] x, const long nrow):
#    cdef dict value_dict = dict()
#    cdef dict count_dict = dict()
    cdef unordered_map[double, double] value_map
    cdef unordered_map[double, double] count_map
    cdef long i
    for i in range(nrow):
        if value_map[x[i]] == None:
            value_map[x[i]] = y[i]
            count_map[x[i]] = 1
        else:
            value_map[x[i]] += y[i]
            count_map[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_map[x[i]] - y[i])/(count_map[x[i]]-1)