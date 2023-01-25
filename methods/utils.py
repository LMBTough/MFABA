import time


def get_time(fun_name):
    def warpper(fun):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = fun(*arg, **kwarg)
            e_time = time.time()
            # print('{} ：{} FPS'.format(fun_name, math.floor(1/(e_time - s_time)* 100) / 100))            return res
            print(f"{fun_name}: {e_time - s_time} s")
            return res
        return inner
    return warpper