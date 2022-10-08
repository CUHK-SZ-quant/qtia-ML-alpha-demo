from typing import Callable
import time


def clock(func: Callable):

    def f(*args, **kwargs):
        t0 = time.time()
        output = func(*args, **kwargs)
        print(f'{func.__qualname__.split(".")[0]} done in {time.time()-t0:.2f}s')
        return output

    return f