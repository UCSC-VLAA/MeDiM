import random
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable

import torch
from joblib import Parallel, delayed


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def cuda_timer_func(func):
    def wrap_func(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        result = func(*args, **kwargs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time_ms:.3f} ms")
        return result

    return wrap_func


def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def parallelize(func: Callable, num_processes=1, shuffle=False, custom_chunking=False, use_joblib=False):
    def wrapper(iterable, **kwargs):
        if use_joblib:
            executor = Parallel(n_jobs=num_processes)
            tasks = (delayed(func)(iterable, idx, **kwargs) for idx in range(len(iterable)))
            return executor(tasks)
        else:
            with Pool(num_processes) as p:
                num_arr = list(range(len(iterable)))
                if shuffle:
                    random.shuffle(num_arr)
                for _ in p.imap_unordered(partial(func, **kwargs), split_range(num_arr, num_processes) if custom_chunking else iterable):
                    pass

    return wrapper
