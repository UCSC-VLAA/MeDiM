from image_utils.process_utils import parallelize
from image_utils.file_utils import get_rand_hex
import pytest
from pathlib import Path
import numpy as np

img_path = Path("tests/high_res.png")
save_path = Path(__file__).parent / "output"


# memory = Memory(location='./cachedir', verbose=0, mmap_mode='r')
# @memory.cache
# def costly_compute(data, column):
#     """Emulate a costly function by sleeping and returning a column."""
#     time.sleep(1)
#     return data[column] + 1


def do_func(lst, idx, arr):
    print(len(lst), arr.shape)


def get_test_list(list_size):
    return [get_rand_hex() for _ in range(list_size)]


@pytest.mark.parametrize("num_processes", [1, 2, 10])
@pytest.mark.parametrize("list_size", [1, 10, 100])
def test_parallelize(num_processes, list_size):
    test_list = get_test_list(list_size)
    rng = np.random.RandomState(42)
    test_arr = rng.randn(int(1e4), 4)
    test_parallel = parallelize(do_func, num_processes=num_processes, shuffle=False, use_joblib=True)(test_list, arr=test_arr)
