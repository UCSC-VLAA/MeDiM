import torch
import numpy as np
import random
from . import is_tensor, is_ndarray, is_arr


def generic_print(self, arr_values):
    assert is_arr(self)

    if len(self.shape) == 0:
        return arr_values

    if is_ndarray(self):
        lib = np
        num_elements = lib.prod(self.shape)
        device = ""
    else:
        lib = torch
        num_elements = lib.prod(torch.tensor(list(self.shape))).item()
        device = f"{self.device.type} "

    if self.dtype in (np.bool_, torch.bool):
        specific_data = f" sum: {self.sum()}, unique: {len(lib.unique(self))},"
    elif (is_ndarray(self) and np.issubdtype(self.dtype, np.integer)) or (is_tensor(self) and not torch.is_floating_point(self)):
        specific_data = f" unique: {len(lib.unique(self))},"
    else:
        specific_data = f" avg: {self.mean():.3f},"

    shape_str = ",".join([str(self.shape[i]) for i in range(len(self.shape))])
    finite_str = "finite" if lib.isfinite(self).all() else "non-finite"
    basic_info = f"[{shape_str}] {self.dtype} {device}{finite_str}"
    numerical_info = f"\nelems: {num_elements},{specific_data} min: {self.min():.3f}, max: {self.max().item():.3f}"

    def get_first_and_last_lines(text):
        if text.count("\n") > 4:
            lines = text.split("\n")
            first_lines = "\n".join(lines[:2])
            end_lines = "\n".join(lines[-2:])
            return f"{first_lines} ...\n{end_lines}"
        else:
            return text

    return basic_info + numerical_info + f"\n{arr_values}\n" + basic_info


torch.set_printoptions(sci_mode=False, precision=3, threshold=10, edgeitems=2, linewidth=120)
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: generic_print(self, normal_repr(self))

np.set_printoptions(suppress=True, precision=3, threshold=10, edgeitems=2, linewidth=120)

normal_repr_ = np.ndarray.__str__
if int(np.__version__.split(".")[0]) >= 2:
    np.set_printoptions(override_repr=lambda self: generic_print(np.array(self), normal_repr_(np.array(self))))
else:
    np.set_string_function(lambda self: generic_print(self, normal_repr_(self)), repr=True)


def disable():
    torch.set_printoptions(profile="default")
    torch.Tensor.__repr__ = normal_repr
    if int(np.__version__.split(".")[0]) >= 2:
        pass
        # TODO: Currently broken for numpy 2.x
        # np.set_printoptions(formatter={
        #     'int_kind': normal_repr_,
        #     'float_kind': normal_repr_,
        # })
    else:
        np.set_string_function(normal_repr_, repr=True)


def set_random_seeds():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
