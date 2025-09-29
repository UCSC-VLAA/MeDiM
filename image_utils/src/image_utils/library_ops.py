def enable():
    try:
        import importlib

        if importlib.util.find_spec("numpy") is not None:
            import numpy as np

            np.set_printoptions(suppress=True, precision=3, threshold=10, edgeitems=2, linewidth=120)
            try:
                from lovely_numpy import lovely, set_config

                set_config(repr=lovely)
            except:
                print(f"Failed to enable lovely_numpy.")

        if importlib.util.find_spec("torch") is not None:
            import torch

            torch.set_printoptions(sci_mode=False, precision=3, threshold=10, edgeitems=2, linewidth=120)
            import lovely_tensors as lt

            lt.monkey_patch()
    except ImportError as e:
        print("lovely_tensors is not installed. Run `pip install lovely-tensors` if you wish to use it.")


enable()
