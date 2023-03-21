# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ._version import __version__

import importlib


try:
    import torch

    torch.multiprocessing.set_start_method("fork", force=True)
except:
    import sys

    print(
        "Your OS does not support multiprocessing based on fork, please use num_workers=0",
        file=sys.stderr,
        flush=True,
    )
