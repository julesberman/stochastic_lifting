
from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers


class DNN(nn.Module):
    widths: list[int]
    activation: Callable = jax.nn.swish

    @nn.compact
    def __call__(self, x):
        depth = len(self.widths)

        A = self.activation
        for i, w in enumerate(self.widths):
            is_last = i == depth - 1
            L = nn.Dense(w)
            x = L(x)
            if not is_last:
                x = A(x)

        return x
