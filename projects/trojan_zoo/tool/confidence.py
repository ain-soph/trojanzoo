import torch
import math

from typing import List

single: float = 0.0
vec: List[float] = [22.2281, 54.7394, 30.2743, 41.7436, 30.8135, 76.0631, 45.3438, 46.7116,
                    44.5743, 41.3735]
vec = torch.tensor(vec)

std = vec.std()
mean = vec.mean()

confidence = 0.5 + 0.5 * torch.erf((single - mean) / std / math.sqrt(2))
confidence = 1 - confidence
print(confidence)
