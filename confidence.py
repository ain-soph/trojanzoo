import torch
import math

from typing import List

single: float = 0.0
vec: List[float] = []
vec = torch.tensor(vec)

std = vec.std()
mean = vec.mean()

confidence = 0.5 + 0.5 * torch.erf((single - mean) / std / math.sqrt(2))
confidence = 1 - confidence
print(confidence)
