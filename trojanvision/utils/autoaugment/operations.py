#!/usr/bin/env python3

from . import operations

from trojanvision.utils.transform import cutout
# import torchvision.transforms.functional_tensor as F_t
# import torchvision.transforms.functional as F
from . import functional as F
from . import functional_tensor as F_t

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.autograd import Function


PRIMITIVES = ['Zero',
              'ShearX', 'ShearY',
              'TranslateX', 'TranslateY',
              'HorizontalFlip', 'Rotate',
              'Brightness', 'Contrast', 'AutoContrast',
              'Color', 'Sharpness',
              'Posterize', 'Solarize',
              'Equalize', 'Invert',
              'Cutout', 'SamplePairing']


def get_op(primitive: str, **kwargs) -> 'Operation':
    OpClass: type[Operation] = getattr(operations, primitive)
    return OpClass(**kwargs)


class Clamp(nn.Module):
    def __init__(self, _min: float = 0.0, _max: float = 1.0):
        super().__init__()
        self._min = _min
        self._max = _max
        self.scale = _max - _min

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.clamp(0.0, 1.0) * self.scale + self._min


class _STE(Function):
    @staticmethod
    def forward(ctx,
                tensor: torch.Tensor,
                param: torch.Tensor):
        ctx.shape = param.shape
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone(), grad_output.sum_to_size(ctx.shape)


def ste(tensor: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
    if len(params) == 0:
        return tensor.clone()
    return ste(_STE.apply(tensor, params[0]), *params[1:])


class Operation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return _input

    def create_transform(self):
        return self


class MagnitudeOperation(Operation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 1.0),
                 **kwargs):
        super().__init__()
        self.value_range = value_range
        self.magnitude = nn.Parameter(torch.rand(1))
        parametrize.register_parametrization(self, 'magnitude',
                                             Clamp(*value_range))

    def set_magnitude(self, magnitude: float):
        if parametrize.is_parametrized(self, 'magnitude'):
            mag = (magnitude - self.value_range[0]) / (self.value_range[1] - self.value_range[0])
            self.parametrizations.magnitude.original.data.fill_(mag)
        else:
            self.magnitude.data.fill_(magnitude)

    def create_transform(self):
        parametrize.remove_parametrizations(self, 'magnitude')
        self.requires_grad_(False)
        self.cpu()
        self.eval()
        return self


class Zero(Operation):
    def create_transform(self):
        return nn.Identity()


class ShearX(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (-16.7, 16.7),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        shear = torch.cat([self.magnitude, torch.zeros_like(self.magnitude)])
        return F.affine(_input, shear=shear)


class ShearY(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (-16.7, 16.7),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        shear = torch.cat([torch.zeros_like(self.magnitude), self.magnitude])
        return F.affine(_input, shear=shear)


class TranslateX(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (-0.45, 0.45),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        translate = torch.cat([self.magnitude * _input.shape[-1],
                               torch.zeros_like(self.magnitude)])
        return F.affine(_input, translate=translate)


class TranslateY(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (-0.45, 0.45),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        translate = torch.cat([torch.zeros_like(self.magnitude),
                               self.magnitude * _input.shape[-2]])
        return F.affine(_input, translate=translate)


class HorizontalFlip(Operation):
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # return _input.flip(-1)
        return F_t.hflip(_input)


class Rotate(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (-30.0, 30.0),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # not differential w.r.t. self.magnitude
        # return F.rotate(_input, angle=self.magnitude)
        return F.affine(_input, angle=self.magnitude)

    #     rot = self.magnitude * torch.pi / 180    # radians
    #     matrix = self._get_matrix(rot)
    #     F_t.rotate(_input, matrix=matrix)

    # @staticmethod
    # def _get_matrix(rot: torch.Tensor) -> list[float]:
    #     cos, sin = rot.cos(), rot.sin()
    #     zero = torch.zeros_like(rot)
    #     matrix = torch.cat([cos, sin, zero, -sin, cos, zero])
    #     return matrix

    # @staticmethod
    # def rotate(img: torch.Tensor, matrix: torch.Tensor,
    #            interpolation: str = 'nearest',
    #            expand: bool = False, fill: None | list[float] = None
    #            ) -> torch.Tensor:
    #     w, h = img.shape[-1], img.shape[-2]
    #     ow, oh = F_t._compute_output_size(matrix, w, h) if expand else (w, h)
    #     theta = matrix.view(1, 2, 3)
    #     # grid will be generated on the same device as theta and img
    #     grid = F_t._gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
    #     return F_t._apply_grid_transform(img, grid, interpolation, fill=fill)


class Invert(Operation):
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # return 1.0 - _input
        return F_t.invert(_input)


class Color(MagnitudeOperation):
    # alias of Saturate
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # degenerate = F_t.rgb_to_grayscale(_input)
        # result: torch.Tensor = self.magnitude * _input \
        #     + (1 - self.magnitude) * degenerate
        # return result
        return F_t.adjust_saturation(_input, saturation_factor=self.magnitude)


class Brightness(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 2.0),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # return (self.magnitude * _input).clamp(0.0, 1.0)
        return F_t.adjust_brightness(_input, brightness_factor=self.magnitude)


class Contrast(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 2.0),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # degenerate = F_t.rgb_to_grayscale(_input).flatten(1).mean()
        # result: torch.Tensor = self.magnitude * _input \
        #     + (1 - self.magnitude) * degenerate
        # return result.clamp(0.0, 1.0)
        return F_t.adjust_contrast(_input, contrast_factor=self.magnitude)


class AutoContrast(Operation):
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return F_t.autocontrast(_input)


class Sharpness(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 2.0),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return F_t.adjust_sharpness(_input, sharpness_factor=self.magnitude)


class Equalize(Operation):
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return ste(F_t.equalize((_input * 255).byte()).float().div(255), _input)


class Posterize(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 8.0),
                 **kwargs):
        super().__init__(value_range=value_range, ste=ste, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # not differential w.r.t. self.magnitude
        return ste(F_t.posterize((_input * 255).byte(), bits=int(self.magnitude)).float().div(255),
                   _input, self.magnitude)


class Solarize(MagnitudeOperation):
    """Invert all pixel values above a threshold."""

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        # not differential w.r.t. self.magnitude
        # return torch.where(_input > self.magnitude, 1 - _input, _input)
        return ste(F_t.solarize(_input, threshold=self.magnitude), _input, self.magnitude)


class Cutout(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 0.2),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        h, w = _input.shape[-2:]
        length = (self.magnitude * torch.tensor([h, w], dtype=torch.float, device=self.magnitude.device)).int()
        return ste(cutout(_input, length), _input, self.magnitude)


class SamplePairing(MagnitudeOperation):
    def __init__(self, value_range: tuple[float, float] = (0.0, 0.4),
                 **kwargs):
        super().__init__(value_range=value_range, **kwargs)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(_input.shape[0])
        degenerate = _input[idx]
        result: torch.Tensor = (1 - self.magnitude) * _input \
            + self.magnitude * degenerate
        return result
