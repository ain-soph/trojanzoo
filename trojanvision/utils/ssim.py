#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_zero_padding(kernel_size: int) -> int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2


class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    # Examples::

    #     >>> input1 = torch.rand(1, 4, 5, 5)
    #     >>> input2 = torch.rand(1, 4, 5, 5)
    #     >>> ssim = kornia.losses.SSIM(5, reduction='none')
    #     >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(self, window_size: int = 11, reduction: str = "none", max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = _compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img1):
            raise TypeError(f'Input img1 type is not a torch.Tensor. Got {type(img1)}')
        if not torch.is_tensor(img2):
            raise TypeError(f'Input img2 type is not a torch.Tensor. Got {type(img2)}')
        if not img1.dim() == 4:
            raise ValueError(f'Invalid img1 shape, we expect BxCxHxW. Got: {img1.shape}')
        if not img2.dim() == 4:
            raise ValueError(f'Invalid img2 shape, we expect BxCxHxW. Got: {img2.shape}')
        if not img1.shape == img2.shape:
            raise ValueError(f'img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}')
        if not img1.device == img2.device:
            raise ValueError(f'img1 and img2 must be in the same device. Got: {img1.device} and {img2.device}')
        if not img1.dtype == img2.dtype:
            raise ValueError(f'img1 and img2 must be in the same dtype. Got: {img1.dtype} and {img2.dtype}')
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2
        ssim_map = ((2. * mu1_mu2 + self.C1) * (2. * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        loss = torch.clamp(-ssim_map + 1., min=0, max=1) / 2.
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss


######################
# functional interface
######################


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int, reduction: str = "none", max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`~kornia.losses.SSIM` for details.
    """
    return SSIM(window_size, reduction, max_val)(img1, img2)


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError(f'input should be at least 2D tensor. Got {input.size()}')
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def gaussian(window_size: int, sigma: float):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    # Examples::

    #     >>> kornia.image.get_gaussian_kernel(3, 2.5)
    #     tensor([0.3243, 0.3513, 0.3243])

    #     >>> kornia.image.get_gaussian_kernel(5, 1.5)
    #     tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            f"Got {kernel_size}"
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size: tuple[int, int], sigma: tuple[float, float],
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    # Examples::

    #     >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
    #     tensor([[0.0947, 0.1183, 0.0947],
    #             [0.1183, 0.1478, 0.1183],
    #             [0.0947, 0.1183, 0.0947]])

    #     >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
    #     tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
    #             [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
    #             [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


def compute_padding(kernel_size: tuple[int, int]) -> list[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]


def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f'Input kernel type is not a torch.Tensor. Got {type(kernel)}')
    if not isinstance(border_type, str):
        raise TypeError(f'Input border_type is not string. Got {type(kernel)}')
    if not input.dim() == 4:
        raise ValueError(f'Invalid input shape, we expect BxCxHxW. Got: {input.shape}')
    if not kernel.dim() == 3:
        raise ValueError(f'Invalid kernel shape, we expect 1xHxW. Got: {kernel.shape}')
    borders_list: list[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError(f"Invalid border_type, we expect the following: {borders_list}."
                         f"Got: {border_type}")
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: list[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape
    # convolve the tensor with the kernel. Pick the fastest alg
    kernel_numel: int = height * width
    if kernel_numel > 81:
        return F.conv2d(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
    return F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)
