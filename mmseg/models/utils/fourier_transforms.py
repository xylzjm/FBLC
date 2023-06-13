import torch
import torch.fft as fft
import numpy as np

from .dacs_transforms import denorm_, renorm_
from mmcv.utils import print_log


def fourier_transform(data, mean, std, ratio=0.01):
    denorm_(data, mean, std)
    data = amplitude_copypaste(data[0], data[1], ratio)
    renorm_(data, mean, std)
    return data


def fftshift(x: torch.Tensor, dim=None):
    if dim is None:
        dim = tuple(range(x.ndim))
        shift = [d // 2 for d in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[d] // 2 for d in dim]

    return torch.roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim=None):
    if dim is None:
        dim = tuple(range(x.ndim))
        shift = [-(d // 2) for d in x.shape]
    elif isinstance(dim, int):
        shift = -(x.shape[dim] // 2)
    else:
        shift = [-(x.shape[d] // 2) for d in dim]

    return torch.roll(x, shift, dim)


def get_paste_bbox(amp, ratio):
    _, h, w = amp.shape
    b = np.floor(np.amin((h, w)) * ratio).astype(int)
    c_b = np.floor(b / 2.0).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1, h2 = c_h - c_b, c_h + c_b
    w1, w2 = c_w - c_b, c_w + c_b
    # print_log(f'amp shape: {img1_amp.shape}', 'mmseg')
    # print_log(f'h1: {h1}, h2: {h2}, w1: {w1}, w2: {w2}', 'mmseg')
    return h1, h2, w1, w2


def amplitude_copypaste(img1, img2, ratio):
    """Input image size: tensor of [C, H, W]"""
    assert img1.shape == img2.shape

    img1_fft = fft.fftn(img1, dim=(-2, -1))
    img2_fft = fft.fftn(img2, dim=(-2, -1))

    img1_amp, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_amp, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
    h1, h2, w1, w2 = get_paste_bbox(img1_amp, ratio)

    img1_amp = fftshift(img1_amp, dim=(-2, -1))
    img2_amp = fftshift(img2_amp, dim=(-2, -1))

    img1_amp_, img2_amp_ = torch.clone(img1_amp), torch.clone(img2_amp)
    img1_amp[..., h1:h2, w1:w2] = img2_amp_[..., h1:h2, w1:w2]

    img1_amp = ifftshift(img1_amp, dim=(-2, -1))
    img2_amp = ifftshift(img2_amp, dim=(-2, -1))

    new_img1 = img1_amp * torch.exp(1j * img1_pha)
    new_img1 = torch.real(fft.ifftn(new_img1, dim=(-2, -1)))
    new_img1 = torch.clamp(new_img1, 0, 1)

    return new_img1.unsqueeze(0)
