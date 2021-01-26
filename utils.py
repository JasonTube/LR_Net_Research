import torch
import torch.nn.functional as F


def get_indices(X_shape, HF, WF, stride, pad, device):
    m, n_C, n_H, n_W = X_shape
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    level1 = torch.arange(HF, device=device).repeat_interleave(WF)
    level1 = level1.repeat(n_C)
    everyLevels = stride * torch.arange(out_h, device=device).repeat_interleave(out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    slide1 = torch.arange(WF, device=device).repeat(HF)
    slide1 = slide1.repeat(n_C)
    everySlides = stride * torch.arange(out_w, device=device).repeat(out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = torch.arange(n_C, device=device).repeat_interleave(HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad, device):
    X_padded = F.pad(X, [0, 0, 0, 0, pad, pad, pad, pad], mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad, device)
    cols = X_padded[:, d, i, j]
    cols = torch.cat(tuple(cols), dim=-1)
    return cols
