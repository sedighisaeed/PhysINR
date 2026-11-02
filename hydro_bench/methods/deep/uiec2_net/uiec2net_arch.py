"""UIEC^2-Net architecture, vendored self-contained copy.

Copied from the official repository (``core/Models/UWModels/UIEC2Net.py`` plus
the ``RGB2HSV``/``HSV2RGB`` blocks of ``core/Models/blocks/`` in
https://github.com/BIGWangYuDong/UWEnhancement) so the released checkpoint
maps 1:1 onto the constructed model.

Two changes only, both mechanical:

* the original calls ``.cuda()`` on tensors built inside ``forward``,
  ``piece_function_org`` and ``sgn_m``, which would force a CUDA device on a
  host that has none; those are replaced by ``torch.zeros_like`` /
  ``torch.ones_like`` so the constants follow the input device instead;
* the framework registry (``core.Models.builder``, ``BaseNet``) is dropped.
  ``backbone`` is ``None`` in the released config, so ``build_backbone`` is
  never called and ``BaseNet.__init__`` does nothing beyond
  ``nn.Module.__init__``.

No weight, layer, hyperparameter or formula differs from the published
network; the HSV round trip and the piecewise colour-transfer functions are
transcribed unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB2HSV(nn.Module):
    """RGB -> HSV, transcribed from ``core/Models/blocks/RGB2HSV.py``."""

    def forward(self, rgb):
        batch, c, w, h = rgb.size()
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        V, max_index = torch.max(rgb, dim=1)
        min_rgb = torch.min(rgb, dim=1)[0]
        v_plus_min = V - min_rgb
        S = v_plus_min / (V + 0.0001)
        H = torch.zeros_like(rgb[:, 0, :, :])
        mark = max_index == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (v_plus_min[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360
        return torch.cat([H.view(batch, 1, w, h), S.view(batch, 1, w, h),
                          V.view(batch, 1, w, h)], 1)


def function_delta(x):
    return torch.clamp(x, min=0, max=60)


class HSV2RGB(nn.Module):
    """HSV -> RGB, transcribed from ``core/Models/blocks/HSV2RGB.py``."""

    def forward(self, hsv):
        batch, c, w, height = hsv.size()
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        htemp = (h * 360) % 360
        h = htemp / 360
        vs = torch.div(torch.mul(v, s), 60)
        R1_delta = function_delta(torch.add(torch.mul(h, 360), -60))
        R2_delta = function_delta(torch.add(torch.mul(h, 360), -240))

        G1_delta = function_delta(torch.add(torch.mul(h, 360), 0))
        G2_delta = function_delta(torch.add(torch.mul(h, 360), -180))

        B1_delta = function_delta(torch.add(torch.mul(h, 360), -120))
        B2_delta = function_delta(torch.add(torch.mul(h, 360), -300))

        one_minus_s = 1.0 - s
        R_1 = v - torch.mul(vs, R1_delta)
        R_2 = torch.mul(vs, R2_delta)
        R = R_1 + R_2

        G_1 = torch.mul(v, one_minus_s) + torch.mul(vs, G1_delta)
        G_2 = torch.mul(vs, G2_delta)
        G = G_1 - G_2

        B_1 = torch.mul(v, one_minus_s) + torch.mul(vs, B1_delta)
        B_2 = torch.mul(vs, B2_delta)
        B = B_1 - B_2

        R = torch.reshape(R, (batch, 1, w, height))
        G = torch.reshape(G, (batch, 1, w, height))
        B = torch.reshape(B, (batch, 1, w, height))
        return torch.cat([R, G, B], 1)


def piece_function_org(x_m, para_m, M):
    b, c, w, h = x_m.shape
    r_m = para_m[:, 0].view(b, c, 1, 1).expand(b, c, w, h)
    for i in range(M - 1):
        para = (para_m[:, i + 1] - para_m[:, i]).view(b, c, 1, 1).expand(
            b, c, w, h)
        r_m = r_m + para * sgn_m(M * x_m - i * torch.ones_like(x_m))
    return r_m


def sgn_m(x):
    zero_lab = torch.zeros_like(x)
    s_t = torch.where(x < 0, zero_lab, x)
    one_lab = torch.ones_like(x)
    return torch.where(s_t > 1, one_lab, s_t)


class UIEC2Net(nn.Module):
    """UIEC^2-Net: dual colour-space enhancement (Wang et al., 2021).

    RGB stream + HSV piecewise colour-transfer stream, fused by a learned
    per-pixel confidence map.
    """

    def __init__(self):
        super().__init__()
        self.rgb2hsv = RGB2HSV()
        self.hsv2rgb = HSV2RGB()

        self.norm_batch = nn.InstanceNorm2d

        self.rgb_norm_batch1 = self.norm_batch(64)
        self.rgb_con1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch2 = self.norm_batch(64)
        self.rgb_con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch3 = self.norm_batch(64)
        self.rgb_con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch4 = self.norm_batch(64)
        self.rgb_con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch5 = self.norm_batch(64)
        self.rgb_con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_norm_batch6 = self.norm_batch(64)
        self.rgb_con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_con7 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.rgb_fuction_down = nn.LeakyReLU(inplace=True)
        self.rgb_fuction_up = nn.ReLU(inplace=True)

        self.relu = nn.LeakyReLU(inplace=True)
        self.rrelu = nn.ReLU(inplace=True)
        self.M = 11
        number_f = 64
        self.e_conv1 = nn.Conv2d(6, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_convfc = nn.Linear(number_f, 44)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False,
                                    ceil_mode=False)
        self.avagepool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.norm_batch1 = self.norm_batch(64)
        self.con1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch2 = self.norm_batch(64)
        self.con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch3 = self.norm_batch(64)
        self.con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch4 = self.norm_batch(64)
        self.con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch5 = self.norm_batch(64)
        self.con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch6 = self.norm_batch(64)
        self.con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.con7 = nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0)

        self.fuction_down = nn.LeakyReLU(inplace=True)
        self.fuction_up = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.rgb_fuction_down(self.rgb_norm_batch1(self.rgb_con1(x)))
        h = self.rgb_fuction_down(self.rgb_norm_batch2(self.rgb_con2(h)))
        h = self.rgb_fuction_down(self.rgb_norm_batch3(self.rgb_con3(h)))
        h = self.rgb_fuction_down(self.rgb_norm_batch4(self.rgb_con4(h)))
        h = self.rgb_fuction_up(self.rgb_norm_batch5(self.rgb_con5(h)))
        h = self.rgb_fuction_up(self.rgb_norm_batch6(self.rgb_con6(h)))
        rgb_out = torch.sigmoid(self.rgb_con7(h))
        rgb_out = rgb_out[:, 0:3, :, :]
        hsv_fromrgbout = self.rgb2hsv(rgb_out)
        hsv_frominput = self.rgb2hsv(x)

        hsv_input = torch.cat([hsv_fromrgbout, hsv_fromrgbout], dim=1)
        batch_size = hsv_input.size()[0]
        x1 = self.relu(self.e_conv1(hsv_input))
        x1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        x2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        x3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))
        x_r = self.relu(self.e_conv7(x4))
        x_r = self.avagepool(x_r).view(batch_size, -1)
        x_r = self.e_convfc(x_r)
        H, S, V, H2S = torch.split(x_r, self.M, dim=1)
        H_in = hsv_input[:, 0:1, :, :]
        S_in = hsv_input[:, 1:2, :, :]
        V_in = hsv_input[:, 2:3, :, :]
        H_out = piece_function_org(H_in, H, self.M)
        S_out1 = piece_function_org(S_in, S, self.M)
        V_out = piece_function_org(V_in, V, self.M)

        S_out2 = piece_function_org(H_in, H2S, self.M)
        S_out = (S_out1 + S_out2) / 2

        zero_lab = torch.zeros_like(S_out)
        s_t = torch.where(S_out < 0, zero_lab, S_out)
        one_lab = torch.ones_like(S_out)
        S_out = torch.where(s_t > 1, one_lab, s_t)

        zero_lab = torch.zeros_like(V_out)
        s_t = torch.where(V_out < 0, zero_lab, V_out)
        one_lab = torch.ones_like(V_out)
        V_out = torch.where(s_t > 1, one_lab, s_t)

        hsv_out = torch.cat([H_out, S_out, V_out], dim=1)
        hsv_out_rgb = self.hsv2rgb(hsv_out)

        confindencenet_input = torch.cat([x, rgb_out, hsv_out_rgb], dim=1)

        h = self.fuction_down(self.norm_batch1(self.con1(confindencenet_input)))
        h = self.fuction_down(self.norm_batch2(self.con2(h)))
        h = self.fuction_down(self.norm_batch3(self.con3(h)))
        h = self.fuction_down(self.norm_batch4(self.con4(h)))
        h = self.fuction_down(self.norm_batch5(self.con5(h)))
        h = self.fuction_down(self.norm_batch6(self.con6(h)))
        confindence_out = torch.sigmoid(self.con7(h))

        confindence_rgb = confindence_out[:, 0:3, :, :]
        confindence_hsv = confindence_out[:, 3:6, :, :]
        output_useconf = (0.5 * confindence_rgb * rgb_out
                          + 0.5 * confindence_hsv * hsv_out_rgb)
        return output_useconf
