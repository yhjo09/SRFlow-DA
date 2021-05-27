# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 3
        self.affine_eps = 0.0001
        self.model_name = opt_get(opt, ['model'])
        if self.model_name == "SRFlow-DA-D":
            self.n_hidden_layers = 36    
        else:
            self.n_hidden_layers = 4
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'],  0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
            self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                                out_channels=self.channels_for_co * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

            self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                    out_channels=self.in_channels * 2,
                                    hidden_channels=self.hidden_channels,
                                    kernel_hidden=self.kernel_hidden,
                                    n_hidden_layers=self.n_hidden_layers)

        else:
            self.fAffine1, self.fAffine2, self.fAffine3 = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                                out_channels=self.channels_for_co * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

            self.fFeatures1, self.fFeatures2, self.fFeatures3 = self.F(in_channels=self.in_channels_rrdb,
                                    out_channels=self.in_channels * 2,
                                    hidden_channels=self.hidden_channels,
                                    kernel_hidden=self.kernel_hidden,
                                    n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            else:
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures1, self.fFeatures2, self.fFeatures3)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            else:
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine1, self.fAffine2, self.fAffine3)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            else:
                scale, shift = self.feature_extract_aff(z1, ft, self.fAffine1, self.fAffine2, self.fAffine3)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            else:
                scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures1, self.fFeatures2, self.fFeatures3)
            
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f1, f2=None, f3=None):
        h = f1(z)

        if f2 is not None and f3 is not None:
            h_in = h
            for f in f2:
                h2 = f(h_in)
                if self.model_name == "SRFlow-DA-D":
                    h2 = h2 + h_in
                h_in = h2

            if self.model_name == "SRFlow-DA-R":
                h2 = h2 + h

            h = f3(h2)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f1, f2=None, f3=None):
        z = torch.cat([z1, ft], dim=1)
        h = f1(z)

        if f2 is not None and f3 is not None:
            h_in = h
            for f in f2:
                h2 = f(h_in)
                if self.model_name == "SRFlow-DA-D":
                    h2 = h2 + h_in
                h_in = h2
            
            if self.model_name == "SRFlow-DA-R":
                h2 = h2 + h

            h = f3(h2)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        if self.model_name == "SRFlow-DA" or self.model_name == "SRFlow-DA-S":
            layers = [Conv2d(in_channels, hidden_channels, do_actnorm=False), nn.ReLU(inplace=False)]

            for _ in range(n_hidden_layers):
                layers.append(Conv2d(hidden_channels, hidden_channels, do_actnorm=False, kernel_size=[kernel_hidden, kernel_hidden]))
                layers.append(nn.ReLU(inplace=False))

            layers.append(Conv2dZeros(hidden_channels, out_channels))

            return nn.Sequential(*layers)

        else:
            layers1 = [Conv2d(in_channels, hidden_channels, do_actnorm=False), nn.ReLU(inplace=False)]

            layers3 = [Conv2dZeros(hidden_channels, out_channels)]

            if self.model_name == "SRFlow-DA-R":
                layers2 = []
                for _ in range(n_hidden_layers):
                    layers2.append(Conv2d(hidden_channels, hidden_channels, do_actnorm=False, kernel_size=[kernel_hidden, kernel_hidden]))
                    layers2.append(nn.ReLU(inplace=False))

                layers3 = [Conv2dZeros(hidden_channels, out_channels)]

                return nn.Sequential(*layers1), nn.Sequential(*layers2), nn.Sequential(*layers3)

            else:
                layers2 = nn.ModuleList()
                for _ in range(n_hidden_layers//2):
                    layers2.append( nn.Sequential(
                        Conv2d(hidden_channels, hidden_channels, do_actnorm=False, kernel_size=[kernel_hidden, kernel_hidden]),
                        nn.ReLU(inplace=False),
                        Conv2d(hidden_channels, hidden_channels, do_actnorm=False, kernel_size=[kernel_hidden, kernel_hidden]),
                        nn.ReLU(inplace=False),
                    ) )

                return nn.Sequential(*layers1), layers2, nn.Sequential(*layers3)
