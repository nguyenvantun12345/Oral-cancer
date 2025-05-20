
import torch.nn as nn
import torch

# 3. Định nghĩa Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kích thước kernel phải là 3 hoặc 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # đầu vào là đầu ra của chanel attention := (chanel_out)
        # chanel_out := signmoid(chanel_out)
        # avg out = mean
        avg_out = torch.mean(x, dim=1, keepdim=True) # (1 * H * W)
        # max out = max
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (1 * H * W)
        # hợp nhất 2 cái avg với max
        x = torch.cat([avg_out, max_out], dim=1) # (2 * H * W)
        # nhân lớp tích chập 7*7 (2 * H * W)
        x = self.conv(x)
        # đi X qua hàm signmoid (1 * H * W)
        return self.sigmoid(x)