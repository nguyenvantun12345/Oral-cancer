from torch import nn

from project.model.attention.chanel_attention import ChannelAttention
from project.model.attention.spatial_attention import SpatialAttention


# 4. Định nghĩa CBAM
class CBAM(nn.Module):
  def __init__(self, in_channels, ratio=16, kernel_size=7):
      super(CBAM, self).__init__()
      self.channel_attention = ChannelAttention(in_channels, ratio)
      self.spatial_attention = SpatialAttention(kernel_size)

  def forward(self, x):
      x = x * self.channel_attention(x) # F' = F * CAM ([C * H * W]  × [C * 1 * 1])
      x = x * self.spatial_attention(x) # f''= F' * SAM ([C × H × W] × [1 * H * W])
      return x