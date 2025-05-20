import torch.nn as nn


# 2. Định nghĩa Channel Attention Module
class ChannelAttention(nn.Module):
  def __init__(self, in_channels, ratio=16):
      super(ChannelAttention, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.max_pool = nn.AdaptiveMaxPool2d(1)

      self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
      self.relu = nn.ReLU(inplace=True)
      self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
      # Global Avg pool := ( MLP(FC -> relu -> FC) )
      avg_pool = self.avg_pool(x) #(C * 1 * 1)
      avg_pool_fc1 = self.fc1(avg_pool) #(C/r * 1 * 1)
      avg_pool_fc1_relu = self.relu(avg_pool_fc1) #(C/r * 1 * 1)
      avg_pool_fc1_relu_fc2 = self.fc2(avg_pool_fc1_relu) #(C * 1 * 1)
      avg_out = avg_pool_fc1_relu_fc2 #(C * 1 * 1)

      # Global Max pool := ( MLP(FC -> relu -> FC))
      max_pool = self.max_pool(x) #(C * 1 * 1)
      max_pool_fc1 = self.fc1(max_pool) #(C/r * 1 * 1)
      max_pool_fc1_relu = self.relu(max_pool_fc1) #(C/r * 1 * 1)
      max_pool_fc1_relu_fc2 = self.fc2(max_pool_fc1_relu) #(C * 1 * 1)
      max_out = max_pool_fc1_relu_fc2 #(C * 1 * 1)

      # out := Global Avg pool + Global Max pool
      out = avg_out + max_out #(C * 1 * 1)
      return self.sigmoid(out)