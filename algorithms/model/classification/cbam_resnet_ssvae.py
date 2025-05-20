import math

import torch
from torch import nn

from project.model.backbone.cbam_resnet import CBAMResNet
from project.model.vae.decode import Decoder
from project.model.vae.encode import Encoder
import torch.nn.functional as F

class CBAMResNetSSVAE(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, hidden_dim=256, latent_dim=128, pretrained=True):
        super(CBAMResNetSSVAE, self).__init__()

        # Trích xuất đặc trưng với khả năng phân lớp
        self.feature_extractor = CBAMResNet(num_classes=num_classes, pretrained=pretrained)

        # Các thành phần VAE
        self.encoder = Encoder(input_dim=512, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=512)

        # Lưu trữ cấu hình
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, y=None):
        # Trích xuất đặc trưng và phân lớp sử dụng CBAM ResNet
        features, class_output = self.feature_extractor(x)

        # Encoder: Ánh xạ đặc trưng thành tham số phân phối tiềm ẩn
        mu, log_var = self.encoder(features)

        # Tái tham số hóa
        z = self.reparameterize(mu, log_var)

        # Decoder: Tái tạo đặc trưng từ không gian tiềm ẩn
        recon_features = self.decoder(z)

        return {
            'reconstructed_features': recon_features,
            'features': features,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'class_output': class_output
        }

    def loss_function(self, outputs, x, y=None, recon_weight=1.5, kl_weight=0.5, class_weight=2.0,
                      current_epoch=0, annealing_epochs=10, annealing_start=0.0, class_schedule=True):
        """
        Tính toán hàm mất mát với KL Annealing và tăng cường tập trung vào phân lớp

        Tham số:
            outputs: Dict chứa đầu ra của mô hình
            x: Đặc trưng đầu vào
            y: Nhãn mục tiêu (tùy chọn)
            recon_weight: Trọng số cho mất mát tái tạo
            kl_weight: Trọng số tối đa cho thành phần KL divergence
            class_weight: Trọng số cho mất mát phân lớp
            current_epoch: Epoch huấn luyện hiện tại (cho annealing)
            annealing_epochs: Số lượng epochs để điều chỉnh KL
            annealing_start: Giá trị bắt đầu cho trọng số KL (mặc định 0.0)
            class_schedule: Có sử dụng lịch trình cho trọng số phân lớp hay không
        """
        recon_features = outputs['reconstructed_features']
        features = outputs['features']
        mu = outputs['mu']
        log_var = outputs['log_var']
        class_output = outputs['class_output']

        # Mất mát tái tạo (MSE giữa đặc trưng gốc và đặc trưng tái tạo)
        recon_loss = F.mse_loss(recon_features, features, reduction='sum')

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Áp dụng KL Annealing - lịch trình nhẹ nhàng hơn sử dụng hàm sigmoid
        if current_epoch < annealing_epochs:
            x = 10 * (current_epoch / annealing_epochs - 0.5)
            sigmoid = 1 / (1 + math.exp(-x))
            # Sử dụng sigmoid để tăng nhẹ nhàng hơn
            annealing_factor = annealing_start + (kl_weight - annealing_start) * sigmoid
        else:
            annealing_factor = kl_weight

        # Mất mát phân lớp (nếu nhãn được cung cấp)
        class_loss = torch.tensor(0.0, device=features.device)
        if y is not None:
            # Sử dụng cross entropy có trọng số dựa trên tần suất lớp nếu cần
            class_loss = F.cross_entropy(class_output, y, reduction='sum')

            # Tùy chọn tăng trọng số phân lớp theo thời gian
            effective_class_weight = class_weight
            if class_schedule and current_epoch < annealing_epochs:
                # Tăng trọng số lớp khi quá trình huấn luyện tiến triển
                effective_class_weight = class_weight * (1.0 + 0.6 * (current_epoch / annealing_epochs))

        # Tổng mất mát - sử dụng trọng số KL đã điều chỉnh
        total_loss = recon_weight * recon_loss + annealing_factor * kl_loss
        if y is not None:
            total_loss += effective_class_weight * class_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'class_loss': class_loss,
            'kl_weight_used': annealing_factor,
            'class_weight_used': effective_class_weight if y is not None else 0.0
        }

    def generate(self, num_samples=1):
        # Lấy mẫu từ prior
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)

        # Giải mã các vector tiềm ẩn
        generated_features = self.decoder(z)
        return generated_features

    def encode(self, x):
        features, _ = self.feature_extractor(x)
        mu, log_var = self.encoder(features)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)