import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from project.model.backbone.cbam_resnet import CBAMResNet
from project.model.classification.cbam_resnet_ssvae import CBAMResNetSSVAE
from project.utils.dataset import OralCancerDataset

# Kiểm tra CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Tạo transforms riêng cho train và test
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo dataset và dataloader
data_dir = "/content/drive/MyDrive/oral_cancer"
save_dir = "/content/drive/MyDrive/model/model_resnet_cbam_ssvae_weight_classification_4"
os.makedirs(save_dir, exist_ok=True)

# Tạo và chia dataset
dataset = OralCancerDataset(root_dir=data_dir, transform=None)

# Chia dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset_indices, test_dataset_indices = torch.utils.data.random_split(
    range(len(dataset)), [train_size, test_size]
)

# Tạo datasets với transforms riêng biệt
class TransformedSubset:
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.indices)

train_dataset = TransformedSubset(dataset, train_dataset_indices.indices, train_transforms)
test_dataset = TransformedSubset(dataset, test_dataset_indices.indices, test_transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Khởi tạo mô hình
cbam_resnet = CBAMResNet(num_classes=2).to(device)
ssvae = CBAMResNetSSVAE(num_classes=2, hidden_dim=256, latent_dim=128).to(device)

# Tối ưu hóa
optimizer = optim.Adam(ssvae.parameters(), lr=0.0001)

# Số epochs
num_epochs = 50

# Danh sách để theo dõi các chỉ số
train_losses = []
train_accuracies = []
val_accuracies = []
recon_losses = []
kl_losses = []
class_losses = []

# Vòng lặp huấn luyện
ssvae.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    recon_running_loss = 0.0
    kl_running_loss = 0.0
    class_running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Đặt gradient về 0
        optimizer.zero_grad()

        # Truyền xuôi
        outputs = ssvae(inputs, labels)

        # Tính loss
        loss_dict = ssvae.loss_function(outputs, inputs, labels)

        loss = loss_dict['loss']
        recon_loss = loss_dict['recon_loss']
        kl_loss = loss_dict['kl_loss']
        class_loss = loss_dict['class_loss']

        # Lan truyền ngược và tối ưu hóa
        loss.backward()
        optimizer.step()

        # Thống kê
        running_loss += loss.item()
        recon_running_loss += recon_loss.item()
        kl_running_loss += kl_loss.item()
        class_running_loss += class_loss.item()

        # Tính độ chính xác
        _, predicted = torch.max(outputs['class_output'], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # In thống kê epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_recon_loss = recon_running_loss / len(train_loader)
    epoch_kl_loss = kl_running_loss / len(train_loader)
    epoch_class_loss = class_running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    # Lưu các chỉ số để vẽ biểu đồ
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    recon_losses.append(epoch_recon_loss)
    kl_losses.append(epoch_kl_loss)
    class_losses.append(epoch_class_loss)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Tổn thất: {epoch_loss:.4f}, Tái tạo: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}, Phân lớp: {epoch_class_loss:.4f}')
    print(f'Độ chính xác: {epoch_acc:.2f}%')

    # Kiểm tra hiệu suất sau mỗi 5 epoch
    if (epoch + 1) % 5 == 0:
        ssvae.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Truyền xuôi
                outputs = ssvae(inputs)

                # Tính độ chính xác
                _, predicted = torch.max(outputs['class_output'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        print(f'Độ chính xác trên tập kiểm định: {val_acc:.2f}%')
        ssvae.train()

# Lưu mô hình
torch.save(ssvae.state_dict(), f'{save_dir}/ssvae_model.pth')
print("Đã hoàn thành huấn luyện và lưu mô hình!")

# Trực quan hóa các chỉ số huấn luyện
plt.figure(figsize=(16, 12))

# Vẽ biểu đồ tổn thất huấn luyện
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Tổn thất tổng')
plt.title('Tổn thất huấn luyện')
plt.xlabel('Epoch')
plt.ylabel('Tổn thất')
plt.grid(True)
plt.legend()

# Vẽ biểu đồ các thành phần tổn thất
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs+1), recon_losses, 'r-', label='Tổn thất tái tạo')
plt.plot(range(1, num_epochs+1), kl_losses, 'g-', label='Tổn thất KL Divergence')
plt.plot(range(1, num_epochs+1), class_losses, 'b-', label='Tổn thất phân lớp')
plt.title('Các thành phần tổn thất')
plt.xlabel('Epoch')
plt.ylabel('Tổn thất')
plt.grid(True)
plt.legend()

# Vẽ biểu đồ độ chính xác huấn luyện
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs+1), train_accuracies, 'g-', label='Độ chính xác huấn luyện')
plt.title('Độ chính xác huấn luyện')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.legend()

# Vẽ biểu đồ độ chính xác kiểm định
plt.subplot(2, 2, 4)
epochs_with_val = list(range(5, num_epochs+1, 5))
plt.plot(epochs_with_val, val_accuracies, 'r-o', label='Độ chính xác kiểm định')
plt.title('Độ chính xác kiểm định')
plt.xlabel('Epoch')
plt.ylabel('Độ chính xác (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f'{save_dir}/training_metrics.png', dpi=300)
plt.show()

# Đánh giá hiệu suất trên tập test
ssvae.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Truyền xuôi
        outputs = ssvae(inputs)
        class_outputs = outputs['class_output']

        # Tính độ chính xác
        probs = F.softmax(class_outputs, dim=1)
        _, predicted = torch.max(class_outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Xác suất cho lớp dương tính

test_acc = 100 * test_correct / test_total
print(f'Độ chính xác trên tập test: {test_acc:.2f}%')

# Chuyển đổi sang mảng numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Ma trận nhầm lẫn
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ung thư', 'Không ung thư'],
            yticklabels=['Ung thư', 'Không ung thư'])
plt.xlabel('Nhãn dự đoán')
plt.ylabel('Nhãn thực tế')
plt.title('Ma trận nhầm lẫn')
plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
plt.show()

# Tính toán các chỉ số
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# In báo cáo phân loại
print("\nBáo cáo phân loại:")
print(classification_report(all_labels, all_preds, target_names=['Ung thư', 'Không ung thư']))

print(f"\nCác chỉ số bổ sung:")
print(f"Độ nhạy (Recall): {sensitivity:.4f}")
print(f"Độ đặc hiệu: {specificity:.4f}")
print(f"Độ chính xác: {precision:.4f}")
print(f"Điểm F1: {f1:.4f}")

# Đường cong ROC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường cong ROC (diện tích = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tỷ lệ dương tính giả (False Positive Rate)')
plt.ylabel('Tỷ lệ dương tính thật (True Positive Rate)')
plt.title('Đường cong đặc trưng hoạt động (ROC)')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{save_dir}/roc_curve.png', dpi=300)
plt.show()

# Đường cong Precision-Recall
precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'Đường cong PR (diện tích = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Độ nhạy (Recall)')
plt.ylabel('Độ chính xác (Precision)')
plt.title('Đường cong Precision-Recall')
plt.legend(loc="lower left")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{save_dir}/precision_recall_curve.png', dpi=300)
plt.show()

# Trích xuất và lưu đặc trưng
def extract_features(model, dataloader):
    model.eval()
    all_features = []
    all_latents = []
    all_labels = []
    all_reconstructions = []
    all_inputs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            # Trích xuất đặc trưng
            outputs = model(inputs)
            features = outputs['features']
            latents = outputs['z']
            reconstructions = outputs.get('reconstruction', None)

            all_features.append(features.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

            if reconstructions is not None:
                all_reconstructions.append(reconstructions.cpu().numpy())
                all_inputs.append(inputs.cpu().numpy())

    results = {
        'features': np.vstack(all_features),
        'latents': np.vstack(all_latents),
        'labels': np.concatenate(all_labels)
    }

    if all_reconstructions:
        results['reconstructions'] = np.vstack(all_reconstructions)
        results['inputs'] = np.vstack(all_inputs)

    return results

# Trích xuất đặc trưng
print("Trích xuất đặc trưng từ mô hình...")
train_data = extract_features(ssvae, train_loader)
test_data = extract_features(ssvae, test_loader)

# Lưu các đặc trưng đã trích xuất
np.save(f'{save_dir}/train_features.npy', train_data['features'])
np.save(f'{save_dir}/train_latents.npy', train_data['latents'])
np.save(f'{save_dir}/train_labels.npy', train_data['labels'])
np.save(f'{save_dir}/test_features.npy', test_data['features'])
np.save(f'{save_dir}/test_latents.npy', test_data['latents'])
np.save(f'{save_dir}/test_labels.npy', test_data['labels'])

# Nếu có reconstruction, lưu và hiển thị một số ví dụ
if 'reconstructions' in test_data:
    np.save(f'{save_dir}/test_reconstructions.npy', test_data['reconstructions'])

    # Hiển thị một số ví dụ về reconstruction
    n_samples = 5
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        # Ảnh gốc
        plt.subplot(2, n_samples, i + 1)
        img = np.transpose(test_data['inputs'][i], (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Giải chuẩn hóa
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title('Ảnh gốc')
        plt.axis('off')

        # Ảnh tái tạo
        plt.subplot(2, n_samples, i + n_samples + 1)
        recon = np.transpose(test_data['reconstructions'][i], (1, 2, 0))
        recon = recon * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Giải chuẩn hóa
        recon = np.clip(recon, 0, 1)
        plt.imshow(recon)
        plt.title('Ảnh tái tạo')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/reconstruction_examples.png', dpi=300)
    plt.show()