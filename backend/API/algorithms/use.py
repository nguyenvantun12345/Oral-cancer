# Import thư viện cần thiết
# from google.colab import files  # Gỡ bỏ import này vì nó chỉ có trong Google Colab
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import tkinter as tk
from tkinter import filedialog

from project.model.classification.cbam_resnet_ssvae import CBAMResNetSSVAE

# Điều chỉnh đường dẫn lưu mô hình cho phù hợp với máy tính của bạn
save_dir = "D:\College\College\year5s2\datamining\project\Data-Mining---Cancer-Diagnosis\algorithms\testfolder"  # Thay đổi thành đường dẫn trên máy của bạn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def upload_file():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ gốc
    file_path = filedialog.askopenfilename(
        title="Vui lòng tải lên ảnh miệng để phân tích",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not file_path:
        return {}

    file_name = os.path.basename(file_path)
    with open(file_path, 'rb') as f:
        file_content = f.read()

    return {file_name: file_content}


# Load mô hình đã đào tạo
def load_model(model_path):
    model = CBAMResNetSSVAE(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Tạo transformer cho ảnh đầu vào
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Tải và tiền xử lý ảnh
def load_and_process_image():
    print("Vui lòng tải lên ảnh miệng để phân tích:")

    uploaded = upload_file()

    for fn in uploaded.keys():
        print(f'Đã tải lên tệp {fn} {len(uploaded[fn])} bytes')
        img = Image.open(io.BytesIO(uploaded[fn])).convert('RGB')

        # Lưu ảnh gốc để hiển thị
        img_original = np.array(img)

        # Tiền xử lý ảnh
        transform = get_transforms()
        img_tensor = transform(img).unsqueeze(0).to(device)

        return img_tensor, img_original, fn

    return None, None, None


# Tạo heatmap để trực quan hóa vị trí quan trọng
def generate_heatmap(model, img_tensor, original_img):
    # Lấy kích thước ảnh gốc
    height, width = original_img.shape[:2]

    # Lấy feature maps và attention maps từ mô hình
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        # Lấy ra feature sau CBAM layer cuối của ResNet
        feature_extractor = model.feature_extractor

        # Trích xuất feature map từ layer4
        x = feature_extractor.resnet.conv1(img_tensor)
        x = feature_extractor.resnet.bn1(x)
        x = feature_extractor.resnet.relu(x)
        x = feature_extractor.resnet.maxpool(x)

        x = feature_extractor.resnet.layer1(x)
        x = feature_extractor.cbam1(x)

        x = feature_extractor.resnet.layer2(x)
        x = feature_extractor.cbam2(x)

        x = feature_extractor.resnet.layer3(x)
        x = feature_extractor.cbam3(x)

        x = feature_extractor.resnet.layer4(x)
        feature_map = feature_extractor.cbam4.channel_attention(x)
        spatial_attention = feature_extractor.cbam4.spatial_attention(x)

        # Tính trung bình các kênh feature map
        feature_map_avg = torch.mean(feature_map, dim=1, keepdim=True)
        spatial_attention_avg = spatial_attention.squeeze(1)

        # Chuyển về numpy và resize về kích thước ảnh gốc
        heatmap_feature = feature_map_avg.squeeze().cpu().numpy()
        heatmap_spatial = spatial_attention_avg.squeeze().cpu().numpy()

        # Chuẩn hóa heatmap về [0, 1]
        heatmap_spatial = cv2.resize(heatmap_spatial, (width, height))
        heatmap_spatial = np.uint8(255 * heatmap_spatial)

        # Áp dụng colormap
        heatmap_colored = cv2.applyColorMap(heatmap_spatial, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay heatmap lên ảnh gốc
        alpha = 0.4
        overlay = original_img.copy()
        overlay = np.float32(overlay) / 255.0
        heatmap_colored = np.float32(heatmap_colored) / 255.0

        overlayed_img = overlay * alpha + heatmap_colored * (1 - alpha)
        overlayed_img = np.uint8(overlayed_img * 255)

        return overlayed_img, heatmap_spatial


# Dự đoán và visualize kết quả
def predict_and_visualize(image_path=None):
    """
    Thực hiện dự đoán và hiển thị kết quả phân loại ảnh

    Args:
        image_path (str, optional): Đường dẫn đến file ảnh. Nếu không cung cấp,
                                    sẽ hiển thị hộp thoại để chọn ảnh.
    """
    # Tải mô hình
    model_path = f'{save_dir}/ssvae_model.pth'
    model = load_model(model_path)

    # Tải và xử lý ảnh
    if image_path:
        img_tensor, original_img, filename = load_image_from_path(image_path)
    else:
        img_tensor, original_img, filename = load_and_process_image()

    if img_tensor is None:
        print("Không có ảnh nào được tải lên.")
        return

    # Thực hiện dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        class_output = outputs['class_output']
        prob = torch.nn.functional.softmax(class_output, dim=1)

        # Lấy kết quả dự đoán và xác suất
        _, predicted = torch.max(class_output, 1)
        confidence = prob[0][predicted.item()].item() * 100

        class_names = ['Cancer', 'noCancer']
        predicted_class = class_names[predicted.item()]

    # Tạo heatmap
    heatmap_img, heatmap_raw = generate_heatmap(model, img_tensor, original_img)

    # Trực quan hóa kết quả
    plt.figure(figsize=(15, 5))

    # Hiển thị ảnh gốc
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f'Ảnh gốc: {filename}')
    plt.axis('off')

    # Hiển thị heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_img)
    plt.title('Attention Map')
    plt.axis('off')

    # Hiển thị kết quả dự đoán
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f'Kết quả: {predicted_class}\nĐộ tin cậy: {confidence:.2f}%',
             ha='center', va='center', fontsize=15,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Kết quả phân loại')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Hiển thị phân bố xác suất
    plt.figure(figsize=(8, 4))
    probs = prob[0].cpu().numpy() * 100
    plt.bar(class_names, probs)
    plt.title('Phân bố xác suất')
    plt.ylabel('Xác suất (%)')
    plt.ylim(0, 100)

    for i, p in enumerate(probs):
        plt.text(i, p + 1, f'{p:.2f}%', ha='center')

    plt.tight_layout()
    plt.show()

    # Phân tích không gian tiềm ẩn
    latent_vector = outputs['z'].cpu().numpy()

    print(f"\nDự đoán: {predicted_class} với độ tin cậy {confidence:.2f}%")

    # Tải các latent vectors từ tập huấn luyện để so sánh
    try:
        train_latents = np.load(f'{save_dir}/train_latents.npy')
        train_labels = np.load(f'{save_dir}/train_labels.npy')

        # Giảm chiều latent space xuống 2D bằng t-SNE
        from sklearn.manifold import TSNE

        # Kết hợp latent vector mới với dữ liệu train
        combined_latents = np.vstack([train_latents, latent_vector])

        # Áp dụng t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(combined_latents)

        # Tách riêng dữ liệu train và vector mới
        train_latents_2d = latents_2d[:-1]
        new_latent_2d = latents_2d[-1:]

        # Vẽ biểu đồ phân bố
        plt.figure(figsize=(10, 8))

        # Vẽ các điểm từ tập train
        for i, label_name in enumerate(['Cancer', 'noCancer']):
            plt.scatter(train_latents_2d[train_labels == i, 0],
                        train_latents_2d[train_labels == i, 1],
                        label=label_name, alpha=0.5)

        # Vẽ điểm mới với kích thước lớn hơn và màu khác
        plt.scatter(new_latent_2d[0, 0], new_latent_2d[0, 1],
                    color='red', s=100, marker='*', label='Ảnh hiện tại')

        plt.title('Vị trí của ảnh trong không gian tiềm ẩn')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Không thể tải dữ liệu latent space để so sánh: {e}")

    # Trả về kết quả dự đoán để có thể sử dụng trong các hàm khác nếu cần
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'latent_vector': latent_vector
    }


def load_image_from_path(image_path):
    """
    Tải ảnh từ đường dẫn được chỉ định

    Args:
        image_path (str): Đường dẫn đến file ảnh

    Returns:
        tuple: (img_tensor, original_img, filename) chứa tensor ảnh đã xử lý,
               ảnh gốc và tên file
    """
    if not os.path.exists(image_path):
        print(f"Không tìm thấy file ảnh tại: {image_path}")
        return None, None, None

    filename = os.path.basename(image_path)

    # Đọc ảnh
    img = Image.open(image_path).convert('RGB')

    # Lưu ảnh gốc để hiển thị
    img_original = np.array(img)

    # Tiền xử lý ảnh
    transform = get_transforms()
    img_tensor = transform(img).unsqueeze(0).to(device)

    return img_tensor, img_original, filename
# Để chạy chương trình
if __name__ == "__main__":
    img_url = "D:\College\College\year5s2\datamining\cancer_duydata\cancer33\0_1.jpg"
    predict_and_visualize(img_url)