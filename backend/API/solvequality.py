import cv2
import numpy as np

# Enhance contrast
def solve_contrast(image):
    # Tách ảnh thành các kênh màu (Red, Green, Blue)
    channels = cv2.split(image)
    
    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) cho từng kênh
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced_channels = []
    
    for channel in channels:
        enhanced_channel = clahe.apply(channel)  # Áp dụng CLAHE cho mỗi kênh
        enhanced_channels.append(enhanced_channel)
    
    # Gộp các kênh lại để tạo thành ảnh màu
    enhanced_image = cv2.merge(enhanced_channels)
    
    return enhanced_image

# Enhance sharpness
def solve_blur(image):
    # Chuyển ảnh sang RGB nếu là BGR
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa ảnh về giá trị [0,1]
    img_norm = img_rgb.astype(float) / 255.0
    
    # Làm mờ ảnh
    blur = cv2.GaussianBlur(img_norm, (9, 9), 10.0)
    
    # Áp dụng Unsharp Mask
    sharpened = cv2.addWeighted(img_norm, 1.5, blur, -0.5, 0)
    
    # Chuyển về khoảng [0,255] và kiểu uint8
    sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    
    # Chuyển lại về BGR
    result = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)
    
    return result

# Reduce noise
def solve_noise(image):
    # Áp dụng Bilateral Filter để khử nhiễu mà vẫn giữ cạnh
    # Tham số: ảnh, đường kính của mỗi pixel lân cận, sigmaColor, sigmaSpace
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised

# Color balance
def solve_balance(image):
    # Tách các kênh màu
    b, g, r = cv2.split(image)
    
    # Tính toán giá trị trung bình cho mỗi kênh
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    
    # Tính giá trị trung bình của cả 3 kênh
    avg = (r_avg + g_avg + b_avg) / 3
    
    # Tính hệ số điều chỉnh cho mỗi kênh
    r_scale = avg / r_avg if r_avg > 0 else 1
    g_scale = avg / g_avg if g_avg > 0 else 1
    b_scale = avg / b_avg if b_avg > 0 else 1
    
    # Điều chỉnh các kênh
    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    
    # Gộp các kênh lại
    balanced_image = cv2.merge([b, g, r])
    
    return balanced_image

# Resize image
def resize_image(image, target_width=256, target_height=256):
    # Lấy kích thước ban đầu
    height, width = image.shape[:2]
    
    # Kiểm tra nếu đã đủ tiêu chuẩn về kích thước
    if width >= target_width and height >= target_height:
        return image
    
    # Tính tỷ lệ để giữ nguyên tỷ lệ khung hình
    scale = max(target_width / width, target_height / height)
    
    # Tính kích thước mới
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize ảnh
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return resized_image

# Brightness adjustment
def solve_brightness(image, target_brightness=127):
    # Chuyển sang grayscale để tính độ sáng trung bình
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    
    # Tính giá trị cần thêm vào để đạt đến độ sáng mong muốn
    brightness_diff = target_brightness - current_brightness
    
    # Điều chỉnh độ sáng bằng cách thêm/bớt giá trị cho mỗi pixel
    adjusted = np.clip(image.astype(np.float32) + brightness_diff, 0, 255).astype(np.uint8)
    
    return adjusted

# Hàm tổng hợp xử lý tất cả các vấn đề
def enhance_image(image):
    # Kiểm tra và điều chỉnh kích thước
    image = resize_image(image)
    
    # Điều chỉnh độ sáng
    image = solve_brightness(image)
    
    # Giảm nhiễu
    image = solve_noise(image)
    
    # Tăng độ sắc nét
    image = solve_blur(image)
    
    # Cải thiện độ tương phản
    image = solve_contrast(image)
    
    # Cân bằng màu sắc
    image = solve_balance(image)
    
    return image

# Hàm xử lý ảnh và lưu kết quả
def process_and_save(input_path, output_path):
    # Đọc ảnh
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Không thể đọc ảnh: {input_path}")
        return False
    
    # Tăng cường chất lượng ảnh
    enhanced = enhance_image(image)
    
    # Lưu ảnh đã xử lý
    cv2.imwrite(output_path, enhanced)
    print(f"Đã lưu ảnh đã xử lý tại: {output_path}")
    
    return True

