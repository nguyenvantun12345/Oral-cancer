import numpy as np
import pandas as pd
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor

"""
Tiêu chuẩn chất lượng ảnh:
Độ phân giải: > 150x150 px
Độ sắc nét (Laplacian var): >= 50
Nhiễu: <= 30
Tương phản (độ lệch chuẩn): >= 15
Độ sáng: > 20 và <= 235
Chênh lệch màu: <= 60
"""
def load_image(image_input):
    """Tải ảnh từ URL hoặc đường dẫn cục bộ."""
    try:
        if image_input.startswith(('http://', 'https://')):
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Failed to load image: {image_input}")
        return img
    except Exception as e:
        print(f"Error loading image {image_input}: {str(e)}")
        return None

def cal_brightness_contrast(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(img)
    contrast = np.std(img)
    return {"contrast": contrast, "brightness": brightness}

def cal_resolution(image):
    height, width = image.shape[:2]
    return {"width": width, "height": height}

def cal_blurry(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return {"sharpness": var}

def cal_noise(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    noise = laplacian.std()
    return {"noise": noise}

def cal_color_balance(image):
    (b, g, r) = cv2.split(image)
    r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
    color_diff = max(abs(r_std - g_std), abs(g_std - b_std), abs(b_std - r_std))
    return {"color_diff": color_diff}

def evaluate_image_parallel(image):
    try:
        functions = [
            cal_brightness_contrast,
            cal_resolution,
            cal_blurry,
            cal_noise,
            cal_color_balance
        ]
        
        with ThreadPoolExecutor(max_workers=len(functions)) as executor:
            futures = [executor.submit(func, image) for func in functions]
            results = {}
            for future in futures:
                results.update(future.result())
        
        metrics = np.array([
            results["brightness"],
            results["contrast"],
            results["width"],
            results["height"],
            results["sharpness"],
            results["noise"],
            results["color_diff"]
        ])
        
        deviations = np.array([
            0 if 20 < results["brightness"] <= 235 else (20 - results["brightness"] if results["brightness"] <= 20 else results["brightness"] - 235),
            0 if results["contrast"] >= 15 else 15 - results["contrast"],
            0 if results["width"] > 150 else 150 - results["width"],
            0 if results["height"] > 150 else 150 - results["height"],
            0 if results["sharpness"] >= 50 else 50 - results["sharpness"],
            0 if results["noise"] <= 30 else results["noise"] - 30,
            0 if results["color_diff"] <= 60 else results["color_diff"] - 60
        ])
        
        result = np.vstack([metrics, deviations])
        return result
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")
        return None

def process_multiple_images(image_paths):
    results = {}
    
    # Tải tất cả các ảnh
    images = {path: load_image(path) for path in image_paths}
    valid_images = {path: img for path, img in images.items() if img is not None}
    
    if not valid_images:
        print("Không có ảnh nào được đọc thành công")
        return results
    
    # Xử lý song song cho mỗi ảnh
    with ThreadPoolExecutor(max_workers=min(len(valid_images), 8)) as executor:
        futures = {executor.submit(evaluate_image_parallel, img): path for path, img in valid_images.items()}
        
        for future in futures:
            path = futures[future]
            try:
                result = future.result()
                if result is not None:
                    column_names = ['Brightness', 'Contrast', 'Width', 'Height', 'Sharpness', 'Noise', 'Color_Diff']
                    row_names = ['Measured', 'Deviation']
                    results[path] = pd.DataFrame(result, index=row_names, columns=column_names)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {path}: {e}")
    
    return results

def print_detailed_assessment(df):
    criteria = {
        'Brightness': 'đạt' if df.loc['Deviation', 'Brightness'] == 0 else 'không đạt',
        'Contrast': 'đạt' if df.loc['Deviation', 'Contrast'] == 0 else 'không đạt',
        'Resolution': 'đạt' if df.loc['Deviation', 'Width'] == 0 and df.loc['Deviation', 'Height'] == 0 else 'không đạt',
        'Sharpness': 'đạt' if df.loc['Deviation', 'Sharpness'] == 0 else 'không đạt',
        'Noise': 'đạt' if df.loc['Deviation', 'Noise'] == 0 else 'không đạt',
        'Color Balance': 'đạt' if df.loc['Deviation', 'Color_Diff'] == 0 else 'không đạt'
    }
    
    print("\nChi tiết đánh giá:")
    for criterion, status in criteria.items():
        print(f"- {criterion}: {status}")

if __name__ == "__main__":
    image_paths = ["enhanced_dog1.webp", "dog2.webp", "dog1.webp"]
    results = process_multiple_images(image_paths)
    for path, df in results.items():
        print(f"\nKết quả đánh giá cho ảnh: {path}")
        print(df)