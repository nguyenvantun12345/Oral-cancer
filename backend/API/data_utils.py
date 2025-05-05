import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import logging
from scipy import stats
from typing import Dict, Optional, List
from marshmallow import ValidationError

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đường dẫn lưu file
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/patient_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fill_missing_data(patient_data: Dict, reference_excel: Optional[str] = None) -> Dict:
    """
    Điền giá trị thiếu dựa trên phân bố từ file Excel của Airflow.
    
    Args:
        patient_data: Dữ liệu bệnh nhân.
        reference_excel: Đường dẫn file Excel từ Airflow (nếu có).
    
    Returns:
        Dữ liệu bệnh nhân đã được điền.
    """
    try:
        filled_data = patient_data.copy()
        
        # Kiểm tra định dạng birthdate nếu có
        if 'birthdate' in filled_data and filled_data['birthdate']:
            try:
                datetime.strptime(filled_data['birthdate'], '%d/%m/%Y')
            except ValueError:
                raise ValidationError(
                    "birthdate must be in format dd/mm/yyyy",
                    field_name="birthdate"
                )
        
        # Mặc định nếu không có Excel
        defaults = {
            'name': 'Unknown',
            'birthdate': '01/01/2000',
            'gender': 'other',
            'role': 'patient',
            'work': 'unknown',
            'username': f"user_{int(datetime.now().timestamp())}",
            'email': f"user_{int(datetime.now().timestamp())}@example.com"
        }
        
        # Load phân bố từ Excel (nếu có)
        if reference_excel and os.path.exists(reference_excel):
            df = pd.read_excel(reference_excel)
            gender_dist = df['gender'].value_counts(normalize=True).to_dict()
            age_dist = df['birthdate'].dropna().apply(
                lambda x: (datetime.now() - pd.to_datetime(x)).days // 365
            )
            work_dist = df['work'].value_counts(normalize=True).to_dict()
            
            # Điền gender dựa trên phân bố
            if not filled_data.get('gender'):
                filled_data['gender'] = np.random.choice(
                    list(gender_dist.keys()), p=list(gender_dist.values())
                )
            
            # Điền birthdate (tuổi) dựa trên phân bố
            if not filled_data.get('birthdate'):
                mean_age = age_dist.mean()
                std_age = age_dist.std()
                age = int(np.random.normal(mean_age, std_age))
                age = max(0, min(120, age))  # Giới hạn tuổi
                filled_data['birthdate'] = (
                    datetime.now() - pd.DateOffset(years=age)
                ).strftime('%d/%m/%Y')
            
            # Điền work dựa trên phân bố
            if not filled_data.get('work'):
                filled_data['work'] = np.random.choice(
                    list(work_dist.keys()), p=list(work_dist.values())
                )
        
        # Điền các giá trị còn thiếu bằng mặc định
        for key, default in defaults.items():
            if not filled_data.get(key):
                filled_data[key] = default
        
        logger.info(f"Filled missing data for patient: {filled_data.get('username')}")
        return filled_data
    except Exception as e:
        logger.error(f"Error filling missing data: {str(e)}")
        raise

def aggregate_and_visualize(patients: List[Dict], output_prefix: str) -> Dict:
    """
    Tổng hợp dữ liệu bệnh nhân và tạo biểu đồ để phát hiện ngoại lệ, nhiễu.
    
    Args:
        patients: Danh sách bệnh nhân.
        output_prefix: Tiền tố cho file đầu ra.
    
    Returns:
        Báo cáo về ngoại lệ và nhiễu.
    """
    try:
        # Giới hạn số lượng bệnh nhân để tối ưu hiệu suất
        if len(patients) > 1000:
            logger.warning(f"Input size {len(patients)} exceeds 1000, truncating to 1000")
            patients = patients[:1000]
        
        df = pd.DataFrame(patients)
        report = {'outliers': {}, 'noise': {}}
        
        # Kiểm tra và chuyển đổi birthdate
        def parse_birthdate(date_str):
            try:
                return pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
            except ValueError:
                logger.warning(f"Invalid birthdate format: {date_str}")
                return pd.NaT
        
        df['birthdate'] = df['birthdate'].apply(parse_birthdate)
        df['age'] = ((datetime.now() - df['birthdate']).dt.days // 365).fillna(-1)
        
        # Phát hiện nhiễu (dữ liệu không hợp lệ)
        noise_conditions = {
            'age_negative': df['age'] < 0,
            'age_unrealistic': df['age'] > 120,
            'diagnosis_score_invalid': (
                df['diagnosis_score'].notnull() & 
                ((df['diagnosis_score'] < 0) | (df['diagnosis_score'] > 1))
            ) if 'diagnosis_score' in df.columns else pd.Series(False, index=df.index)
        }
        
        for key, condition in noise_conditions.items():
            noise_ids = df[condition]['user_id'].tolist()
            report['noise'][key] = noise_ids
            logger.info(f"Detected {len(noise_ids)} {key} noise cases")
        
        # Phát hiện ngoại lệ (outliers) bằng Z-score
        numeric_cols = ['age']
        if 'diagnosis_score' in df.columns:
            numeric_cols.append('diagnosis_score')
        
        for col in numeric_cols:
            valid_data = df[col][df[col].notnull() & (df[col] >= 0)]
            z_scores = np.abs(stats.zscore(valid_data))
            outlier_indices = valid_data.index[z_scores > 3].tolist()
            report['outliers'][col] = df.loc[outlier_indices, 'user_id'].tolist()
            logger.info(f"Detected {len(report['outliers'][col])} outliers in {col}")
        
        # Tạo biểu đồ
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Phân bố tuổi
        plt.figure(figsize=(10, 6))
        df['age'].dropna().hist(bins=20, color='lightgreen')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        age_plot = os.path.join(OUTPUT_DIR, f"{output_prefix}_age_dist_{timestamp}.png")
        plt.savefig(age_plot)
        plt.close()
        logger.info(f"Saved age distribution plot to {age_plot}")
        
        # 2. Phân bố giới tính
        if 'gender' in df.columns:
            plt.figure(figsize=(8, 6))
            df['gender'].value_counts().plot(kind='bar', color='skyblue')
            plt.title('Gender Distribution')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            gender_plot = os.path.join(OUTPUT_DIR, f"{output_prefix}_gender_dist_{timestamp}.png")
            plt.savefig(gender_plot)
            plt.close()
            logger.info(f"Saved gender distribution plot to {gender_plot}")
        
        # 3. Tương quan tuổi và diagnosis_score (nếu có)
        if 'diagnosis_score' in df.columns:
            corr_df = df[['age', 'diagnosis_score']].dropna()
            if not corr_df.empty:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=corr_df, x='age', y='diagnosis_score', color='coral')
                plt.title('Age vs Diagnosis Score')
                plt.xlabel('Age')
                plt.ylabel('Diagnosis Score')
                corr_plot = os.path.join(OUTPUT_DIR, f"{output_prefix}_corr_{timestamp}.png")
                plt.savefig(corr_plot)
                plt.close()
                logger.info(f"Saved correlation plot to {corr_plot}")
        
        # Lưu báo cáo
        report_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {report_file}")
        
        return report
    except Exception as e:
        logger.error(f"Error in aggregate_and_visualize: {str(e)}")
        raise