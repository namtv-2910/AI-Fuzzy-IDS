import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def generate_sample_data(n_samples=500, random_state=42):
    """
    Tạo dữ liệu mẫu giả lập traffic mạng để demo.
    Các features:
        - packet_rate    : số gói tin/giây (0 - 1000)
        - failed_logins  : số lần đăng nhập thất bại (0 - 100)
        - port_scan_count: số port bị quét (0 - 1000)
        - traffic_volume : lượng dữ liệu MB (0 - 1000)
        - label          : 0 = bình thường, 1 = tấn công
    """
    np.random.seed(random_state)

    # --- Traffic bình thường ---
    n_normal = int(n_samples * 0.6)
    normal = pd.DataFrame({
        'packet_rate':     np.random.uniform(0, 200, n_normal),
        'failed_logins':   np.random.uniform(0, 10,  n_normal),
        'port_scan_count': np.random.uniform(0, 100, n_normal),
        'traffic_volume':  np.random.uniform(0, 300, n_normal),
        'label': 0
    })

    # --- Traffic tấn công ---
    n_attack = n_samples - n_normal
    attack = pd.DataFrame({
        'packet_rate':     np.random.uniform(400, 1000, n_attack),
        'failed_logins':   np.random.uniform(30,  100,  n_attack),
        'port_scan_count': np.random.uniform(300, 1000, n_attack),
        'traffic_volume':  np.random.uniform(500, 1000, n_attack),
        'label': 1
    })

    df = pd.concat([normal, attack], ignore_index=True).sample(frac=1, random_state=random_state)
    return df.reset_index(drop=True)


def load_or_generate_data(filepath='data/sample_data.csv'):
    """Load dữ liệu từ file CSV, nếu chưa có thì tạo mới."""
    os.makedirs('data', exist_ok=True)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    df = generate_sample_data()
    df.to_csv(filepath, index=False)
    print(f"✅ Đã tạo và lưu dữ liệu tại: {filepath}")
    return df


def preprocess(df):
    """
    Tiền xử lý dữ liệu:
    - Loại bỏ NaN
    - Clamp giá trị về đúng range
    - Chuẩn hoá [0, 1] cho ML models
    """
    df = df.dropna().copy()

    # Giới hạn giá trị hợp lệ
    df['packet_rate']     = df['packet_rate'].clip(0, 1000)
    df['failed_logins']   = df['failed_logins'].clip(0, 100)
    df['port_scan_count'] = df['port_scan_count'].clip(0, 1000)
    df['traffic_volume']  = df['traffic_volume'].clip(0, 1000)

    features = ['packet_rate', 'failed_logins', 'port_scan_count', 'traffic_volume']
    X = df[features].copy()
    y = df['label'].copy()

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    return X, X_scaled, y, scaler
