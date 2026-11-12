# 🛡️ AI-Fuzzy-IDS

> Hệ thống Phát hiện Bất thường Mạng sử dụng Fuzzy Logic (Logic Mờ)
> 
> **Tiểu luận Trí Tuệ Nhân Tạo Nâng Cao** | Python + scikit-fuzzy + Streamlit

---

## 📋 Giới thiệu

Dự án xây dựng hệ thống **Phát hiện Xâm nhập Mạng (IDS)** dựa trên **Fuzzy Logic (Mamdani FIS)**, kết hợp so sánh với các mô hình Machine Learning truyền thống.

### Điểm nổi bật
- ✅ **Explainable AI** — giải thích được lý do cảnh báo (không phải hộp đen)
- ✅ **Không cần dữ liệu huấn luyện** — hoạt động dựa trên luật chuyên gia
- ✅ **So sánh** với Random Forest, SVM, KNN, Decision Tree
- ✅ **Web App** trực quan bằng Streamlit

---

## 🗂️ Cấu trúc project

```
AI-Fuzzy-IDS/
├── app.py                  # Web App Streamlit
├── requirements.txt        # Thư viện cần thiết
├── src/
│   ├── fuzzy_ids.py        # Hệ thống Fuzzy Logic chính (FIS)
│   ├── ml_comparison.py    # So sánh với các mô hình ML
│   └── preprocessing.py    # Tiền xử lý dữ liệu
├── data/
│   └── sample_data.csv     # Dữ liệu traffic mạng mẫu
└── models/
    └── best_model.pkl      # Model ML tốt nhất (sau khi train)
```

---

## ⚙️ Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/AI-Fuzzy-IDS.git
cd AI-Fuzzy-IDS

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy Web App
streamlit run app.py
```

---

## 🧠 Kiến trúc Fuzzy IDS

### Biến đầu vào (Input Variables)

| Biến | Mô tả | Range | Giá trị ngôn ngữ |
|------|-------|-------|-----------------|
| `packet_rate` | Tốc độ gói tin (gói/giây) | [0, 1000] | Thấp / Trung bình / Cao |
| `failed_logins` | Số lần login thất bại | [0, 100] | Ít / Vừa / Nhiều |
| `port_scan_count` | Số port bị quét | [0, 1000] | Ít / Vừa / Nhiều |
| `traffic_volume` | Lưu lượng dữ liệu (MB) | [0, 1000] | Nhỏ / Vừa / Lớn |

### Biến đầu ra (Output Variable)

| Biến | Mô tả | Range | Mức cảnh báo |
|------|-------|-------|-------------|
| `threat_level` | Mức độ nguy hiểm | [0, 100] | Thấp / Vừa / Cao / Nguy hiểm |

### Ví dụ luật Fuzzy IF-THEN

```
IF packet_rate = CAO   AND port_scan = NHIỀU  → threat_level = NGUY_HIỂM
IF failed_logins = NHIỀU AND packet_rate = CAO → threat_level = CAO
IF packet_rate = THẤP  AND failed_logins = ÍT  → threat_level = THẤP
```

---

## 🔄 Quy trình FIS (Mamdani)

```
Giá trị số → [Fuzzification] → [Rule Evaluation] → [Aggregation] → [Defuzzification] → Giá trị số
   350            μ(CAO)=0.9      Rule matching         Merge          Centroid            78.5
```

---

## 📊 Kết quả

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Fuzzy Logic** | ~0.88 | ~0.86 | ~0.91 | ~0.88 |
| Random Forest | ~0.95 | ~0.94 | ~0.96 | ~0.95 |
| SVM | ~0.93 | ~0.92 | ~0.94 | ~0.93 |
| KNN | ~0.91 | ~0.90 | ~0.92 | ~0.91 |
| Decision Tree | ~0.90 | ~0.89 | ~0.91 | ~0.90 |

> **Nhận xét:** Fuzzy Logic tuy có accuracy thấp hơn ML, nhưng **không cần dữ liệu**, **giải thích được**, và phù hợp với môi trường thiếu dữ liệu chất lượng.

---

## 🛠️ Công nghệ sử dụng

- **Python 3.10+**
- **scikit-fuzzy** — xây dựng Fuzzy Inference System
- **scikit-learn** — các mô hình ML để so sánh
- **Streamlit** — Web App
- **Plotly / Matplotlib** — trực quan hoá
- **Pandas / NumPy** — xử lý dữ liệu

---

## 👨‍💻 Tác giả

Sinh viên: *[Tên của bạn]*  
Môn học: Trí Tuệ Nhân Tạo Nâng Cao  
Trường: *[Tên trường]*  
Năm: 2026
