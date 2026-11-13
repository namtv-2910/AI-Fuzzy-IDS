"""
app.py — Streamlit Web App
==========================
Giao diện web cho hệ thống Fuzzy IDS.
Chạy bằng: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from fuzzy_ids import FuzzyIDS
from preprocessing import load_or_generate_data, preprocess
from ml_comparison import train_and_evaluate, evaluate_fuzzy, plot_comparison

# ─────────────────────────── Cấu hình trang ───────────────────────────
st.set_page_config(
    page_title='Fuzzy IDS — Phát hiện bất thường mạng',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────── CSS tuỳ chỉnh ────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:700; color:#2c3e50; }
    .sub-title    { font-size:1.1rem; color:#7f8c8d; margin-bottom:1.5rem; }
    .metric-card  { background:#f8f9fa; border-radius:12px; padding:1rem;
                    border-left:4px solid #3498db; margin-bottom:0.5rem; }
    .alert-low      { background:#d5f5e3; border-left:4px solid #2ecc71;
                      padding:1rem; border-radius:8px; color: #1e8449; }
    .alert-medium   { background:#fef9e7; border-left:4px solid #f1c40f;
                      padding:1rem; border-radius:8px; color: #b7950b; }
    .alert-high     { background:#fdebd0; border-left:4px solid #e67e22;
                      padding:1rem; border-radius:8px; color: #b9770e; }
    .alert-critical { background:#fadbd8; border-left:4px solid #e74c3c;
                      padding:1rem; border-radius:8px; color: #922b21; }
    .pipeline-step  { background: #ffffff; border: 1px solid #e1e8ed; border-radius: 8px; padding: 12px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Cache nặng ───────────────────────────────
@st.cache_resource(show_spinner='Khởi tạo Fuzzy IDS...')
def load_fuzzy():
    return FuzzyIDS()

@st.cache_data(show_spinner='Tải dữ liệu...')
def load_data():
    return load_or_generate_data()

@st.cache_data(show_spinner='Huấn luyện mô hình...')
def run_ml(_df):
    X_raw, X_scaled, y, scaler = preprocess(_df)
    results, trained, X_test, y_test = train_and_evaluate(X_scaled, y)
    return results, trained, X_raw, X_scaled, y, scaler, X_test, y_test


# ─────────────────────────── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fuzzy IDS")
    st.markdown('*Phát hiện bất thường mạng bằng Fuzzy Logic*')
    st.divider()
    page = st.radio('📋 Điều hướng', [
        '🏠 Trang chủ',
        '🔍 Phân tích đơn',
        '📊 So sánh mô hình',
        '📈 Phân tích dữ liệu',
        '🧠 Lý thuyết Fuzzy',
    ])
    st.divider()
    st.caption('Tiểu luận AI Nâng Cao | Python + scikit-fuzzy')


# ═══════════════════════════════════════════════════════════════════════
# TRANG 1 — Trang chủ
# ═══════════════════════════════════════════════════════════════════════
if page == '🏠 Trang chủ':
    st.markdown('<div class="main-title">🛡️ Hệ thống Phát hiện Bất thường Mạng</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sử dụng Fuzzy Logic (Logic Mờ) — Tiểu luận AI Nâng Cao</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('📡 Biến đầu vào', '4')
    col2.metric('📋 Luật Fuzzy', '15')
    col3.metric('⚡ Phương pháp', 'Mamdani FIS')
    col4.metric('🎯 Mức cảnh báo', '4 cấp')

    st.divider()
    st.markdown('### 🔄 Quy trình hoạt động (Mamdani Inference Pipeline)')
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class="pipeline-step">
            <h4>1. Fuzzification</h4>
            <p>Chuyển đổi dữ liệu số đo (packet, login, port, vol) thành độ thuộc mờ (Low, Med, High).</p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="pipeline-step">
            <h4>2. Rule Evaluation</h4>
            <p>Khớp 15 luật IF-THEN chuyên gia với các toán tử mờ AND (min), OR (max).</p>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="pipeline-step">
            <h4>3. Aggregation</h4>
            <p>Tổng hợp các tập mờ đầu ra từ tất cả luật được kích hoạt.</p>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="pipeline-step">
            <h4>4. Defuzzification</h4>
            <p>Giải mờ theo phương pháp Trọng tâm (Centroid) cho ra điểm nguy hiểm (0-100).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('### 🗂️ Tính năng chính')
    c1, c2 = st.columns(2)
    with c1:
        st.success('✅ Phân tích traffic mạng theo thời gian thực')
        st.success('✅ Giải thích lý do cảnh báo rõ ràng (Explainable AI)')
        st.success('✅ So sánh đa chiều với 4 thuật toán Machine Learning')
    with c2:
        st.info('📊 Trực quan hoá các hàm thuộc (Membership Functions)')
        st.info('📈 Phân tích dữ liệu luồng mạng trực quan qua biểu đồ tương tác')
        st.info('🧠 Tích hợp tài liệu lý thuyết Logic Mờ học thuật')


# ═══════════════════════════════════════════════════════════════════════
# TRANG 2 — Phân tích đơn
# ═══════════════════════════════════════════════════════════════════════
elif page == '🔍 Phân tích đơn':
    st.markdown('## 🔍 Phân tích Traffic Mạng')
    st.markdown('Nhập thông số traffic mạng hoặc chọn kịch bản mẫu để hệ thống đánh giá mức độ nguy hiểm.')
    st.divider()

    fuzzy = load_fuzzy()

    # Khởi tạo session_state cho các slider nếu chưa có
    if 'packet_rate' not in st.session_state:
        st.session_state.packet_rate = 100
    if 'failed_logins' not in st.session_state:
        st.session_state.failed_logins = 5
    if 'port_scan_count' not in st.session_state:
        st.session_state.port_scan_count = 50
    if 'traffic_volume' not in st.session_state:
        st.session_state.traffic_volume = 150

    # Callback cập nhật preset
    def set_preset(pr, fl, ps, tv):
        st.session_state.packet_rate = pr
        st.session_state.failed_logins = fl
        st.session_state.port_scan_count = ps
        st.session_state.traffic_volume = tv

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('### 🎯 Kịch bản mẫu (Presets)')
        c1b, c2b, c3b = st.columns(3)
        c1b.button('✅ Bình thường', on_click=set_preset, args=(50, 2, 20, 100))
        c2b.button('⚠️ Nghi ngờ', on_click=set_preset, args=(400, 30, 400, 500))
        c3b.button('🚨 Tấn công', on_click=set_preset, args=(900, 80, 900, 950))

        st.markdown('### ⚙️ Điều chỉnh thông số đầu vào')

        pr = st.slider('📡 Tốc độ gói tin (packet/giây)', 0, 1000, key='packet_rate',
                       help='Số lượng gói tin mạng mỗi giây')
        fl = st.slider('🔑 Số lần đăng nhập thất bại', 0, 100, key='failed_logins',
                       help='Số lần login thất bại trong 1 phút')
        ps = st.slider('🔍 Số port bị quét', 0, 1000, key='port_scan_count',
                       help='Số lượng port bị scan')
        tv = st.slider('💾 Lưu lượng dữ liệu (MB)', 0, 1000, key='traffic_volume',
                       help='Tổng lưu lượng dữ liệu truyền qua')

    with col2:
        st.markdown('### 📊 Kết quả đánh giá mức độ rủi ro')
        result = fuzzy.predict(pr, fl, ps, tv)
        score  = result['score']

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=score,
            title={'text': 'Điểm Nguy Hiểm (0 - 100)', 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': result['color']},
                'steps': [
                    {'range': [0,  25], 'color': '#d5f5e3'},
                    {'range': [25, 50], 'color': '#fef9e7'},
                    {'range': [50, 75], 'color': '#fdebd0'},
                    {'range': [75, 100], 'color': '#fadbd8'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': score}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig)

        # Alert box
        level_map = {
            'low':      ('alert-low',      '🟢 BÌNH THƯỜNG',    'Traffic mạng hoạt động trong ngưỡng an toàn. Không phát hiện dấu hiệu bất thường.'),
            'medium':   ('alert-medium',   '🟡 CẢNH BÁO THẤP', 'Phát hiện một số dấu hiệu đáng ngờ. Khuyến nghị theo dõi thêm luồng gói tin.'),
            'high':     ('alert-high',     '🟠 CẢNH BÁO CAO',  'Dấu hiệu xâm nhập/tấn công rõ rệt! Cần rà soát ngay IP nguồn và tường lửa.'),
            'critical': ('alert-critical', '🔴 NGUY HIỂM!',     'Hệ thống đang bị tấn công dồn dập (DDoS, Brute force)! Kích hoạt cơ chế phòng vệ ngay.'),
        }
        css_cls, title, msg = level_map[result['level']]
        st.markdown(f'<div class="{css_cls}"><strong>{title}</strong><br>{msg}</div>',
                    unsafe_allow_html=True)

        st.divider()
        st.markdown('**📋 Giải thích suy luận (Fuzzy Membership State):**')
        reasons = {
            f'Tốc độ gói ({pr} p/s)': ('Thấp' if pr < 350 else 'Trung bình' if pr < 650 else 'Cao'),
            f'Login thất bại ({fl})': ('Ít' if fl < 25 else 'Vừa' if fl < 60 else 'Nhiều'),
            f'Port Scan ({ps})': ('Ít' if ps < 300 else 'Vừa' if ps < 650 else 'Nhiều'),
            f'Lưu lượng ({tv} MB)': ('Nhỏ' if tv < 300 else 'Vừa' if tv < 650 else 'Lớn'),
        }
        for feat, val in reasons.items():
            st.markdown(f'- **{feat}** → `{val}`')


# ═══════════════════════════════════════════════════════════════════════
# TRANG 3 — So sánh mô hình
# ═══════════════════════════════════════════════════════════════════════
elif page == '📊 So sánh mô hình':
    st.markdown('## 📊 So sánh Fuzzy Logic vs Machine Learning')

    fuzzy = load_fuzzy()
    df    = load_data()

    with st.spinner('Đang huấn luyện & đánh giá các mô hình...'):
        results, trained, X_raw, X_scaled, y, scaler, X_test, y_test = run_ml(df)
        fuzzy_result, fuzzy_pred = evaluate_fuzzy(fuzzy, X_raw, y)
        all_results = pd.concat([results, pd.DataFrame([fuzzy_result])], ignore_index=True)

    st.markdown('### 📋 Bảng chỉ số đánh giá tổng hợp')
    styled = all_results.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
        'Recall': '{:.4f}', 'F1-Score': '{:.4f}',
        'CV Mean': '{:.4f}', 'CV Std': '{:.4f}',
    }).highlight_max(subset=['Accuracy', 'F1-Score'], color='#d5f5e3') \
      .highlight_min(subset=['Accuracy', 'F1-Score'], color='#fadbd8')
    st.dataframe(styled)

    st.markdown('### 📊 Biểu đồ so sánh')
    fig = plot_comparison(all_results)
    st.pyplot(fig)

    st.markdown('### 💡 Nhận xét học thuật')
    best = all_results.loc[all_results['F1-Score'].idxmax(), 'Model']
    st.info(f"""
    - **Mô hình có F1-Score cao nhất:** **{best}**
    - **Fuzzy Logic:** Hoạt động dựa trên tri thức chuyên gia định nghĩa sẵn, không phụ thuộc vào nhãn huấn luyện, cho phép giải thích rõ ràng tại sao traffic bị đánh dấu rủi ro.
    - **Machine Learning thuần (RF, SVM, KNN, DT):** Đạt độ chính xác định lượng cao trên tập dữ liệu chuẩn, nhưng mang tính chất "hộp đen" khó diễn giải tường minh.
    """)


# ═══════════════════════════════════════════════════════════════════════
# TRANG 4 — Phân tích dữ liệu
# ═══════════════════════════════════════════════════════════════════════
elif page == '📈 Phân tích dữ liệu':
    st.markdown('## 📈 Phân tích Dữ liệu Traffic Mạng (EDA)')

    df = load_data()
    st.markdown(f'**Tổng số mẫu:** {len(df)} | **Bình thường (0):** {(df["label"] == 0).sum()} | **Tấn công (1):** {(df["label"] == 1).sum()}')
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='packet_rate', color='label',
                           title='Phân phối tốc độ gói tin',
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                           labels={'label': 'Nhãn', 'packet_rate': 'Packet Rate'},
                           barmode='overlay', opacity=0.7)
        st.plotly_chart(fig)

    with col2:
        fig2 = px.scatter(df, x='packet_rate', y='failed_logins',
                          color=df['label'].map({0: 'Bình thường', 1: 'Tấn công'}),
                          title='Packet Rate vs Failed Logins',
                          color_discrete_map={'Bình thường': '#2ecc71', 'Tấn công': '#e74c3c'},
                          opacity=0.6)
        st.plotly_chart(fig2)

    fig3 = px.scatter_matrix(df, dimensions=['packet_rate', 'failed_logins', 'port_scan_count', 'traffic_volume'],
                              color=df['label'].map({0: 'Bình thường', 1: 'Tấn công'}),
                              color_discrete_map={'Bình thường': '#2ecc71', 'Tấn công': '#e74c3c'},
                              title='Ma trận phân tán giữa các đặc trưng mạng', opacity=0.5)
    st.plotly_chart(fig3)


# ═══════════════════════════════════════════════════════════════════════
# TRANG 5 — Lý thuyết Fuzzy
# ═══════════════════════════════════════════════════════════════════════
elif page == '🧠 Lý thuyết Fuzzy':
    st.markdown('## 🧠 Cơ sở Lý thuyết Logic Mờ trong An Toàn Mạng')

    fuzzy = load_fuzzy()

    st.markdown("""
    ### Khái niệm cơ bản
    Khác với Logic cổ điển (Boolean Logic) chỉ nhận hai giá trị 0 hoặc 1, **Logic Mờ (Fuzzy Logic)** mở rộng miền chân lý thành đoạn liên tục **[0, 1]**. 
    Điều này đặc biệt phù hợp trong môi trường an ninh mạng, nơi ranh giới giữa một hành vi "truy cập đột biến hợp lệ" và "tấn công từ chối dịch vụ (DoS)" thường không tuyệt đối.
    """)

    st.markdown('### 📊 Các hàm thuộc (Membership Functions)')
    fig = fuzzy.plot_membership_functions()
    st.pyplot(fig)

    st.markdown('### 📋 Bảng luật mờ Mamdani (Rule Base)')
    rules_data = [
        ('packet_rate=CAO',   'port_scan=NHIỀU',  '—',              '→ NGUY HIỂM'),
        ('failed_logins=NHIỀU','port_scan=NHIỀU',  '—',              '→ NGUY HIỂM'),
        ('failed_logins=NHIỀU','packet_rate=CAO',  'traffic=LỚN',   '→ NGUY HIỂM'),
        ('failed_logins=NHIỀU','packet_rate=CAO',  '—',              '→ CAO'),
        ('packet_rate=CAO',   'traffic=LỚN',       '—',              '→ CAO'),
        ('packet_rate=VỪA',   'port_scan=VỪA',     '—',              '→ VỪA'),
        ('failed_logins=VỪA', 'traffic=VỪA',       '—',              '→ VỪA'),
        ('packet_rate=THẤP',  'failed_logins=ÍT',  'port_scan=ÍT',  '→ THẤP'),
        ('failed_logins=ÍT',  'port_scan=ÍT',      'traffic=NHỎ',   '→ THẤP'),
    ]
    rules_df = pd.DataFrame(rules_data, columns=['Điều kiện 1', 'Điều kiện 2', 'Điều kiện 3', 'Kết luận Nguy Cơ'])
    st.dataframe(rules_df)
