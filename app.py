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
                      padding:1rem; border-radius:8px; }
    .alert-medium   { background:#fef9e7; border-left:4px solid #f1c40f;
                      padding:1rem; border-radius:8px; }
    .alert-high     { background:#fdebd0; border-left:4px solid #e67e22;
                      padding:1rem; border-radius:8px; }
    .alert-critical { background:#fadbd8; border-left:4px solid #e74c3c;
                      padding:1rem; border-radius:8px; }
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
    st.image('https://img.icons8.com/fluency/96/shield.png', width=80)
    st.markdown('## 🛡️ Fuzzy IDS')
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
    st.markdown('### 🔄 Quy trình hoạt động')
    st.image('https://mermaid.ink/img/pako:eNpVkMFqwzAMhl_F-JRC3iCHHVZa2G2ww2CHXoStJF5tK9hyIIS8-5ymGxuGQH_-T7-krJCFVFCQaEiXJG-xfAcFG-0E4gj3FNrxTkMI7pGRR2rh2X2HjSfLKVwL8gS-o5VqJtRBzidCY4kL7cA5F3YJPuBk3zHGiOicqhWPdl98NONkHiuKL1r5jNGCvEYKlTkdAfVbIR6YqHSk3S4yjqnIMhf3v6YnLMk6C7W-oFH5-nxHp6jbhSSQA_iSV4xWx6EJ1LGMJJjvPJNd38h6p2BrjcmTkwQgJzikuuIiLLxBkSR-B-Q_Y2uMx', use_container_width=True)

    st.markdown('### 🗂️ Tính năng')
    c1, c2 = st.columns(2)
    with c1:
        st.success('✅ Phân tích traffic mạng theo thời gian thực')
        st.success('✅ Giải thích lý do cảnh báo (Explainable AI)')
        st.success('✅ So sánh với 4 mô hình ML truyền thống')
    with c2:
        st.info('📊 Trực quan hoá hàm thuộc Fuzzy')
        st.info('📈 Biểu đồ phân tích dữ liệu')
        st.info('🧠 Hướng dẫn lý thuyết Fuzzy Logic')


# ═══════════════════════════════════════════════════════════════════════
# TRANG 2 — Phân tích đơn
# ═══════════════════════════════════════════════════════════════════════
elif page == '🔍 Phân tích đơn':
    st.markdown('## 🔍 Phân tích Traffic Mạng')
    st.markdown('Nhập thông số traffic mạng để hệ thống đánh giá mức độ nguy hiểm.')
    st.divider()

    fuzzy = load_fuzzy()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('### ⚙️ Thông số đầu vào')

        pr = st.slider('📡 Tốc độ gói tin (packet/giây)', 0, 1000, 100,
                       help='Số lượng gói tin mạng mỗi giây')
        fl = st.slider('🔑 Số lần đăng nhập thất bại', 0, 100, 5,
                       help='Số lần login thất bại trong 1 phút')
        ps = st.slider('🔍 Số port bị quét', 0, 1000, 50,
                       help='Số lượng port bị scan')
        tv = st.slider('💾 Lưu lượng dữ liệu (MB)', 0, 1000, 150,
                       help='Tổng lưu lượng dữ liệu truyền qua')

        st.divider()
        st.markdown('**🎯 Thử nhanh:**')
        c1b, c2b, c3b = st.columns(3)
        preset = None
        if c1b.button('✅ Bình thường'):  preset = (50, 2, 20, 100)
        if c2b.button('⚠️ Nghi ngờ'):    preset = (400, 30, 400, 500)
        if c3b.button('🚨 Tấn công'):     preset = (900, 80, 900, 950)

        if preset:
            pr, fl, ps, tv = preset

    with col2:
        st.markdown('### 📊 Kết quả đánh giá')
        result = fuzzy.predict(pr, fl, ps, tv)
        score  = result['score']

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=score,
            title={'text': 'Mức độ nguy hiểm', 'font': {'size': 18}},
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
        st.plotly_chart(fig, use_container_width=True)

        # Alert box
        level_map = {
            'low':      ('alert-low',      '🟢 BÌNH THƯỜNG',    'Traffic mạng hoạt động bình thường. Không phát hiện dấu hiệu tấn công.'),
            'medium':   ('alert-medium',   '🟡 CẢNH BÁO THẤP', 'Phát hiện một số dấu hiệu bất thường. Cần theo dõi thêm.'),
            'high':     ('alert-high',     '🟠 CẢNH BÁO CAO',  'Dấu hiệu tấn công rõ ràng! Nên kiểm tra ngay hệ thống.'),
            'critical': ('alert-critical', '🔴 NGUY HIỂM!',     'Hệ thống đang bị tấn công! Hành động ngay lập tức!'),
        }
        css_cls, title, msg = level_map[result['level']]
        st.markdown(f'<div class="{css_cls}"><strong>{title}</strong><br>{msg}</div>',
                    unsafe_allow_html=True)

        # Độ thuộc từng biến
        st.divider()
        st.markdown('**📋 Lý do phân loại (Fuzzy membership):**')
        reasons = {
            f'packet_rate = {pr}':     ('Thấp' if pr < 350 else 'Trung bình' if pr < 650 else 'Cao'),
            f'failed_logins = {fl}':   ('Ít' if fl < 25 else 'Vừa' if fl < 60 else 'Nhiều'),
            f'port_scan = {ps}':       ('Ít' if ps < 300 else 'Vừa' if ps < 650 else 'Nhiều'),
            f'traffic_volume = {tv}':  ('Nhỏ' if tv < 300 else 'Vừa' if tv < 650 else 'Lớn'),
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

    st.markdown('### 📋 Bảng so sánh')
    styled = all_results.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
        'Recall': '{:.4f}', 'F1-Score': '{:.4f}',
        'CV Mean': '{:.4f}', 'CV Std': '{:.4f}',
    }).highlight_max(subset=['Accuracy', 'F1-Score'], color='#d5f5e3') \
      .highlight_min(subset=['Accuracy', 'F1-Score'], color='#fadbd8')
    st.dataframe(styled, use_container_width=True)

    st.markdown('### 📊 Biểu đồ so sánh')
    fig = plot_comparison(all_results)
    st.pyplot(fig, use_container_width=True)

    st.markdown('### 💡 Nhận xét')
    best = all_results.loc[all_results['F1-Score'].idxmax(), 'Model']
    st.info(f"""
    - **Mô hình tốt nhất** (F1-Score): **{best}**
    - Fuzzy Logic **không cần dữ liệu huấn luyện** — hoạt động dựa trên luật chuyên gia
    - Fuzzy Logic có ưu thế về **tính giải thích được (Explainability)**
    - ML models có lợi thế khi có **nhiều dữ liệu chất lượng cao**
    """)


# ═══════════════════════════════════════════════════════════════════════
# TRANG 4 — Phân tích dữ liệu
# ═══════════════════════════════════════════════════════════════════════
elif page == '📈 Phân tích dữ liệu':
    st.markdown('## 📈 Phân tích Dữ liệu Traffic Mạng')

    df = load_data()
    st.markdown(f'**Dataset:** {len(df)} mẫu — {df["label"].value_counts()[0]} bình thường / {df["label"].value_counts()[1]} tấn công')
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='packet_rate', color='label',
                           title='Phân phối tốc độ gói tin',
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                           labels={'label': 'Nhãn', 'packet_rate': 'Packet Rate'},
                           barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x='packet_rate', y='failed_logins',
                          color=df['label'].map({0: 'Bình thường', 1: 'Tấn công'}),
                          title='Packet Rate vs Failed Logins',
                          color_discrete_map={'Bình thường': '#2ecc71', 'Tấn công': '#e74c3c'},
                          opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter_matrix(df, dimensions=['packet_rate', 'failed_logins', 'port_scan_count', 'traffic_volume'],
                              color=df['label'].map({0: 'Bình thường', 1: 'Tấn công'}),
                              color_discrete_map={'Bình thường': '#2ecc71', 'Tấn công': '#e74c3c'},
                              title='Ma trận phân tán các đặc trưng', opacity=0.5)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TRANG 5 — Lý thuyết Fuzzy
# ═══════════════════════════════════════════════════════════════════════
elif page == '🧠 Lý thuyết Fuzzy':
    st.markdown('## 🧠 Lý thuyết Fuzzy Logic')

    fuzzy = load_fuzzy()

    st.markdown("""
    ### Fuzzy Logic là gì?
    **Logic Mờ (Fuzzy Logic)** cho phép giá trị chân lý nằm trong khoảng **[0, 1]**
    thay vì chỉ True/False như logic nhị phân. Điều này phản ánh sự không chắc chắn
    trong thế giới thực.

    ### Hệ thống suy luận Mamdani (FIS)
    1. **Fuzzification** — Chuyển giá trị số thành độ thuộc fuzzy
    2. **Rule Evaluation** — Áp dụng các luật IF-THEN
    3. **Aggregation** — Tổng hợp kết quả các luật
    4. **Defuzzification** — Chuyển kết quả mờ thành giá trị số (Centroid method)
    """)

    st.markdown('### 📊 Hàm thuộc (Membership Functions)')
    fig = fuzzy.plot_membership_functions()
    st.pyplot(fig, use_container_width=True)

    st.markdown('### 📋 Các luật Fuzzy IF-THEN trong hệ thống')
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
    rules_df = pd.DataFrame(rules_data, columns=['Điều kiện 1', 'Điều kiện 2', 'Điều kiện 3', 'Kết quả'])
    st.dataframe(rules_df, use_container_width=True)
