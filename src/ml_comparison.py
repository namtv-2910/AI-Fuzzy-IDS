"""
ml_comparison.py
================
So sánh Fuzzy IDS với các mô hình Machine Learning truyền thống:
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Decision Tree
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import joblib
import os


MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'KNN':           KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
}


def train_and_evaluate(X_scaled, y, test_size=0.2, random_state=42):
    """
    Huấn luyện và đánh giá tất cả mô hình ML.
    Trả về DataFrame kết quả và dict các model đã train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = []
    trained_models = {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        results.append({
            'Model':        name,
            'Accuracy':     accuracy_score(y_test, y_pred),
            'Precision':    precision_score(y_test, y_pred, zero_division=0),
            'Recall':       recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':     f1_score(y_test, y_pred, zero_division=0),
            'CV Mean':      cv_scores.mean(),
            'CV Std':       cv_scores.std(),
        })
        trained_models[name] = (model, X_test, y_test, y_pred)
        print(f"✅ {name}: Accuracy={accuracy_score(y_test, y_pred):.4f}, F1={f1_score(y_test, y_pred):.4f}")

    return pd.DataFrame(results), trained_models, X_test, y_test


def evaluate_fuzzy(fuzzy_ids, X_raw, y, threshold=50.0):
    """
    Đánh giá Fuzzy IDS trên toàn bộ dữ liệu.
    Score > threshold → tấn công (1), ngược lại bình thường (0).
    """
    scores = fuzzy_ids.predict_batch(X_raw)
    y_pred = [1 if s >= threshold else 0 for s in scores]

    return {
        'Model':     'Fuzzy Logic',
        'Accuracy':  accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall':    recall_score(y, y_pred, zero_division=0),
        'F1-Score':  f1_score(y, y_pred, zero_division=0),
        'CV Mean':   accuracy_score(y, y_pred),
        'CV Std':    0.0,
    }, y_pred


def plot_comparison(results_df, save_path=None):
    """Vẽ biểu đồ so sánh các mô hình."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('So sánh hiệu năng: Fuzzy Logic vs ML Models', fontsize=14, fontweight='bold')

    # --- Bar chart ---
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Các chỉ số đánh giá')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Radar / Line chart ---
    ax2 = axes[1]
    for _, row in results_df.iterrows():
        ax2.plot(metrics, [row[m] for m in metrics], marker='o', linewidth=2, label=row['Model'])
        ax2.fill(metrics, [row[m] for m in metrics], alpha=0.05)

    ax2.set_ylim(0, 1.1)
    ax2.set_title('So sánh theo từng metric')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name='Model', save_path=None):
    """Vẽ confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bình thường', 'Tấn công'],
                yticklabels=['Bình thường', 'Tấn công'])
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    ax.set_ylabel('Thực tế')
    ax.set_xlabel('Dự đoán')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def save_best_model(trained_models, results_df, save_dir='models'):
    """Lưu model tốt nhất (theo F1-Score)."""
    os.makedirs(save_dir, exist_ok=True)
    best_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_model = trained_models[best_name][0]
    path = os.path.join(save_dir, 'best_model.pkl')
    joblib.dump(best_model, path)
    print(f"💾 Đã lưu model tốt nhất [{best_name}] tại: {path}")
    return best_name, path


if __name__ == '__main__':
    from preprocessing import load_or_generate_data, preprocess

    df = load_or_generate_data()
    X_raw, X_scaled, y, scaler = preprocess(df)

    results_df, trained_models, X_test, y_test = train_and_evaluate(X_scaled, y)
    print("\n", results_df.to_string(index=False))

    save_best_model(trained_models, results_df)
