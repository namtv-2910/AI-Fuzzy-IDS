"""
fuzzy_ids.py
============
Hệ thống phát hiện bất thường mạng bằng Fuzzy Logic (Mamdani FIS).

Input variables:
    - packet_rate      : Tốc độ gói tin (gói/giây)  → [0, 1000]
    - failed_logins    : Số lần đăng nhập thất bại   → [0, 100]
    - port_scan_count  : Số port bị quét              → [0, 1000]
    - traffic_volume   : Lưu lượng dữ liệu (MB)       → [0, 1000]

Output variable:
    - threat_level : Mức độ nguy hiểm                 → [0, 100]
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FuzzyIDS:
    """Fuzzy Inference System cho phát hiện xâm nhập mạng."""

    def __init__(self):
        self._define_variables()
        self._define_membership_functions()
        self._define_rules()
        self._build_system()

    # ------------------------------------------------------------------
    # 1. Định nghĩa biến ngôn ngữ
    # ------------------------------------------------------------------
    def _define_variables(self):
        self.packet_rate     = ctrl.Antecedent(np.arange(0, 1001, 1), 'packet_rate')
        self.failed_logins   = ctrl.Antecedent(np.arange(0, 101,  1), 'failed_logins')
        self.port_scan_count = ctrl.Antecedent(np.arange(0, 1001, 1), 'port_scan_count')
        self.traffic_volume  = ctrl.Antecedent(np.arange(0, 1001, 1), 'traffic_volume')
        self.threat_level    = ctrl.Consequent(np.arange(0, 101,  1), 'threat_level')

    # ------------------------------------------------------------------
    # 2. Định nghĩa hàm thuộc (Membership Functions)
    # ------------------------------------------------------------------
    def _define_membership_functions(self):
        # --- packet_rate ---
        self.packet_rate['low']    = fuzz.trimf(self.packet_rate.universe,     [0,   0,   350])
        self.packet_rate['medium'] = fuzz.trimf(self.packet_rate.universe,     [150, 400, 650])
        self.packet_rate['high']   = fuzz.trimf(self.packet_rate.universe,     [500, 1000, 1000])

        # --- failed_logins ---
        self.failed_logins['few']      = fuzz.trimf(self.failed_logins.universe, [0,  0,  25])
        self.failed_logins['moderate'] = fuzz.trimf(self.failed_logins.universe, [10, 35, 60])
        self.failed_logins['many']     = fuzz.trimf(self.failed_logins.universe, [45, 100, 100])

        # --- port_scan_count ---
        self.port_scan_count['few']      = fuzz.trimf(self.port_scan_count.universe, [0,   0,   300])
        self.port_scan_count['moderate'] = fuzz.trimf(self.port_scan_count.universe, [150, 400, 650])
        self.port_scan_count['many']     = fuzz.trimf(self.port_scan_count.universe, [500, 1000, 1000])

        # --- traffic_volume ---
        self.traffic_volume['small']  = fuzz.trimf(self.traffic_volume.universe, [0,   0,   300])
        self.traffic_volume['medium'] = fuzz.trimf(self.traffic_volume.universe, [150, 400, 650])
        self.traffic_volume['large']  = fuzz.trimf(self.traffic_volume.universe, [500, 1000, 1000])

        # --- threat_level (output) ---
        self.threat_level['low']      = fuzz.trimf(self.threat_level.universe, [0,  0,  30])
        self.threat_level['medium']   = fuzz.trimf(self.threat_level.universe, [20, 50, 80])
        self.threat_level['high']     = fuzz.trimf(self.threat_level.universe, [60, 80, 95])
        self.threat_level['critical'] = fuzz.trimf(self.threat_level.universe, [80, 100, 100])

    # ------------------------------------------------------------------
    # 3. Định nghĩa luật Fuzzy IF-THEN
    # ------------------------------------------------------------------
    def _define_rules(self):
        pr  = self.packet_rate
        fl  = self.failed_logins
        ps  = self.port_scan_count
        tv  = self.traffic_volume
        thr = self.threat_level

        self.rules = [
            # === NGUY HIỂM NGHIÊM TRỌNG (CRITICAL) ===
            ctrl.Rule(pr['high'] & ps['many'],                      thr['critical']),
            ctrl.Rule(fl['many'] & ps['many'],                      thr['critical']),
            ctrl.Rule(fl['many'] & pr['high'] & tv['large'],        thr['critical']),

            # === NGUY HIỂM CAO (HIGH) ===
            ctrl.Rule(fl['many'] & pr['high'],                      thr['high']),
            ctrl.Rule(pr['high'] & tv['large'],                     thr['high']),
            ctrl.Rule(fl['many'] & ps['moderate'],                  thr['high']),
            ctrl.Rule(ps['many'] & tv['large'],                     thr['high']),

            # === CẢNH BÁO (MEDIUM) ===
            ctrl.Rule(pr['medium'] & ps['moderate'],                thr['medium']),
            ctrl.Rule(fl['moderate'] & pr['medium'],                thr['medium']),
            ctrl.Rule(fl['moderate'] & tv['medium'],                thr['medium']),
            ctrl.Rule(ps['moderate'] & tv['medium'],                thr['medium']),
            ctrl.Rule(pr['high'] & fl['few'] & ps['few'],           thr['medium']),

            # === BÌNH THƯỜNG (LOW) ===
            ctrl.Rule(pr['low'] & fl['few'] & ps['few'],            thr['low']),
            ctrl.Rule(fl['few'] & ps['few'] & tv['small'],          thr['low']),
            ctrl.Rule(pr['low'] & tv['small'],                      thr['low']),
        ]

    # ------------------------------------------------------------------
    # 4. Xây dựng hệ thống điều khiển
    # ------------------------------------------------------------------
    def _build_system(self):
        system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(system)

    # ------------------------------------------------------------------
    # 5. Dự đoán
    # ------------------------------------------------------------------
    def predict(self, packet_rate: float, failed_logins: float,
                port_scan_count: float, traffic_volume: float) -> dict:
        """
        Trả về dict gồm:
            - score     : giá trị threat_level [0, 100]
            - label     : nhãn tiếng Việt
            - level     : 'low' | 'medium' | 'high' | 'critical'
            - emoji     : biểu tượng cảm xúc
            - color     : màu hiển thị
        """
        self.simulation.input['packet_rate']     = float(np.clip(packet_rate,     0, 1000))
        self.simulation.input['failed_logins']   = float(np.clip(failed_logins,   0, 100))
        self.simulation.input['port_scan_count'] = float(np.clip(port_scan_count, 0, 1000))
        self.simulation.input['traffic_volume']  = float(np.clip(traffic_volume,  0, 1000))

        self.simulation.compute()
        score = self.simulation.output['threat_level']

        if score < 25:
            return dict(score=score, label='BÌNH THƯỜNG',       level='low',      emoji='🟢', color='#2ecc71')
        elif score < 50:
            return dict(score=score, label='CẢNH BÁO THẤP',    level='medium',   emoji='🟡', color='#f1c40f')
        elif score < 75:
            return dict(score=score, label='CẢNH BÁO CAO',     level='high',     emoji='🟠', color='#e67e22')
        else:
            return dict(score=score, label='NGUY HIỂM!',       level='critical', emoji='🔴', color='#e74c3c')

    def predict_batch(self, df) -> list:
        """Dự đoán cho toàn bộ DataFrame."""
        results = []
        for _, row in df.iterrows():
            try:
                r = self.predict(
                    row['packet_rate'],
                    row['failed_logins'],
                    row['port_scan_count'],
                    row['traffic_volume']
                )
                results.append(r['score'])
            except Exception:
                results.append(50.0)  # giá trị mặc định nếu lỗi
        return results

    # ------------------------------------------------------------------
    # 6. Vẽ membership functions
    # ------------------------------------------------------------------
    def plot_membership_functions(self, save_path=None):
        """Vẽ hàm thuộc của tất cả biến."""
        variables = {
            'Tốc độ gói tin (packet/s)': self.packet_rate,
            'Đăng nhập thất bại':        self.failed_logins,
            'Số port bị quét':           self.port_scan_count,
            'Lưu lượng (MB)':            self.traffic_volume,
            'Mức độ nguy hiểm (output)': self.threat_level,
        }

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Hàm Thuộc (Membership Functions) — Fuzzy IDS', fontsize=14, fontweight='bold')
        axes = axes.flatten()

        colors = {'low': '#2ecc71', 'few': '#2ecc71', 'small': '#2ecc71',
                  'medium': '#f1c40f', 'moderate': '#f1c40f',
                  'high': '#e67e22', 'many': '#e67e22', 'large': '#e67e22',
                  'critical': '#e74c3c'}

        for ax, (title, var) in zip(axes, variables.items()):
            for label, mf in var.terms.items():
                color = colors.get(label, '#3498db')
                ax.plot(var.universe, mf.mf, label=label, linewidth=2, color=color)
                ax.fill_between(var.universe, mf.mf, alpha=0.1, color=color)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('μ(x)')
            ax.set_ylim(-0.05, 1.1)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == '__main__':
    ids = FuzzyIDS()
    print("=== TEST FUZZY IDS ===\n")

    test_cases = [
        ("Bình thường",    50,   3,   20,  100),
        ("Nghi ngờ",      300,  25,  300,  400),
        ("Tấn công",      800,  70,  800,  900),
        ("DDoS",         1000,   5, 1000, 1000),
    ]

    for name, pr, fl, ps, tv in test_cases:
        result = ids.predict(pr, fl, ps, tv)
        print(f"[{name}]")
        print(f"  Input  : packet_rate={pr}, failed={fl}, port_scan={ps}, traffic={tv}")
        print(f"  Output : {result['emoji']} {result['label']} (score={result['score']:.1f})\n")
