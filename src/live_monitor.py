"""
live_monitor.py
===============
Module giám sát và thu thập thông số mạng thời gian thực (Live Network Sniffer)
trên máy tính thực tế sử dụng `psutil`.

Trích xuất các đặc trưng theo chu kỳ (ví dụ: mỗi 2 giây):
    - packet_rate     : Số gói tin/giây gửi và nhận
    - failed_logins   : Ước lượng/mô phỏng hoặc đọc từ security log
    - port_scan_count : Số lượng cổng (ports) độc nhất đang kết nối/mở trong chu kỳ
    - traffic_volume  : Lưu lượng dữ liệu truyền tải quy đổi ra MB/giây
"""

import time
import psutil
from collections import deque

class LiveNetworkMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.last_io = psutil.net_io_counters()
        self.last_time = time.time()
        self.history = deque(maxlen=30)  # Lưu 30 điểm lịch sử gần nhất

    def get_realtime_metrics(self, simulated_failed_logins=0):
        """
        Lấy thông số mạng thực tế đo được qua 1 khoảng thời gian interval.
        """
        current_time = time.time()
        current_io = psutil.net_io_counters()
        elapsed = current_time - self.last_time

        if elapsed <= 0:
            elapsed = 0.001

        # 1. Tốc độ gói tin (gói/giây)
        pkts_diff = (current_io.packets_recv + current_io.packets_sent) - \
                    (self.last_io.packets_recv + self.last_io.packets_sent)
        packet_rate = max(0.0, pkts_diff / elapsed)

        # 2. Lưu lượng (MB/giây)
        bytes_diff = (current_io.bytes_recv + current_io.bytes_sent) - \
                     (self.last_io.bytes_recv + self.last_io.bytes_sent)
        traffic_volume = max(0.0, (bytes_diff / (1024 * 1024)) / elapsed * 10)  # scale tương thích dải hiển thị

        # 3. Quét cổng: Đếm số lượng remote/local ports độc nhất đang hoạt động
        unique_ports = set()
        try:
            conns = psutil.net_connections(kind='inet')
            for c in conns:
                if c.raddr:
                    unique_ports.add(c.raddr.port)
                if c.laddr:
                    unique_ports.add(c.laddr.port)
        except Exception:
            pass
        
        port_scan_count = len(unique_ports)

        # Cập nhật mốc thời gian
        self.last_io = current_io
        self.last_time = current_time

        data_point = {
            'timestamp': time.strftime('%H:%M:%S', time.localtime(current_time)),
            'packet_rate': round(packet_rate, 2),
            'failed_logins': simulated_failed_logins,
            'port_scan_count': port_scan_count,
            'traffic_volume': round(traffic_volume, 2),
        }
        self.history.append(data_point)
        return data_point


if __name__ == '__main__':
    monitor = LiveNetworkMonitor(interval=1.0)
    print("🛰️ Bắt đầu quét thông số mạng thực tế (nhấn Ctrl+C để dừng)...")
    for _ in range(5):
        time.sleep(1)
        m = monitor.get_realtime_metrics()
        print(f"[{m['timestamp']}] Gói/s: {m['packet_rate']} | Ports: {m['port_scan_count']} | Lưu lượng MB/s: {m['traffic_volume']}")
