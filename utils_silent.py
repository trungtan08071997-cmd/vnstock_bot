"""
Utility để chặn banner quảng cáo vnstock
"""
import sys
import io
import contextlib


class _SilentFilter:
    """
    Lọc các dòng có chứa từ khóa quảng cáo vnstock ra khỏi stdout.
    """
    KEYWORDS = [
        "VNSTOCK INSIDERS PROGRAM",
        "vnstocks.com/insiders",
        "NÂNG TẦM TRẢI NGHIỆM",
        "Ẩn toàn bộ thông báo",
        "Mở rộng khả năng sử dụng API",
        "Tăng tốc tải dữ liệu",
        "Tăng giới hạn truy cập API",
        "Dùng AI Agent viết code",
        "Tham gia ngay",
        "╔" * 0,  # tránh lọc nhầm
        "╚" * 0,
    ]

    def __init__(self, original):
        self.original = original
        self.buffer = []

    def write(self, text):
        # Nếu text chứa bất kỳ keyword nào → bỏ qua
        if any(kw in text for kw in self.KEYWORDS if kw):
            return
        self.original.write(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextlib.contextmanager
def silence_vnstock():
    """
    Context manager: chặn mọi output từ vnstock trong block.

    Usage:
        with silence_vnstock():
            listing = Listing(source="KBS")
            df = listing.all_symbols()
    """
    old_stdout = sys.stdout
    sys.stdout = _SilentFilter(old_stdout)
    try:
        yield
    finally:
        sys.stdout = old_stdout