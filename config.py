"""
Cấu hình toàn bộ bot
"""
import os

# ============================================================
# TẮT QUẢNG CÁO VNSTOCK
# ============================================================
os.environ["VNSTOCK_SHOW_ADS"] = "0"
os.environ["VNSTOCK_QUIET"] = "1"

# ============================================================
# KÊNH THÔNG BÁO
# ============================================================
NOTIFY_CHANNELS = ["console", "discord"]

# Discord Webhook - DÁN URL CỦA BẠN VÀO ĐÂY
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discordapp.com/api/webhooks/1550126816365252650/HW-ygjoYgZSq-TYH_15IBFR0qYAP-ahpP22L5_jJG5x9zL3O5B2g3uTLV4LxhOLzetzu"
)

# ============================================================
# CHẾ ĐỘ PHÂN TÍCH
# ============================================================
# "vn30" = chỉ VN30 | "news" = từ tin tức | "both" = kết hợp
ANALYSIS_MODE = "vn30"

# ============================================================
# DANH SÁCH VN30 (cập nhật kỳ review mới nhất)
# ============================================================
VN30_LIST = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]

# ============================================================
# WATCHLIST FALLBACK
# ============================================================
WATCHLIST = VN30_LIST

# ============================================================
# CRAWL TIN TỨC
# ============================================================
NEWS_HOURS_BACK = 72        # Lấy tin trong 72h gần đây
TOP_HOT_SYMBOLS = 30        # Số mã hot tối đa
MIN_MENTIONS = 2            # Số lần xuất hiện tối thiểu

# ============================================================
# BỘ LỌC CƠ BẢN
# ============================================================
FUNDAMENTAL_FILTERS = {
    "pe_max": 30,           # P/E tối đa
    "pe_min": 0,            # P/E tối thiểu
    "pb_max": 5.0,          # P/B tối đa
    "roe_min": 8.0,         # ROE tối thiểu (%)
}

# ============================================================
# TRỌNG SỐ CHẤM ĐIỂM (khi có đủ dữ liệu cơ bản)
# ============================================================
SCORING_WEIGHTS = {
    "technical": 0.40,      # 40% - chỉ báo kỹ thuật
    "fundamental": 0.35,    # 35% - chỉ số cơ bản
    "sentiment": 0.25,      # 25% - tin tức
}

# ============================================================
# TRỌNG SỐ KHI THIẾU DỮ LIỆU CƠ BẢN (P/E, P/B, ROE = None)
# ============================================================
SCORING_WEIGHTS_NO_FUNDAMENTAL = {
    "technical": 0.65,      # 65% - chỉ báo kỹ thuật
    "sentiment": 0.35,      # 35% - tin tức
}

# ============================================================
# TRỌNG SỐ NGÀNH (30% điểm ngành, 70% điểm cổ phiếu)
# ============================================================
SECTOR_WEIGHT = 0.30

# ============================================================
# NGƯỠNG TÍN HIỆU
# ============================================================
SIGNAL_THRESHOLDS = {
    "strong_buy": 75,       # >= 75: MUA MẠNH
    "buy": 60,              # >= 60: MUA
    "hold": 45,             # >= 45: GIỮ
    "sell": 30,             # >= 30: BÁN
                            # < 30:  BÁN MẠNH
}

# ============================================================
# OUTPUT & HIỂN THỊ
# ============================================================
OUTPUT_DIR = "output"
REPORT_FILE = "report.json"
TOP_SECTORS = 6             # Số ngành ưu tiên
TOP_STOCKS = 10             # Số cổ phiếu top
HISTORY_DAYS = 250          # Số ngày lịch sử (~1 năm)

# ============================================================
# CACHE
# ============================================================
CACHE_DIR = "output/cache"