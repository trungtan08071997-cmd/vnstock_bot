"""
Wrapper để test nhanh backtest với ít mã
"""
import config

# Override VN30_LIST để test nhanh
config.VN30_LIST = ["VCB", "HPG", "FPT", "MWG", "VHM"]

# Chạy backtest
from backtest_engine import main
main()
