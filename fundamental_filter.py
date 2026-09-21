"""
Fundamental Filter - Sàng lọc cổ phiếu theo cơ bản
"""
import pandas as pd
import config


# ===== NGƯỠNG LỌC =====
FUNDAMENTAL_CRITERIA = {
    "pe_max": 20,           # P/E tối đa
    "pe_min": 3,            # P/E tối thiểu
    "pb_max": 3.5,          # P/B tối đa
    "roe_min": 12,          # ROE tối thiểu
    "net_margin_min": 5,    # Biên LN ròng tối thiểu
}


def passes_fundamental_filter(ratios):
    """
    Kiểm tra cổ phiếu có đạt tiêu chuẩn cơ bản không.
    
    Đây là bộ lọc CHẶT — chỉ giữ 15-20% cổ phiếu tốt nhất.
    """
    if not ratios:
        return False, "no_data"
    
    pe = ratios.get("pe")
    pb = ratios.get("pb")
    roe = ratios.get("roe")
    nm = ratios.get("net_margin")
    
    c = FUNDAMENTAL_CRITERIA
    
    # Phải có ít nhất P/E và ROE
    if pe is None or roe is None:
        return False, "missing_pe_roe"
    
    # P/E trong khoảng hợp lý
    if pe <= c["pe_min"]:
        return False, f"pe_too_low_{pe:.1f}"
    if pe > c["pe_max"]:
        return False, f"pe_too_high_{pe:.1f}"
    
    # ROE >= 12%
    if roe < c["roe_min"]:
        return False, f"roe_low_{roe:.1f}"
    
    # P/B (nếu có)
    if pb is not None and pb > c["pb_max"]:
        return False, f"pb_high_{pb:.2f}"
    
    # Biên LN ròng (nếu có)
    if nm is not None and nm < c["net_margin_min"]:
        return False, f"margin_low_{nm:.1f}"
    
    return True, "pass"


def score_fundamental_quality(ratios):
    """
    Chấm điểm CHẤT LƯỢNG cơ bản (0-100).
    Khác với fundamental_score() — đây là chất lượng, không phải điểm số cổ phiếu.
    """
    if not ratios:
        return 0
    
    score = 0
    count = 0
    
    pe = ratios.get("pe")
    pb = ratios.get("pb")
    roe = ratios.get("roe")
    nm = ratios.get("net_margin")
    
    # P/E (30 điểm) — thấp tốt
    if pe is not None and pe > 0:
        count += 1
        if pe <= 8: score += 30
        elif pe <= 12: score += 25
        elif pe <= 15: score += 20
        elif pe <= 20: score += 12
        else: score += 3
    
    # ROE (35 điểm) — cao tốt
    if roe is not None:
        count += 1
        if roe >= 25: score += 35
        elif roe >= 20: score += 30
        elif roe >= 15: score += 22
        elif roe >= 12: score += 15
        else: score += 5
    
    # P/B (20 điểm)
    if pb is not None and pb > 0:
        count += 1
        if pb <= 1.2: score += 20
        elif pb <= 1.8: score += 15
        elif pb <= 2.5: score += 10
        elif pb <= 3.5: score += 5
        else: score += 0
    
    # Biên LN (15 điểm)
    if nm is not None:
        count += 1
        if nm >= 25: score += 15
        elif nm >= 18: score += 12
        elif nm >= 12: score += 8
        elif nm >= 5: score += 4
    
    return min(100, score)


def filter_universe_by_fundamental(universe, ratios_dict):
    """
    Lọc universe, chỉ giữ mã đạt tiêu chuẩn cơ bản.
    
    Args:
        universe: List mã
        ratios_dict: dict {symbol: ratios}
    
    Returns:
        (passed_symbols, rejected_info)
    """
    passed = []
    rejected = {}
    
    for sym in universe:
        ratios = ratios_dict.get(sym)
        ok, reason = passes_fundamental_filter(ratios)
        if ok:
            passed.append(sym)
        else:
            rejected[sym] = reason
    
    return passed, rejected
