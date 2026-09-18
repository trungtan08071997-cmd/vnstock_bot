"""
Chấm điểm chỉ số cơ bản
"""
import config


def fundamental_score(ratios):
    """Chấm điểm cơ bản (0-100)."""
    if not ratios or all(ratios.get(k) is None for k in ["pe", "pb", "roe"]):
        return 50, {"status": "no_data", "note": "Không có dữ liệu cơ bản"}

    if not ratios:
        return 0, {"error": "Không có dữ liệu"}
    score = 0
    details = {}

    # P/E (30 điểm)
    pe = ratios.get("pe")
    pe_score = 0
    if pe is not None and pe > 0:
        if pe <= 8: pe_score = 30
        elif pe <= 12: pe_score = 25
        elif pe <= 18: pe_score = 18
        elif pe <= 25: pe_score = 10
        else: pe_score = 3
    details["pe"] = {"value": pe, "score": pe_score}
    score += pe_score

    # P/B (25 điểm)
    pb = ratios.get("pb")
    pb_score = 0
    if pb is not None and pb > 0:
        if pb <= 1.0: pb_score = 25
        elif pb <= 1.5: pb_score = 20
        elif pb <= 2.5: pb_score = 14
        elif pb <= 4.0: pb_score = 8
        else: pb_score = 2
    details["pb"] = {"value": pb, "score": pb_score}
    score += pb_score

    # ROE (25 điểm)
    roe = ratios.get("roe")
    roe_score = 0
    if roe is not None:
        if roe >= 25: roe_score = 25
        elif roe >= 20: roe_score = 22
        elif roe >= 15: roe_score = 18
        elif roe >= 10: roe_score = 12
        elif roe >= 5: roe_score = 6
    details["roe"] = {"value": roe, "score": roe_score}
    score += roe_score

    # Biên LN ròng (20 điểm)
    nm = ratios.get("net_margin")
    nm_score = 0
    if nm is not None:
        if nm >= 30: nm_score = 20
        elif nm >= 20: nm_score = 16
        elif nm >= 12: nm_score = 12
        elif nm >= 6: nm_score = 7
        else: nm_score = 3
    details["net_margin"] = {"value": nm, "score": nm_score}
    score += nm_score

    return min(100, score), details


def passes_basic_filter(ratios):
    """
    Kiểm tra bộ lọc cơ bản - SOFT FILTER.
    Nếu không có dữ liệu → cho qua (chỉ cần cảnh báo).
    """
    if not ratios:
        return True  # Không có dữ liệu → vẫn cho qua
    if all(ratios.get(k) is None for k in ["pe", "pb", "roe"]):
        return True
    
    f = config.FUNDAMENTAL_FILTERS
    pe = ratios.get("pe")
    pb = ratios.get("pb")
    roe = ratios.get("roe")
    
    # Nếu TẤT CẢ đều None → cho qua (không đủ dữ liệu để đánh giá)
    if pe is None and pb is None and roe is None:
        return True
    
    # Chỉ reject khi có dữ liệu VÀ vi phạm ngưỡng
    if pe is not None:
        if pe <= f["pe_min"] or pe > f["pe_max"]:
            return False
    if pb is not None and pb > f["pb_max"]:
        return False
    if roe is not None and roe < f["roe_min"]:
        return False
    
    return True
