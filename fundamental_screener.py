"""
Fundamental Screener - Chấm điểm dài hạn theo 8 tiêu chí
"""
import pandas as pd


# ===== 8 TIÊU CHÍ DÀI HẠN =====
CRITERIA = {
    "pe_max": 15,           # P/E < 15
    "pb_max": 2.5,          # P/B < 2.5
    "roe_min": 15,          # ROE > 15%
    "debt_max": 1.0,        # Nợ/VCSH < 1.0
    "eps_growth_min": 10,   # EPS growth > 10%/năm
    "dividend_min": 3,      # Cổ tức > 3%/năm
    "volume_min": 1_000_000_000,  # Thanh khoản > 1 tỷ/ngày
    "market_cap_min": 5_000_000_000_000,  # Vốn hóa > 5.000 tỷ
}


def check_hard_criteria(ratios, market_data=None):
    """
    Kiểm tra 8 tiêu chí cứng.
    Trả về (pass, failed_criteria_list)
    """
    if not ratios:
        return False, ["no_data"]

    failed = []
    
    pe = ratios.get("pe")
    pb = ratios.get("pb")
    roe = ratios.get("roe")
    
    # 3 tiêu chí chính — bắt buộc có
    if pe is None or pe <= 0 or pe > CRITERIA["pe_max"]:
        failed.append(f"pe_{pe}")
    
    if pb is None or pb <= 0 or pb > CRITERIA["pb_max"]:
        failed.append(f"pb_{pb}")
    
    if roe is None or roe < CRITERIA["roe_min"]:
        failed.append(f"roe_{roe}")
    
    # Các tiêu chí phụ — nếu thiếu thì bỏ qua
    debt = ratios.get("debt_equity")
    if debt is not None and debt > CRITERIA["debt_max"]:
        failed.append(f"debt_{debt}")
    
    eps_g = ratios.get("eps_growth")
    if eps_g is not None and eps_g < CRITERIA["eps_growth_min"]:
        failed.append(f"eps_{eps_g}")
    
    div = ratios.get("dividend_yield")
    if div is not None and div < CRITERIA["dividend_min"]:
        failed.append(f"div_{div}")
    
    # Nếu fail 1 tiêu chí chính → loại
    critical_failed = [f for f in failed if f.startswith(("pe_", "pb_", "roe_"))]
    if critical_failed:
        return False, failed
    
    # Fail 2+ tiêu chí phụ → loại
    if len(failed) >= 2:
        return False, failed
    
    return True, failed


def score_fundamental_longterm(ratios):
    """
    Chấm điểm chất lượng dài hạn (0-100).
    Càng cao càng tốt cho dài hạn.
    """
    if not ratios:
        return 0, {}

    score = 0
    breakdown = {}

    pe = ratios.get("pe")
    pb = ratios.get("pb")
    roe = ratios.get("roe")
    de = ratios.get("debt_equity")
    div = ratios.get("dividend_yield")
    eps_g = ratios.get("eps_growth")

    # ===== ROE (30 điểm) — quan trọng nhất =====
    roe_score = 0
    if roe is not None:
        if roe >= 25: roe_score = 30
        elif roe >= 20: roe_score = 25
        elif roe >= 17: roe_score = 20
        elif roe >= 15: roe_score = 15
        elif roe >= 12: roe_score = 10
        else: roe_score = 5
    breakdown["roe"] = {"value": roe, "score": roe_score}
    score += roe_score

    # ===== P/E (20 điểm) =====
    pe_score = 0
    if pe is not None and pe > 0:
        if pe <= 8: pe_score = 20
        elif pe <= 10: pe_score = 17
        elif pe <= 12: pe_score = 14
        elif pe <= 15: pe_score = 10
        else: pe_score = 3
    breakdown["pe"] = {"value": pe, "score": pe_score}
    score += pe_score

    # ===== P/B (15 điểm) =====
    pb_score = 0
    if pb is not None and pb > 0:
        if pb <= 1.0: pb_score = 15
        elif pb <= 1.5: pb_score = 12
        elif pb <= 2.0: pb_score = 9
        elif pb <= 2.5: pb_score = 5
        else: pb_score = 2
    breakdown["pb"] = {"value": pb, "score": pb_score}
    score += pb_score

    # ===== EPS growth (15 điểm) =====
    eg_score = 0
    if eps_g is not None:
        if eps_g >= 25: eg_score = 15
        elif eps_g >= 20: eg_score = 13
        elif eps_g >= 15: eg_score = 10
        elif eps_g >= 10: eg_score = 7
        else: eg_score = 3
    breakdown["eps_growth"] = {"value": eps_g, "score": eg_score}
    score += eg_score

    # ===== Cổ tức (10 điểm) =====
    div_score = 0
    if div is not None:
        if div >= 6: div_score = 10
        elif div >= 5: div_score = 8
        elif div >= 4: div_score = 6
        elif div >= 3: div_score = 4
        else: div_score = 2
    breakdown["dividend"] = {"value": div, "score": div_score}
    score += div_score

    # ===== Nợ/VCSH (10 điểm) =====
    de_score = 0
    if de is not None:
        if de <= 0.3: de_score = 10
        elif de <= 0.5: de_score = 8
        elif de <= 0.7: de_score = 6
        elif de <= 1.0: de_score = 4
        else: de_score = 1
    breakdown["debt"] = {"value": de, "score": de_score}
    score += de_score

    return min(100, score), breakdown


def rank_by_longterm_score(universe_ratios):
    """
    Xếp hạng universe theo điểm dài hạn.
    
    Returns:
        List[dict]: [{symbol, score, breakdown, pass_hard}]
    """
    results = []
    
    for sym, ratios in universe_ratios.items():
        if not ratios:
            continue
        
        passed, failed = check_hard_criteria(ratios)
        score, breakdown = score_fundamental_longterm(ratios)
        
        results.append({
            "symbol": sym,
            "score": score,
            "breakdown": breakdown,
            "pass_hard": passed,
            "failed": failed,
        })
    
    # Sắp xếp theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results
