"""
Phân tích ngành dựa trên dữ liệu lợi nhuận
- SECTOR_DATA: Dữ liệu tăng trưởng Q2, dự báo cả năm, biên LN
- SYMBOL_SECTOR_MAP: Mapping mã CK → ngành (đã bổ sung VN30 + mở rộng)
"""
import config


# ============================================================
# DỮ LIỆU NGÀNH (cập nhật theo quý)
# ============================================================
SECTOR_DATA = {
    "Ngân hàng": {
        "q2_growth": 24.7, "fy_growth": 18, "margin": 25,
        "concentration": "medium",
        "top_stocks": ["VCB", "CTG", "VPB", "TCB", "MBB", "HDB", "ACB"],
        "risk": "NIM chịu áp lực, nợ xấu tăng",
    },
    "Bất động sản dân cư": {
        "q2_growth": 257, "fy_growth": 180, "margin": 35,
        "concentration": "high",
        "top_stocks": ["VHM", "VIC", "VRE", "KDH", "NLG", "BCM"],
        "risk": "Phân hóa cực mạnh, 80% LN từ Vingroup",
    },
    "Dầu khí": {
        "q2_growth": 244, "fy_growth": 57, "margin": 20,
        "concentration": "high",
        "top_stocks": ["BSR", "PVD", "GAS", "PVS", "PLX"],
        "risk": "Tăng trưởng chu kỳ, sẽ giảm khi nền so sánh cao",
    },
    "Thép & Nguyên vật liệu": {
        "q2_growth": 83.2, "fy_growth": 39, "margin": 15,
        "concentration": "high",
        "top_stocks": ["HPG", "HSG", "NKG", "GVR", "SMA", "VSC"],
        "risk": "Phụ thuộc giá thép và nhu cầu xây dựng",
    },
    "Bán lẻ": {
        "q2_growth": 105.5, "fy_growth": 25, "margin": 8,
        "concentration": "low",
        "top_stocks": ["MWG", "PNJ", "FRT", "DGW"],
        "risk": "Phục hồi từ nền thấp, cần theo dõi sức mua",
    },
    "Chứng khoán": {
        "q2_growth": 61, "fy_growth": 47, "margin": 30,
        "concentration": "high",
        "top_stocks": ["SSI", "VND", "HCM", "VCI", "MBS"],
        "risk": "Phân hóa cực mạnh, thanh khoản biến động",
    },
    "Thực phẩm & Đồ uống": {
        "q2_growth": 34.5, "fy_growth": 12, "margin": 18,
        "concentration": "low",
        "top_stocks": ["VNM", "MSN", "SAB"],
        "risk": "Cạnh tranh gay gắt, chi phí nguyên liệu",
    },
    "Công nghệ thông tin": {
        "q2_growth": 18.9, "fy_growth": 20, "margin": 15,
        "concentration": "medium",
        "top_stocks": ["FPT", "CMG", "ELC"],
        "risk": "FPT giảm do thay đổi hạch toán",
    },
    "Điện & Tiện ích": {
        "q2_growth": 46.1, "fy_growth": 15, "margin": 12,
        "concentration": "low",
        "top_stocks": ["POW", "REE", "NT2", "GEG"],
        "risk": "Phụ thuộc thủy văn và giá điện",
    },
    "Bảo hiểm": {
        "q2_growth": 20.1, "fy_growth": 15, "margin": 20,
        "concentration": "medium",
        "top_stocks": ["BVH", "BMI", "PVI"],
        "risk": "Lãi suất và tỷ lệ bồi thường",
    },
    "Y tế & Dược phẩm": {
        "q2_growth": 12.0, "fy_growth": 10, "margin": 15,
        "concentration": "low",
        "top_stocks": ["DHG", "IMP", "DMC", "TRA"],
        "risk": "Cạnh tranh giá, phụ thuộc nhập khẩu nguyên liệu",
    },
    "Hóa chất": {
        "q2_growth": 55.4, "fy_growth": 30, "margin": 12,
        "concentration": "medium",
        "top_stocks": ["DPM", "DCM", "DGC", "CSV"],
        "risk": "Phụ thuộc giá dầu và nguyên liệu đầu vào",
    },
    "Xây dựng & Vật liệu": {
        "q2_growth": 13.2, "fy_growth": 20, "margin": 10,
        "concentration": "low",
        "top_stocks": ["CTD", "VCG", "C4G", "FCN"],
        "risk": "Phụ thuộc đầu tư công và thị trường BĐS",
    },
    "Du lịch & Giải trí": {
        "q2_growth": -81.2, "fy_growth": -20, "margin": 5,
        "concentration": "low",
        "top_stocks": ["VJC", "HVN"],
        "risk": "Biến động mạnh, phụ thuộc mùa du lịch",
    },
}


# ============================================================
# MAPPING MÃ CK → NGÀNH (đã bổ sung VN30 + nhiều mã khác)
# ============================================================
SYMBOL_SECTOR_MAP = {
    # ===== NGÂN HÀNG =====
    "VCB": "Ngân hàng", "CTG": "Ngân hàng", "BID": "Ngân hàng",
    "TCB": "Ngân hàng", "VPB": "Ngân hàng", "MBB": "Ngân hàng",
    "ACB": "Ngân hàng", "HDB": "Ngân hàng", "STB": "Ngân hàng",
    "SHB": "Ngân hàng", "SSB": "Ngân hàng", "VIB": "Ngân hàng",
    "TPB": "Ngân hàng", "LPB": "Ngân hàng", "OCB": "Ngân hàng",
    "MSB": "Ngân hàng", "EIB": "Ngân hàng", "NAB": "Ngân hàng",
    "PGB": "Ngân hàng", "SGB": "Ngân hàng", "BVB": "Ngân hàng",
    "BAB": "Ngân hàng", "KLB": "Ngân hàng", "VBB": "Ngân hàng",

    # ===== BẤT ĐỘNG SẢN DÂN CƯ =====
    "VHM": "Bất động sản dân cư", "VIC": "Bất động sản dân cư",
    "VRE": "Bất động sản dân cư", "BCM": "Bất động sản dân cư",
    "KDH": "Bất động sản dân cư", "NLG": "Bất động sản dân cư",
    "DXG": "Bất động sản dân cư", "PDR": "Bất động sản dân cư",
    "NVL": "Bất động sản dân cư", "SCR": "Bất động sản dân cư",
    "DIG": "Bất động sản dân cư", "CEO": "Bất động sản dân cư",
    "HDG": "Bất động sản dân cư", "TCH": "Bất động sản dân cư",

    # ===== THÉP & NGUYÊN VẬT LIỆU =====
    "HPG": "Thép & Nguyên vật liệu", "HSG": "Thép & Nguyên vật liệu",
    "NKG": "Thép & Nguyên vật liệu", "GVR": "Thép & Nguyên vật liệu",
    "SMA": "Thép & Nguyên vật liệu", "VSC": "Thép & Nguyên vật liệu",
    "TTF": "Thép & Nguyên vật liệu", "BIG": "Thép & Nguyên vật liệu",
    "TLH": "Thép & Nguyên vật liệu", "POM": "Thép & Nguyên vật liệu",
    "SMC": "Thép & Nguyên vật liệu",

    # ===== DẦU KHÍ =====
    "GAS": "Dầu khí", "BSR": "Dầu khí", "PVD": "Dầu khí",
    "PVS": "Dầu khí", "PLX": "Dầu khí", "OIL": "Dầu khí",
    "PVC": "Dầu khí", "PVB": "Dầu khí", "PVP": "Dầu khí",

    # ===== THỰC PHẨM & ĐỒ UỐNG =====
    "VNM": "Thực phẩm & Đồ uống", "MSN": "Thực phẩm & Đồ uống",
    "SAB": "Thực phẩm & Đồ uống", "SBT": "Thực phẩm & Đồ uống",
    "KDC": "Thực phẩm & Đồ uống", "MCH": "Thực phẩm & Đồ uống",
    "QNS": "Thực phẩm & Đồ uống", "DBC": "Thực phẩm & Đồ uống",
    "BAF": "Thực phẩm & Đồ uống",

    # ===== BÁN LẺ =====
    "MWG": "Bán lẻ", "PNJ": "Bán lẻ", "FRT": "Bán lẻ",
    "DGW": "Bán lẻ", "PET": "Bán lẻ",

    # ===== CÔNG NGHỆ THÔNG TIN =====
    "FPT": "Công nghệ thông tin", "CMG": "Công nghệ thông tin",
    "ELC": "Công nghệ thông tin", "ITD": "Công nghệ thông tin",

    # ===== CHỨNG KHOÁN =====
    "SSI": "Chứng khoán", "VND": "Chứng khoán", "HCM": "Chứng khoán",
    "VCI": "Chứng khoán", "MBS": "Chứng khoán", "KAI": "Chứng khoán",
    "IPA": "Chứng khoán", "SVC": "Chứng khoán", "AGR": "Chứng khoán",
    "BSI": "Chứng khoán", "CTS": "Chứng khoán", "FTS": "Chứng khoán",
    "VIX": "Chứng khoán", "SHS": "Chứng khoán", "VDS": "Chứng khoán",
    "ORS": "Chứng khoán", "TVB": "Chứng khoán", "VPB": "Ngân hàng",  # VPB ưu tiên ngân hàng

    # ===== ĐIỆN & TIỆN ÍCH =====
    "POW": "Điện & Tiện ích", "REE": "Điện & Tiện ích",
    "NT2": "Điện & Tiện ích", "GEG": "Điện & Tiện ích",
    "PPC": "Điện & Tiện ích", "BTP": "Điện & Tiện ích",
    "SJD": "Điện & Tiện ích", "TBC": "Điện & Tiện ích",

    # ===== BẢO HIỂM =====
    "BVH": "Bảo hiểm", "BMI": "Bảo hiểm", "PVI": "Bảo hiểm",
    "BIC": "Bảo hiểm", "MIG": "Bảo hiểm", "PGI": "Bảo hiểm",
    "EVF": "Bảo hiểm", "ABI": "Bảo hiểm",

    # ===== Y TẾ & DƯỢC PHẨM =====
    "DHG": "Y tế & Dược phẩm", "IMP": "Y tế & Dược phẩm",
    "DMC": "Y tế & Dược phẩm", "TRA": "Y tế & Dược phẩm",
    "DBD": "Y tế & Dược phẩm", "DCL": "Y tế & Dược phẩm",

    # ===== HÓA CHẤT =====
    "DPM": "Hóa chất", "DCM": "Hóa chất", "DGC": "Hóa chất",
    "CSV": "Hóa chất", "BFC": "Hóa chất", "LAS": "Hóa chất",
    "DDV": "Hóa chất", "NFC": "Hóa chất", "AAA": "Hóa chất",
    "BMP": "Hóa chất", "NTP": "Hóa chất",

    # ===== XÂY DỰNG & VẬT LIỆU =====
    "ACV": "Xây dựng & Vật liệu", "CTD": "Xây dựng & Vật liệu",
    "VCG": "Xây dựng & Vật liệu", "C4G": "Xây dựng & Vật liệu",
    "FCN": "Xây dựng & Vật liệu", "HHV": "Xây dựng & Vật liệu",
    "LCG": "Xây dựng & Vật liệu", "CII": "Xây dựng & Vật liệu",
    "PC1": "Xây dựng & Vật liệu", "VC1": "Xây dựng & Vật liệu",

    # ===== DU LỊCH & GIẢI TRÍ =====
    "VJC": "Du lịch & Giải trí", "HVN": "Du lịch & Giải trí",
    "AST": "Du lịch & Giải trí", "DAH": "Du lịch & Giải trí",
    "SKG": "Du lịch & Giải trí",
}


# ============================================================
# CÁC HÀM XỬ LÝ
# ============================================================
def get_sector(symbol):
    """Lấy tên ngành của một mã cổ phiếu."""
    return SYMBOL_SECTOR_MAP.get(symbol, "Khác")


def sector_score(sector_name):
    """
    Chấm điểm ngành (0-100).
    
    Thành phần:
    - Tăng trưởng Q2 (30 điểm)
    - Dự báo cả năm (25 điểm)
    - Biên lợi nhuận (20 điểm)
    - Đa dạng hóa (15 điểm)
    - Rủi ro (10 điểm)
    """
    data = SECTOR_DATA.get(sector_name)
    if not data:
        return {
            "sector": sector_name,
            "total": 0,
            "error": "Không có dữ liệu ngành",
        }

    # 1. Tăng trưởng Q2 (30 điểm)
    growth = min(30, max(0, data["q2_growth"] / 257 * 30))

    # 2. Dự báo cả năm (25 điểm)
    fy = min(25, max(0, data["fy_growth"] / 180 * 25))

    # 3. Biên LN (20 điểm)
    margin = min(20, max(0, data["margin"] / 35 * 20))

    # 4. Đa dạng hóa (15 điểm)
    conc = {"low": 15, "medium": 10, "high": 5}.get(
        data["concentration"], 5)

    # 5. Rủi ro (10 điểm)
    risk_text = data.get("risk", "")
    if "cực mạnh" in risk_text or "chu kỳ" in risk_text:
        risk = 3
    elif "áp lực" in risk_text or "phụ thuộc" in risk_text:
        risk = 6
    else:
        risk = 9

    total = growth + fy + margin + conc + risk

    return {
        "sector": sector_name,
        "growth": round(growth, 1),
        "forecast": round(fy, 1),
        "margin": round(margin, 1),
        "diversification": conc,
        "risk": risk,
        "total": round(total, 1),
        "q2_growth": data["q2_growth"],
        "fy_growth": data["fy_growth"],
        "risk_note": data["risk"],
    }


def rank_sectors():
    """Xếp hạng tất cả ngành theo điểm."""
    results = [sector_score(s) for s in SECTOR_DATA.keys()]
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def get_top_sectors(n=None):
    """Lấy top N ngành tốt nhất."""
    if n is None:
        n = config.TOP_SECTORS
    return [s["sector"] for s in rank_sectors()[:n]]