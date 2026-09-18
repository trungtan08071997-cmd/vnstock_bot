"""
Phân tích sentiment tin tức tiếng Việt - NÂNG CẤP
5 sub-indicators:
1. Sentiment Polarity (40%)
2. Mention Frequency (20%)
3. Source Diversity (15%)
4. Recency (15%)
5. Sentiment Consistency (10%)
"""
import re
from datetime import datetime, timedelta, timezone
from collections import defaultdict


# ===== MÚI GIỜ VN =====
VN_TZ = timezone(timedelta(hours=7))


# ===== TỪ ĐIỂN SENTIMENT =====
POSITIVE_WORDS = {
    "tăng mạnh": 2.0, "bùng nổ": 2.0, "vượt đỉnh": 2.0,
    "lợi nhuận": 1.0, "tăng trưởng": 1.0, "khởi sắc": 1.5,
    "mua ròng": 1.5, "tích cực": 1.0, "khả quan": 1.0,
    "đột phá": 1.5, "bứt phá": 1.5, "hồi phục": 1.0,
    "tăng": 0.5, "lãi": 0.8, "cổ tức": 0.5, "mở rộng": 0.5,
    "ký kết": 0.5, "hợp tác": 0.5, "kỷ lục": 1.0,
    "vượt kế hoạch": 1.5, "tăng vốn": 0.5, "khởi công": 0.5,
    "trúng thầu": 1.0, "thắng thầu": 1.0, "tăng cổ tức": 1.5,
}

NEGATIVE_WORDS = {
    "giảm sâu": 2.0, "bán tháo": 2.0, "lao dốc": 2.0,
    "nợ xấu": 1.5, "rút vốn": 1.5, "thua lỗ": 1.5,
    "lỗ": 1.0, "giảm": 0.5, "tiêu cực": 1.0,
    "sụt giảm": 1.5, "khó khăn": 1.0, "rủi ro": 0.8,
    "điều chỉnh": 0.5, "bán ròng": 1.0, "phá sản": 2.0,
    "vi phạm": 1.5, "xử phạt": 1.5, "chậm tiến độ": 1.0,
    "hủy niêm yết": 2.0, "kiểm toán ngoại trừ": 2.0,
    "đình chỉ": 1.5, "cảnh báo": 1.0, "xuống thấp nhất": 1.5,
}


# ===== TRỌNG SỐ SUB-INDICATORS =====
SENTIMENT_WEIGHTS = {
    "polarity": 0.50,      # 40% - độ tích cực/tiêu cực
    "frequency": 0.15,     # 20% - số lần nhắc
    "diversity": 0.15,     # 15% - số nguồn khác nhau
    "recency": 0.10,       # 15% - độ mới
    "consistency": 0.10,   # 10% - đồng thuận
}


# ===== TRUST WEIGHT NGUỒN =====
SOURCE_TRUST = {
    "cafef": 1.0, "cafef_stock": 1.0,
    "vietstock": 1.0, "ndh": 1.0,
    "vnexpress_business": 0.9, "vnexpress_stock": 0.9,
    "vnexpress": 0.9,
    "thanhnien_business": 0.8, "tuoitre_business": 0.8,
    "vtv_market": 0.9,
}


# ============================================================
# HÀM CƠ BẢN
# ============================================================
def analyze_text(text):
    """Phân tích sentiment của text, trả về [-1, 1]."""
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(t.count(w) * v for w, v in POSITIVE_WORDS.items())
    neg = sum(t.count(w) * v for w, v in NEGATIVE_WORDS.items())
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _hours_ago(published_dt):
    """Tính số giờ từ lúc đăng đến bây giờ."""
    if published_dt is None:
        return 72  # Mặc định coi như cũ
    if isinstance(published_dt, str):
        return 72
    now = datetime.now(VN_TZ)
    if published_dt.tzinfo is None:
        published_dt = published_dt.replace(tzinfo=VN_TZ)
    delta = now - published_dt
    return max(0, delta.total_seconds() / 3600)


def _recency_weight(hours):
    """
    Trọng số theo độ mới (dùng cho weighted average).
    Tin mới → trọng số cao.
    """
    if hours <= 6: return 3.0
    elif hours <= 24: return 2.0
    elif hours <= 48: return 1.5
    elif hours <= 72: return 1.0
    else: return 0.5


# ============================================================
# 5 SUB-INDICATORS
# ============================================================
def _score_polarity(articles_with_sentiment):
    """
    Sub-indicator 1: Sentiment Polarity (40 điểm)
    Weighted average theo thời gian + trust nguồn.
    """
    if not articles_with_sentiment:
        return 0, {"value": 0, "status": "no_data"}

    weighted_sum = 0
    weight_total = 0

    for art in articles_with_sentiment:
        sent = art.get("sentiment", 0)
        hours = _hours_ago(art.get("published_dt"))
        source = art.get("source", "")
        trust = SOURCE_TRUST.get(source, 0.7)

        weight = _recency_weight(hours) * trust
        weighted_sum += sent * weight
        weight_total += weight

    if weight_total == 0:
        return 0, {"value": 0, "status": "no_data"}

    weighted_avg = weighted_sum / weight_total

    # Chuyển [-1, 1] → điểm [0-40]
    if weighted_avg >= 0.5:
        score = 40
        status = "RẤT TÍCH CỰC"
    elif weighted_avg >= 0.2:
        score = 32
        status = "TÍCH CỰC"
    elif weighted_avg >= -0.1:
        score = 20
        status = "TRUNG TÍNH"
    elif weighted_avg >= -0.5:
        score = 10
        status = "TIÊU CỰC"
    else:
        score = 0
        status = "RẤT TIÊU CỰC"

    return score, {
        "value": round(weighted_avg, 3),
        "status": status,
    }


def _score_frequency(mentions):
    """
    Sub-indicator 2: Mention Frequency (20 điểm)
    Số lần được nhắc trên tin tức.
    """
    if mentions >= 20: return 20, "RẤT NHIỀU"
    elif mentions >= 15: return 18, "NHIỀU"
    elif mentions >= 10: return 15, "KHÁ NHIỀU"
    elif mentions >= 5: return 12, "TRUNG BÌNH"
    elif mentions >= 3: return 8, "ÍT"
    elif mentions >= 1: return 4, "RẤT ÍT"
    else: return 0, "KHÔNG"


def _score_diversity(sources_set):
    """
    Sub-indicator 3: Source Diversity (15 điểm)
    Số nguồn khác nhau đưa tin.
    """
    n = len(sources_set)
    if n >= 6: return 15, "RẤT ĐA DẠNG"
    elif n >= 4: return 12, "ĐA DẠNG"
    elif n >= 3: return 10, "KHÁ ĐA DẠNG"
    elif n >= 2: return 6, "ÍT"
    else: return 2, "1 NGUỒN"


def _score_recency(articles_with_sentiment):
    """
    Sub-indicator 4: Recency (15 điểm)
    Độ mới của tin tức.
    """
    if not articles_with_sentiment:
        return 0, "KHÔNG CÓ TIN"

    hours_list = [_hours_ago(a.get("published_dt"))
                  for a in articles_with_sentiment]
    avg_hours = sum(hours_list) / len(hours_list)

    if avg_hours <= 6:
        return 15, "CỰC MỚI"
    elif avg_hours <= 24:
        return 13, "TRONG NGÀY"
    elif avg_hours <= 48:
        return 10, "HÔM QUA"
    elif avg_hours <= 72:
        return 6, "2-3 NGÀY"
    else:
        return 3, "CŨ"


def _score_consistency(articles_with_sentiment):
    """
    Sub-indicator 5: Consistency (10 điểm)
    Mức độ đồng thuận giữa các bài.
    """
    if not articles_with_sentiment:
        return 5, "KHÔNG CÓ DỮ LIỆU"

    pos = sum(1 for a in articles_with_sentiment
              if a.get("sentiment", 0) > 0.1)
    neg = sum(1 for a in articles_with_sentiment
              if a.get("sentiment", 0) < -0.1)
    total = pos + neg

    if total == 0:
        return 5, "TRUNG TÍNH"  # Không bài nào có sentiment rõ

    ratio = max(pos, neg) / total

    if ratio >= 0.9:
        return 10, "ĐỒNG THUẬN CAO"
    elif ratio >= 0.75:
        return 8, "ĐỒNG THUẬN"
    elif ratio >= 0.6:
        return 6, "HƠI ĐỒNG THUẬN"
    else:
        return 3, "CHIA ĐÔI"


# ============================================================
# HÀM CHÍNH - CHẤM ĐIỂM SENTIMENT TỔNG HỢP
# ============================================================
def sentiment_score_full(symbol, articles):
    """
    Chấm điểm sentiment tổng hợp (0-100) với 5 sub-indicators.
    
    Args:
        symbol: Mã cổ phiếu
        articles: List tất cả bài viết (chứa 'title', 'summary', 
                  'source', 'published_dt', 'published_str')
    
    Returns:
        (score, details_dict)
    """
    # Lọc bài viết chứa mã
    symbol_articles = []
    pattern = re.compile(rf"\b{symbol}\b", re.IGNORECASE)

    for art in articles:
        text = f"{art.get('title', '')} {art.get('summary', '')}"
        if pattern.search(text):
            sent = analyze_text(text)
            symbol_articles.append({
                "title": art.get("title", ""),
                "source": art.get("source", ""),
                "published_dt": art.get("published_dt"),
                "published_str": art.get("published_str", ""),
                "sentiment": sent,
                "url": art.get("url", ""),
            })

    if not symbol_articles:
        # Không có tin → trả về trung tính
        return 50, {
            "mentions": 0,
            "sources": 0,
            "sub_scores": {},
            "total": 50,
            "status": "KHÔNG CÓ TIN",
        }

    # ===== CHẤM TỪNG SUB-INDICATOR =====
    polarity_score, polarity_detail = _score_polarity(symbol_articles)
    frequency_score, freq_status = _score_frequency(len(symbol_articles))

    sources_set = set(a["source"] for a in symbol_articles)
    diversity_score, div_status = _score_diversity(sources_set)

    recency_score, rec_status = _score_recency(symbol_articles)
    consistency_score, cons_status = _score_consistency(symbol_articles)

    # ===== TÍNH ĐIỂM TỔNG HỢP =====
    total = (
        polarity_score * SENTIMENT_WEIGHTS["polarity"]
        + frequency_score * SENTIMENT_WEIGHTS["frequency"]
        + diversity_score * SENTIMENT_WEIGHTS["diversity"]
        + recency_score * SENTIMENT_WEIGHTS["recency"]
        + consistency_score * SENTIMENT_WEIGHTS["consistency"]
    )

    # ===== XÁC ĐỊNH TRẠNG THÁI =====
    if total >= 80:
        status = "🟢 RẤT TÍCH CỰC"
    elif total >= 60:
        status = "🟢 TÍCH CỰC"
    elif total >= 40:
        status = "⚪ TRUNG TÍNH"
    elif total >= 20:
        status = "🔴 TIÊU CỰC"
    else:
        status = "🔴 RẤT TIÊU CỰC"

    details = {
        "mentions": len(symbol_articles),
        "sources": len(sources_set),
        "sub_scores": {
            "polarity": {
                "score": polarity_score,
                "max": 40,
                "weight": SENTIMENT_WEIGHTS["polarity"],
                **polarity_detail,
            },
            "frequency": {
                "score": frequency_score,
                "max": 20,
                "weight": SENTIMENT_WEIGHTS["frequency"],
                "mentions": len(symbol_articles),
                "status": freq_status,
            },
            "diversity": {
                "score": diversity_score,
                "max": 15,
                "weight": SENTIMENT_WEIGHTS["diversity"],
                "n_sources": len(sources_set),
                "status": div_status,
            },
            "recency": {
                "score": recency_score,
                "max": 15,
                "weight": SENTIMENT_WEIGHTS["recency"],
                "status": rec_status,
            },
            "consistency": {
                "score": consistency_score,
                "max": 10,
                "weight": SENTIMENT_WEIGHTS["consistency"],
                "status": cons_status,
            },
        },
        "total": round(total, 1),
        "status": status,
        "top_articles": sorted(
            symbol_articles,
            key=lambda a: abs(a["sentiment"]),
            reverse=True,
        )[:3],
    }

    return round(total, 1), details


# ============================================================
# TEST NHANH
# ============================================================
if __name__ == "__main__":
    # Test đơn giản
    text1 = "Cổ phiếu VCB tăng mạnh, lợi nhuận kỷ lục"
    text2 = "HPG giảm sâu, bán tháo mạnh"
    text3 = "Thị trường đi ngang, thanh khoản thấp"

    print("VCB:", analyze_text(text1))
    print("HPG:", analyze_text(text2))
    print("Neutral:", analyze_text(text3))