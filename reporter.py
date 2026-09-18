"""
Tạo báo cáo HTML cho các kênh thông báo
"""
import json
import os
from datetime import datetime
import config
from market_regime import get_market_regime
from sector_analyzer import rank_sectors
from stock_scorer import format_stock_line


def build_report(stocks, hot_ranking=None, mode="vn30"):
    """
    Tạo báo cáo dạng text (HTML cho Telegram/Discord).
    
    Args:
        stocks: List[dict] từ screen_stocks()
        hot_ranking: List[dict] từ analyze_news_for_symbols()
        mode: "vn30" | "news" | "both"
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    mode_title = {
        "vn30": "VN30",
        "news": "TIN TỨC",
        "both": "VN30 + TIN TỨC",
    }.get(mode, "VN30")

    lines = []
    lines.append("╔══════════════════════════════════════╗")
    lines.append(f"║   BÁO CÁO {mode_title} - {now}   ║")
    lines.append("╚══════════════════════════════════════╝")
    lines.append("")

    # ===== 1. THỊ TRƯỜNG =====
    market = get_market_regime()
    ce = "📈" if market["change"] >= 0 else "📉"
    lines.append(f"📊 <b>VN-INDEX:</b> {market['vnindex']:,.2f} "
                 f"({ce} {market['change']:+.2f}%)")
    lines.append(f"🎯 <b>Chế độ:</b> {market['regime']} (RSI: {market['rsi']})")
    lines.append(f"💡 <b>Khuyến nghị:</b> {market['recommendation']}")
    lines.append("")

    # ===== 2. HOT SYMBOLS (chỉ hiển thị khi mode news/both) =====
    if hot_ranking and mode in ("news", "both"):
        lines.append("🔥 <b>MÃ ĐƯỢC NHẮC NHIỀU NHẤT TRÊN TIN TỨC:</b>")
        lines.append("")
        for i, r in enumerate(hot_ranking[:10], 1):
            ic = "🟢" if r["avg_sentiment"] > 0.1 else (
                "🔴" if r["avg_sentiment"] < -0.1 else "⚪")
            lines.append(
                f"{i:2d}. <b>{r['symbol']}</b> — {r['mentions']} mentions"
                f" / {r['sources']} nguồn — {ic} {r['avg_sentiment']:+.2f}"
                f" — hot {r['hot_score']:.0f}/100"
            )
            # 2 tin mới nhất kèm thời gian
            for t in r.get("sample_titles", [])[:2]:
                if isinstance(t, dict):
                    time_str = t.get("time", "")
                    title_str = t.get("title", "")
                    source_str = t.get("source", "")
                else:
                    time_str = ""
                    title_str = str(t)
                    source_str = ""
                
                prefix = f"[{time_str}]" if time_str else ""
                lines.append(f"     <i>{prefix} {title_str}</i>")
                if source_str:
                    lines.append(f"       └ nguồn: {source_str}")
        lines.append("")

    # ===== 3. XẾP HẠNG NGÀNH =====
    lines.append("🏭 <b>XẾP HẠNG NGÀNH ƯU TIÊN:</b>")
    sectors = rank_sectors()
    for i, s in enumerate(sectors[:6], 1):
        lines.append(
            f"{i}. <b>{s['sector']}</b> — {s['total']:.0f}/100 "
            f"(Q2: {s.get('q2_growth', 0):+.1f}%, "
            f"năm: {s.get('fy_growth', 0):+.0f}%)"
        )
    lines.append("")

    # ===== 4. TOP CỔ PHIẾU =====
    lines.append(f"🏆 <b>TOP CỔ PHIẾU KHUYẾN NGHỊ ({mode_title}):</b>")
    lines.append("")
    if not stocks:
        lines.append("❌ Không có cổ phiếu nào đạt tiêu chí.")
    else:
        for i, st in enumerate(stocks, 1):
            lines.append(f"<b>{i}.</b> {format_stock_line(st)}")
            lines.append("")

    # ===== 5. DISCLAIMER =====
    lines.append("─" * 40)
    lines.append("⚠️ <i>Báo cáo chỉ mang tính tham khảo, không phải "
                 "lời khuyên đầu tư. Nhà đầu tư tự chịu trách nhiệm.</i>")

    return "\n".join(lines)


def save_report_json(stocks, hot_ranking=None, filename="report.json",
                     mode="vn30"):
    """Lưu kết quả dạng JSON."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(config.OUTPUT_DIR, filename)

    data = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "market": get_market_regime(),
        "sectors": rank_sectors(),
        "hot_symbols": hot_ranking or [],
        "stocks": stocks,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"💾 Đã lưu: {filepath}")
    return filepath