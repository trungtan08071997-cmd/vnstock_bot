"""
Gửi báo cáo qua Console + Discord (100% miễn phí)
"""
import re
import requests
import config


def send_console(message):
    """In ra terminal với màu sắc."""
    plain = re.sub(r"<[^>]+>", "", message)
    BOLD = "\033[1m"; RESET = "\033[0m"; CYAN = "\033[96m"
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}📊 BÁO CÁO THỊ TRƯỜNG CHỨNG KHOÁN{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(plain)
    print(f"{CYAN}{'='*60}{RESET}")


def send_discord(message):
    """
    Gửi báo cáo qua Discord Webhook.
    - Tự động chia nhỏ nếu message > 2000 ký tự (giới hạn Discord)
    - Chuyển HTML tags của Telegram sang Markdown Discord
    - Trả về True/False
    """
    url = config.DISCORD_WEBHOOK_URL
    if not url or "PASTE_URL" in url:
        print("⚠️ Chưa cấu hình DISCORD_WEBHOOK_URL trong config.py")
        return False

    # Chuyển đổi HTML → Markdown Discord
    md = message
    md = re.sub(r"<b>(.*?)</b>", r"**\1**", md)
    md = re.sub(r"<i>(.*?)</i>", r"*\1*", md)
    md = re.sub(r"<code>(.*?)</code>", r"`\1`", md)
    md = re.sub(r"<[^>]+>", "", md)  # Xóa tag còn lại
    md = md.replace("─" * 40, "─" * 30)

    # Discord giới hạn 2000 ký tự/tin nhắn
    MAX_LEN = 1900
    chunks = [md[i:i+MAX_LEN] for i in range(0, len(md), MAX_LEN)]

    ok = True
    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "content": chunk,
            "username": "Stock Bot VN",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933177.png",
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code in (200, 204):
                print(f"📨 Discord: gửi phần {idx}/{len(chunks)} OK")
            else:
                print(f"⚠️ Discord lỗi {r.status_code}: {r.text[:200]}")
                ok = False
        except Exception as e:
            print(f"⚠️ Lỗi gửi Discord: {e}")
            ok = False

    return ok


def notify_all(message):
    """Gửi qua tất cả kênh đã cấu hình."""
    channels = config.NOTIFY_CHANNELS
    results = {}

    if "console" in channels or not channels:
        send_console(message)
        results["console"] = True

    if "discord" in channels:
        results["discord"] = send_discord(message)

    return results