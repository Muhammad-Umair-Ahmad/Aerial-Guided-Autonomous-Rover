"""
AGRA Mission Control - Desktop Launcher

Just run:  python run_dashboard.py
Dashboard opens automatically. iPhone URL printed below.
"""

import os
import sys
import socket
import threading
import time
import subprocess
import shutil


def get_local_ip():
    """Auto-detect the laptop's local WiFi IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_browser():
    """Find Edge or Chrome for app-mode (clean window, no address bar)."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in edge_paths:
        if os.path.exists(p):
            return p

    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in chrome_paths:
        if os.path.exists(p):
            return p

    for name in ["msedge", "chrome", "google-chrome"]:
        found = shutil.which(name)
        if found:
            return found

    return None


def start_server():
    """Start the HTTPS signaling server."""
    import uvicorn

    script_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.path.join(script_dir, "key.pem"),
        ssl_certfile=os.path.join(script_dir, "cert.pem"),
        log_level="warning",
    )


def main():
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Change to script directory so imports work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    # Detect IP
    ip = get_local_ip()
    iphone_url = f"https://{ip}:8000/broadcaster"

    # Print banner
    print()
    print("  ============================================================")
    print("         AGRA MISSION CONTROL -- STARTING UP                   ")
    print("  ============================================================")
    print()
    print(f"  [*] Laptop IP Address:  {ip}")
    print(f"  [*] Dashboard URL:      https://localhost:8000")
    print()
    print("  ------------------------------------------------------------")
    print("   HOW TO CONNECT YOUR iPHONE:")
    print("  ------------------------------------------------------------")
    print()
    print("   1. Connect iPhone to the SAME WiFi as this laptop")
    print()
    print("   2. Open Safari on your iPhone")
    print()
    print("   3. Type this URL in Safari:")
    print()
    print(f"      >>> {iphone_url}")
    print()
    print("   4. Safari will show a security warning:")
    print("      - Tap 'Show Details'")
    print("      - Tap 'visit this website'")
    print("      - Tap 'Visit Website' to confirm")
    print()
    print("   5. Tap the 'Start Broadcast' button")
    print()
    print("   Done! Feed will appear on the dashboard.")
    print("  ------------------------------------------------------------")
    print()

    # Start HTTPS server in background
    print("  [~] Starting server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    print("  [+] Server running on https://0.0.0.0:8000")
    print()

    # Open dashboard in app mode (clean window, no address bar/tabs)
    browser = find_browser()
    dashboard_url = "https://localhost:8000"

    if browser:
        print("  [~] Opening dashboard (app mode - no address bar)...")
        subprocess.Popen(
            [
                browser,
                f"--app={dashboard_url}",
                "--ignore-certificate-errors",
                "--disable-features=TranslateUI",
                "--window-size=1280,720",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("  [~] Opening dashboard in browser...")
        import webbrowser
        webbrowser.open(dashboard_url)

    print("  [+] Dashboard is open!")
    print()
    print("  Press Ctrl+C to shut down.")
    print()

    # Keep the script alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  [!] Shutting down AGRA Mission Control...")
        sys.exit(0)


if __name__ == "__main__":
    main()
