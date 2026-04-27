#!/usr/bin/env python3
# File: monitor_server.py
"""XENON Agent Visual Monitor — 实时画面监控（后端主动推送）。

架构（参考 Optimus-3 的 gui_server.py 设计）:

  env.step (wrapper.py) ──POST JPEG──► monitor_server ──MJPEG stream──► browser

端点:
  GET  /                前端 HTML（<img src="/stream.mjpg">）
  GET  /stream.mjpg     MJPEG multipart 流（浏览器原生支持）
  POST /push            后端推送一帧 JPEG（Content-Type: image/jpeg）
  GET  /latest.jpg      最新一帧（兜底：推送 / 磁盘轮询）
  GET  /status          状态 JSON（任务名、帧源、延迟）
  GET  /healthz         探活

启动:
  nohup python3 monitor_server.py --port 8080 > /tmp/xenon_monitor.log 2>&1 &

零依赖：仅 Python 3 标准库。
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 全局状态：最新推送帧 ──
_frame_lock = threading.Lock()
_latest_pushed_frame = None  # type: bytes | None   最新推送的 JPEG 二进制
_latest_pushed_time = 0.0    # 最后一次收到 push 的时间戳
_latest_pushed_count = 0     # 累计收到帧数
_stream_clients = 0          # 当前连接的 MJPEG 流客户端数

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XENON · Minecraft Agent Monitor</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161a22; --fg: #e6e8eb;
    --muted: #8a94a6; --accent: #57b6ff; --good: #69d07a;
    --warn: #fbbf24; --bad: #ef5350;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0; min-height: 100vh;
    background: var(--bg); color: var(--fg);
    font-family: ui-sans-serif, -apple-system, "Segoe UI",
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }
  header {
    width: 100%; padding: 10px 16px;
    display: flex; justify-content: space-between; align-items: center;
    background: var(--panel); border-bottom: 1px solid #222a36;
  }
  h1 { margin: 0; font-size: 15px; color: var(--accent); font-weight: 600; }
  .stat { display: flex; gap: 8px; font-size: 12px; color: var(--muted); }
  .stat span {
    padding: 3px 8px; background: #0b0e14; border: 1px solid #222a36;
    border-radius: 4px; font-variant-numeric: tabular-nums;
  }
  #main {
    flex: 1; width: 100%;
    display: flex; flex-direction: column; align-items: center;
    padding: 16px;
  }
  #frame-wrap {
    position: relative; background: #000;
    border: 1px solid #222a36; border-radius: 6px;
    overflow: hidden; line-height: 0;
    box-shadow: 0 0 30px rgba(87, 182, 255, 0.08);
  }
  #frame {
    display: block;
    width: min(80vw, 1024px);
    aspect-ratio: 16 / 9;
    object-fit: contain;
    image-rendering: pixelated;
    background: #000;
  }
  #overlay {
    position: absolute; left: 10px; top: 10px;
    background: rgba(0, 0, 0, 0.65);
    padding: 6px 10px; border-radius: 4px;
    font-size: 12px; color: #9fe89f; max-width: 70%;
    pointer-events: none;
  }
  #overlay.muted { color: var(--muted); }
  footer {
    color: var(--muted); font-size: 12px;
    padding: 8px 16px 16px; text-align: center;
  }
  code { color: #b4b4ff; }
  .dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 4px; vertical-align: 1px;
  }
</style>
</head>
<body>
<header>
  <h1>🟢 XENON · Minecraft Agent Monitor</h1>
  <div class="stat">
    <span id="status"><span class="dot" style="background:#888"></span>初始化...</span>
    <span id="src">—</span>
    <span id="age">—</span>
    <span id="fps">— fps</span>
  </div>
</header>
<div id="main">
  <div id="frame-wrap">
    <!-- MJPEG 流：浏览器原生解码，后端推送即时更新，无需 JS 轮询 -->
    <img id="frame" src="/stream.mjpg" alt="等待画面...">
    <div id="overlay" class="muted">等待 agent 启动...</div>
  </div>
</div>
<footer>
  后端主动推送 · <code>POST /push</code>（JPEG） → <code>GET /stream.mjpg</code>（MJPEG）
</footer>
<script>
(() => {
  const statusEl = document.getElementById('status');
  const srcEl = document.getElementById('src');
  const ageEl = document.getElementById('age');
  const fpsEl = document.getElementById('fps');
  const overlay = document.getElementById('overlay');
  const img = document.getElementById('frame');

  function setStatus(color, text) {
    statusEl.innerHTML =
      '<span class="dot" style="background:' + color + '"></span>' + text;
  }

  // 仅轮询 /status（轻量 JSON），画面本身由 MJPEG 流驱动
  async function pollStatus() {
    try {
      const r = await fetch('/status', { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const s = await r.json();

      srcEl.textContent = s.source || '—';
      fpsEl.textContent = (s.push_fps || 0).toFixed(1) + ' fps';

      const age = s.age_sec || 0;
      ageEl.textContent = age > 0 ? age.toFixed(1) + 's' : '—';

      if (s.source === 'push' && age < 3) {
        setStatus('#69d07a', '● 实时推送中');
      } else if (s.source === 'push') {
        setStatus('#fbbf24', '● 推送停顿 (' + age.toFixed(0) + 's)');
      } else if (s.source === 'disk') {
        setStatus('#57b6ff', '● 磁盘兜底 (' + age.toFixed(0) + 's)');
      } else {
        setStatus('#8a94a6', '○ 无画面');
      }

      if (s.task) {
        overlay.className = '';
        overlay.textContent = '任务: ' + s.task + (s.uuid ? ' · ' + s.uuid : '');
        overlay.style.display = 'block';
      } else {
        overlay.className = 'muted';
        overlay.textContent = '等待 agent 启动，画面会自动出现...';
        overlay.style.display = s.source === 'push' ? 'none' : 'block';
      }
    } catch (e) {
      setStatus('#ef5350', '✗ 监控服务异常');
    }
  }

  // MJPEG 流错误时自动重连（浏览器有时会断开）
  img.onerror = () => {
    setTimeout(() => { img.src = '/stream.mjpg?' + Date.now(); }, 1000);
  };

  setInterval(pollStatus, 1000);
  pollStatus();
})();
</script>
</body>
</html>
"""


def find_latest_image_on_disk(root):
    """兜底：找磁盘上最新的 imgs/*.jpg（没有推送时使用）。"""
    patterns = [
        os.path.join(root, "logs", "eval", "*", "*", "*", "imgs", "*.jpg"),
        os.path.join(root, "logs", "eval", "*", "*", "imgs", "*.jpg"),
    ]
    for p in patterns:
        candidates = list(glob.iglob(p))
        if candidates:
            try:
                return max(candidates, key=os.path.getmtime)
            except OSError:
                pass
    return None


def find_latest_task_info(root):
    """从最新的 hydra log 解析当前任务名和 run_uuid。"""
    logs = glob.glob(os.path.join(root, "logs", "eval", "*", "*", "main_*.log"))
    if not logs:
        return None, None
    try:
        latest = max(logs, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return None, None

    task = None
    m = re.search(r"Running Tasks:\s*\[['\"]([^'\"]+)['\"]", content)
    if m:
        task = m.group(1)

    uuid = None
    m = re.search(r"run_uuid:\s*(\S+)", content)
    if m:
        uuid = m.group(1)[:6]
    return task, uuid


# ── push_fps 估计（滑动窗口）──
_push_timestamps = []
_push_fps_lock = threading.Lock()


def _record_push():
    """记录一次 push 时间，用于估算 fps。"""
    now = time.time()
    with _push_fps_lock:
        _push_timestamps.append(now)
        # 保留最近 3 秒
        cutoff = now - 3.0
        while _push_timestamps and _push_timestamps[0] < cutoff:
            _push_timestamps.pop(0)


def _current_push_fps():
    with _push_fps_lock:
        n = len(_push_timestamps)
        if n < 2:
            return 0.0
        span = _push_timestamps[-1] - _push_timestamps[0]
        return (n - 1) / span if span > 0 else 0.0


class MonitorHandler(BaseHTTPRequestHandler):
    server_root = DEFAULT_ROOT
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return  # 静默

    # ── 通用辅助 ──
    def _send(self, code, body, ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── POST /push: 接收一帧 JPEG ──
    def do_POST(self):
        global _latest_pushed_frame, _latest_pushed_time, _latest_pushed_count
        path = urlparse(self.path).path
        if path != "/push":
            self._send(404, b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:  # 限制 20MB
                self._send(400, b"bad length")
                return
            data = self.rfile.read(length)
            # 校验 JPEG magic bytes
            if not (len(data) > 4 and data[:2] == b"\xff\xd8"):
                self._send(400, b"not a jpeg")
                return
            with _frame_lock:
                _latest_pushed_frame = data
                _latest_pushed_time = time.time()
                _latest_pushed_count += 1
            _record_push()
            self._send(200, b"ok")
        except Exception as e:
            self._send(500, ("push error: " + repr(e)).encode("utf-8"))

    # ── GET 分发 ──
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"),
                           "text/html; charset=utf-8")
            elif path == "/stream.mjpg":
                self._serve_mjpeg_stream()
            elif path == "/latest.jpg":
                self._serve_latest_jpeg()
            elif path == "/status":
                self._serve_status()
            elif path == "/healthz":
                self._send(200, b"ok")
            else:
                self._send(404, b"not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self._send(500, ("error: " + repr(e)).encode("utf-8"))
            except Exception:
                pass

    # ── MJPEG 流 ──
    def _serve_mjpeg_stream(self):
        global _stream_clients
        boundary = "xenonframe"
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.end_headers()
        _stream_clients += 1

        last_sent_time = 0.0
        # 每隔最多 50ms 检查一次新帧；有新帧立即发
        TICK = 0.05
        # 如果长时间没有新帧，fallback 从磁盘读
        disk_fallback_after = 2.0  # 推送停顿 > 2s 时回退到磁盘
        last_disk_mtime = 0.0

        try:
            while True:
                # 取最新帧
                jpeg_bytes = None
                with _frame_lock:
                    frame = _latest_pushed_frame
                    frame_t = _latest_pushed_time

                if frame is not None and frame_t > last_sent_time:
                    jpeg_bytes = frame
                    last_sent_time = frame_t
                elif frame is None or (time.time() - frame_t) > disk_fallback_after:
                    # 磁盘兜底
                    path = find_latest_image_on_disk(MonitorHandler.server_root)
                    if path and os.path.isfile(path):
                        try:
                            mtime = os.path.getmtime(path)
                            if mtime > last_disk_mtime:
                                with open(path, "rb") as f:
                                    jpeg_bytes = f.read()
                                last_disk_mtime = mtime
                                last_sent_time = mtime
                        except OSError:
                            pass

                if jpeg_bytes:
                    try:
                        self.wfile.write(
                            f"--{boundary}\r\n".encode("ascii"))
                        self.wfile.write(
                            b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                            .encode("ascii"))
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break

                time.sleep(TICK)
        finally:
            _stream_clients = max(0, _stream_clients - 1)

    # ── 单帧 JPEG（兜底接口） ──
    def _serve_latest_jpeg(self):
        with _frame_lock:
            frame = _latest_pushed_frame
            frame_t = _latest_pushed_time
        if frame is not None and (time.time() - frame_t) < 60.0:
            self._send(200, frame, "image/jpeg")
            return
        path = find_latest_image_on_disk(MonitorHandler.server_root)
        if path and os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                self._send(200, data, "image/jpeg")
                return
            except OSError:
                pass
        # 204 No Content
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    # ── 状态 JSON ──
    def _serve_status(self):
        task, uuid = find_latest_task_info(MonitorHandler.server_root)
        now = time.time()
        with _frame_lock:
            has_push = _latest_pushed_frame is not None
            push_t = _latest_pushed_time
            push_count = _latest_pushed_count

        disk_path = find_latest_image_on_disk(MonitorHandler.server_root)
        disk_mtime = 0.0
        if disk_path:
            try:
                disk_mtime = os.path.getmtime(disk_path)
            except OSError:
                pass

        push_age = (now - push_t) if has_push and push_t > 0 else 1e9
        disk_age = (now - disk_mtime) if disk_mtime > 0 else 1e9

        # 选择最新来源
        if has_push and push_age < 60.0:
            source = "push"
            age = push_age
            basename = f"pushed_frame_{push_count}"
        elif disk_mtime > 0:
            source = "disk"
            age = disk_age
            basename = os.path.basename(disk_path) if disk_path else ""
        else:
            source = "none"
            age = 0
            basename = ""

        resp = {
            "source": source,
            "age_sec": float(age) if age < 1e8 else 0.0,
            "basename": basename,
            "push_count": push_count,
            "push_fps": round(_current_push_fps(), 2),
            "stream_clients": _stream_clients,
            "task": task or "",
            "uuid": uuid or "",
        }
        self._send(200, json.dumps(resp).encode("utf-8"),
                   "application/json")


def main():
    p = argparse.ArgumentParser(description="XENON Agent Visual Monitor")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="XENON 项目根目录（含 logs/、exp_results/）")
    args = p.parse_args()

    MonitorHandler.server_root = os.path.abspath(args.root)
    if not os.path.isdir(MonitorHandler.server_root):
        print(f"[!] root directory {MonitorHandler.server_root} does not exist",
              file=sys.stderr)
    else:
        print(f"[+] watching: {MonitorHandler.server_root}")

    try:
        srv = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    except OSError as e:
        print(f"[!] cannot bind {args.host}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] monitor running at http://{args.host}:{args.port}/")
    print(f"[+] push endpoint:  POST http://{args.host}:{args.port}/push")
    print(f"[+] MJPEG stream:   GET  http://{args.host}:{args.port}/stream.mjpg")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] stopped")
        srv.server_close()


if __name__ == "__main__":
    main()
