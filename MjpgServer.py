#!/usr/bin/python3
"""
    Author: Igor Maculan - n3wtron@gmail.com
    A simple MJPG stream HTTP server
"""
import os
import cv2
import time
import threading
import numpy as np
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[MjpgServer] PIL not available, Chinese text display will be disabled")

BOUNDARY = b'--boundarydonotcross'
STREAM_QUALITY = int(os.environ.get("MJPG_STREAM_QUALITY", 60))
SNAPSHOT_QUALITY = int(os.environ.get("MJPG_SNAPSHOT_QUALITY", 85))
FRAME_TIMEOUT = float(os.environ.get("MJPG_FRAME_TIMEOUT", 1.0))

img_show = None
_frame_lock = threading.Lock()

# 文本叠加相关全局变量
overlay_text = ""
_text_lock = threading.Lock()
_chinese_font = None
_latin_font = None
_font_initialized = False


def _init_chinese_font():
    """初始化文本字体（中文字体 + ASCII 字体）"""
    global _chinese_font, _latin_font, _font_initialized

    if _font_initialized:
        return

    _font_initialized = True

    if not PIL_AVAILABLE:
        print("[MjpgServer] PIL not available, text will be ASCII only")
        return

    font_size = 24

    # 尝试加载中文字体
    chinese_font_paths = [
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]

    for font_path in chinese_font_paths:
        if os.path.exists(font_path):
            try:
                candidate = ImageFont.truetype(font_path, font_size)
                test_img = Image.new('RGB', (100, 100))
                test_draw = ImageDraw.Draw(test_img)
                try:
                    test_draw.text((10, 10), "Test 测试", font=candidate)
                    _chinese_font = candidate
                    print(f"[MjpgServer] Successfully loaded and tested font: {font_path}")
                    break
                except Exception as exc:
                    print(f"[MjpgServer] Font loaded but rendering failed {font_path}: {exc}")
            except Exception as exc:
                print(f"[MjpgServer] Failed to load font {font_path}: {exc}")

    if _chinese_font is None:
        print("[MjpgServer] No truetype font available for Chinese text, overlay may be limited")

    # 尝试加载 ASCII 字体
    latin_font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]

    for font_path in latin_font_paths:
        if os.path.exists(font_path):
            try:
                _latin_font = ImageFont.truetype(font_path, font_size)
                print(f"[MjpgServer] Loaded ASCII overlay font: {font_path}")
                break
            except Exception as exc:
                print(f"[MjpgServer] Failed to load ASCII font {font_path}: {exc}")

    if _latin_font is None:
        try:
            _latin_font = ImageFont.load_default()
            if _latin_font:
                print("[MjpgServer] Using PIL default font for ASCII overlay")
        except Exception:
            _latin_font = None


def _draw_chinese_text(img, text, position=(10, 30), font_size=24,
                       text_color=(255, 255, 255), bg_color=(0, 0, 0),
                       padding=8, max_width=None):
    """
    在OpenCV图像上绘制中文文本（带背景，支持自动换行）

    参数:
        img: OpenCV图像 (BGR格式)
        text: 要显示的文本（支持中文）
        position: 文本位置 (x, y)
        font_size: 字体大小
        text_color: 文本颜色 (B, G, R)
        bg_color: 背景颜色 (B, G, R)
        padding: 背景padding
        max_width: 最大宽度（像素），None则使用图像宽度-20

    返回:
        绘制文本后的图像
    """
    if not PIL_AVAILABLE or not text:
        return img

    # 确保字体已初始化
    _init_chinese_font()

    if _chinese_font is None:
        # 如果没有中文字体，使用OpenCV的putText（只能显示ASCII）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, text_color, 2, cv2.LINE_AA)
        return img

    try:
        # 转换为PIL Image (RGB格式)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 设置最大宽度
        img_height, img_width = img.shape[:2]
        if max_width is None:
            max_width = img_width - position[0] - 20

        def _select_font(char):
            if _latin_font is not None and ord(char) < 128:
                return _latin_font
            if _chinese_font is not None:
                return _chinese_font
            return _latin_font or ImageFont.load_default()

        def _char_width(char, font_obj):
            try:
                width = font_obj.getlength(char)
            except Exception:
                width = None
            if width is None or width == 0:
                try:
                    bbox = font_obj.getbbox(char)
                    if bbox:
                        width = bbox[2] - bbox[0]
                except Exception:
                    width = None
            if width is None:
                mask = font_obj.getmask(char)
                width = mask.size[0] if mask else 0
            return max(width, 0.0)

        max_lines = 4
        lines = []
        current_chars = []
        current_width = 0.0
        truncated = False
        idx = 0
        text_len = len(text)

        while idx < text_len:
            char = text[idx]

            if char == '\r':
                idx += 1
                continue

            if char == '\n':
                if current_chars:
                    lines.append({'chars': current_chars, 'width': current_width})
                current_chars = []
                current_width = 0.0
                idx += 1
                if len(lines) >= max_lines:
                    truncated = idx < text_len
                    break
                continue

            font_obj = _select_font(char)
            char_w = _char_width(char, font_obj)

            if max_width and current_chars and (current_width + char_w) > max_width:
                lines.append({'chars': current_chars, 'width': current_width})
                current_chars = []
                current_width = 0.0
                if len(lines) >= max_lines:
                    truncated = True
                    break
                continue

            current_chars.append((char, font_obj, char_w))
            current_width += char_w
            idx += 1

        if not truncated:
            if current_chars:
                lines.append({'chars': current_chars, 'width': current_width})
            if len(lines) > max_lines:
                truncated = True
                lines = lines[:max_lines]

        if not lines:
            return img

        if truncated and lines:
            ellipsis_font = _latin_font or _chinese_font or ImageFont.load_default()
            ellipsis_width = _char_width('.', ellipsis_font)
            total_ellipsis = ellipsis_width * 3
            last_line = lines[-1]

            if max_width:
                # 尽量腾出空间放入省略号
                while last_line['chars'] and (last_line['width'] + total_ellipsis) > max_width:
                    removed_char = last_line['chars'].pop()
                    last_line['width'] -= removed_char[2]
                if not last_line['chars'] and total_ellipsis > max_width:
                    # 即使超出，也要显示省略号
                    pass

            for _ in range(3):
                last_line['chars'].append(('.', ellipsis_font, ellipsis_width))
            last_line['width'] += total_ellipsis

        # 计算总体边界框
        effective_font_size = max(
            getattr(_chinese_font, 'size', 0),
            getattr(_latin_font, 'size', 0),
            font_size
        )
        line_height = effective_font_size + 6
        total_height = len(lines) * line_height
        max_line_width = max(line['width'] for line in lines)

        # 绘制半透明背景矩形
        bg_box = [
            position[0] - padding,
            position[1] - padding - effective_font_size,
            position[0] + int(round(max_line_width)) + padding,
            position[1] + total_height + padding
        ]

        # 创建半透明背景
        overlay = img_pil.copy()
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(bg_box, fill=(bg_color[2], bg_color[1], bg_color[0]))

        # 混合背景（70%透明度）
        img_pil = Image.blend(img_pil, overlay, alpha=0.7)
        draw = ImageDraw.Draw(img_pil)

        # 绘制多行文本
        y_offset = position[1]
        for line in lines:
            x_offset = float(position[0])
            for char, font_obj, char_w in line['chars']:
                draw.text(
                    (int(round(x_offset)), int(round(y_offset))),
                    char,
                    font=font_obj,
                    fill=(text_color[2], text_color[1], text_color[0])
                )
                x_offset += char_w
            y_offset += line_height

        # 转换回OpenCV格式 (BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"[MjpgServer] Error drawing text: {e}")
        return img


def set_overlay_text(text):
    """设置要叠加到画面上的文本"""
    global overlay_text
    with _text_lock:
        overlay_text = text


def get_overlay_text():
    """获取当前要叠加的文本"""
    with _text_lock:
        return overlay_text


def clear_overlay_text():
    """清除叠加文本"""
    set_overlay_text("")


def _get_frame():
    with _frame_lock:
        if img_show is None:
            return None
        return img_show.copy()


def set_frame(frame):
    global img_show
    if frame is None:
        with _frame_lock:
            img_show = None
        return

    # 克隆frame避免修改原图
    frame_with_text = frame.copy()

    # 获取要叠加的文本
    text = get_overlay_text()
    if text:
        # 在左上角绘制文本
        frame_with_text = _draw_chinese_text(
            frame_with_text,
            text,
            position=(10, 30),
            text_color=(0, 255, 0),  # 绿色文字
            bg_color=(0, 0, 0),      # 黑色背景
            padding=8
        )

    with _frame_lock:
        img_show = frame_with_text


def clear_frame():
    global img_show
    with _frame_lock:
        img_show = None


class MJPGHandler(BaseHTTPRequestHandler):
    server_version = "MJPGServer/1.0"
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        # Silence default logging to avoid flooding stdout
        return

    def _write_frame(self, frame):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY]
        ret, jpg = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            return
        jpg_bytes = jpg.tobytes()
        try:
            self.wfile.write(BOUNDARY + b"\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode('ascii'))
            self.wfile.write(jpg_bytes)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    def do_GET(self):
        if self.path == '/?action=snapshot':
            frame = _get_frame()
            if frame is None:
                self.send_error(503, "No frame available")
                return
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_QUALITY]
            ret, jpg = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                self.send_error(500, "Failed to encode snapshot")
                return
            jpg_bytes = jpg.tobytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(jpg_bytes))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            self.wfile.write(jpg_bytes)
            return

        self.send_response(200)
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--boundarydonotcross')
        self.end_headers()

        last_send = time.time()
        try:
            while True:
                frame = _get_frame()
                if frame is None:
                    if time.time() - last_send > FRAME_TIMEOUT:
                        time.sleep(0.01)
                    continue
                self._write_frame(frame)
                last_send = time.time()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print("MJPG stream error:", exc)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True
    allow_reuse_address = True


def startMjpgServer():
    try:
        server = ThreadedHTTPServer(('', 8080), MJPGHandler)
        print("MJPG server started on port 8080")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
