#!/usr/bin/env python3
"""
Prank UI (non-destructive) - improved and PyInstaller-friendly.

Features:
- resource_path() works with PyInstaller --onefile
- Attempts to import numpy._distributor_init before pygame to avoid
  the "CPU dispatcher tracer already initialized" error in some builds
- Graceful handling of missing image/music assets
- CLI options: --hours, --token-file, --music, --no-fullscreen
- DOES NOT encrypt or modify user files (fake decrypt only)
"""

from __future__ import annotations
import sys
import os
import secrets
import time
import argparse
from datetime import datetime, timedelta

# Try to pre-import numpy distributor init to avoid PyInstaller numpy
# initialization issues when packaging. Fail silently if numpy is not present.

# bytebeat_banger.py
import math
import numpy as np
import sounddevice as sd

def cbrt(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)

# Settings


try:
    import numpy._distributor_init  # type: ignore
except Exception:
    # It's okay if this fails — import pygame next and continue.
    pass

# Now import pygame (after the numpy pre-import attempt)
try:
    import pygame
except Exception as e:
    # If pygame is not installed, provide a clear message and exit.
    print("pygame is required to run this script. Install it with `pip install pygame`.")
    raise

# ----------------------------
# Defaults / Config
# ----------------------------
DEFAULT_TIMER_HOURS = 24
ERROR_SHOW_MS = 1400        # ms that "Wrong token" is visible
DECRYPT_FAKE_TIME = 1.2     # seconds for quick fake decrypt
DEFAULT_TOKEN_FILENAME = "token.txt"
DEFAULT_MUSIC_FILE = "song.mp3"
DEFAULT_LOGO = "fsociety.png"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (220, 20, 20)

# ----------------------------
# Utilities
# ----------------------------
def resource_path(relative_path: str) -> str:
    """
    Return absolute path to resource, works for dev and PyInstaller --onefile.
    """
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

def gen_token(path: str) -> str:
    """Generate a token and write it to `path` (best-effort restrictive perms)."""
    t = secrets.token_hex(4)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(t)
        try:
            os.chmod(tmp, 0o600)  # best-effort on Windows/Unix
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception as e:
        print("[WARN] Could not write token file:", e)
    print(f"[DEBUG] Generated token: {t}")
    return t

def init_pygame_and_audio():
    pygame.init()
    try:
        pygame.mixer.init()
    except Exception:
        # continue without audio if mixer fails
        pass
    pygame.font.init()
    return pygame.time.Clock()

def load_and_scale_image(path: str, max_width: int):
    """Return a Surface or None if not available."""
    try:
        surf = pygame.image.load(path).convert_alpha()
    except Exception:
        return None
    if surf.get_width() > max_width:
        scale = max_width / surf.get_width()
        new_size = (int(surf.get_width() * scale), int(surf.get_height() * scale))
        surf = pygame.transform.smoothscale(surf, new_size)
    return surf

def draw_centered_text(surface: pygame.Surface, text: str, font: pygame.font.Font, color, y: int):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(surface.get_width() // 2, y))
    surface.blit(surf, rect)
    return rect

def try_play_music(filename: str | None):
    if not filename:
        return
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.set_volume(0.8)
        pygame.mixer.music.play()
    except Exception:
        # fail silently
        pass

# ----------------------------
# Main
# ----------------------------
def main(args):
    clock = init_pygame_and_audio()

    # Window mode
    flags = 0 if args.no_fullscreen else pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode((0, 0), flags)
    except Exception:
        # fallback to windowed
        screen = pygame.display.set_mode((1280, 720))

    sw, sh = screen.get_size()
    pygame.display.set_caption("Fsociety - prank")

    # Fonts (fallbacks handled by pygame)
    big_font = pygame.font.SysFont("consolas", max(34, sw // 24))
    med_font = pygame.font.SysFont("consolas", max(18, sw // 48))
    small_font = pygame.font.SysFont("consolas", max(14, sw // 70))

    # Try loading image (use resource_path so it works from exe)
    logo_path = resource_path(args.logo or DEFAULT_LOGO)
    image = load_and_scale_image(logo_path, int(sw * 0.45))
    if image:
        image_rect = image.get_rect()
        top_margin = max(18, sh // 48)
        image_rect.midtop = (sw // 2, top_margin)
    else:
        # fake a small top area if no image
        image_rect = pygame.Rect(0, 0, 0, max(0, sh // 8))
        image_rect.midtop = (sw // 2, max(18, sh // 48))

    spacing = max(16, sh // 64)
    warning_y = image_rect.bottom + spacing * 3
    timer_y   = warning_y + spacing * 3
    input_y   = sh - max(160, sh // 8)

    box_w = min(640, sw // 2)
    box_h = max(44, sh // 22)
    input_box = pygame.Rect(sw // 2 - box_w // 2, input_y, box_w, box_h)
    PLACEHOLDER = "enter decryption key..."
    padding_x = 12

    # State
    token_file = resource_path(args.token_file or DEFAULT_TOKEN_FILENAME)
    token = gen_token(token_file)
    end_time = datetime.now() + timedelta(hours=args.hours)

    # Try play music (resource_path)
    music_path = resource_path(args.music) if args.music else None
    try_play_music(music_path)

    user_text = ""
    error_message = ""
    error_shown_at = None
    caret_visible = True
    caret_last_toggle = pygame.time.get_ticks()

    decryption_started = False
    decrypt_start_time = None

    running = True
    while running:
        now_ticks = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_KP4:
                    running = False
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if user_text == token:
                        decryption_started = True
                        decrypt_start_time = time.time()
                        try:
                            pygame.mixer.music.fadeout(800)
                        except Exception:
                            pass
                    else:
                        error_message = "Wrong token"
                        error_shown_at = now_ticks
                        user_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                else:
                    if event.unicode and event.unicode.isprintable() and len(user_text) < 1024:
                        user_text += event.unicode
                        # reset caret blink on input
                        caret_visible = True
                        caret_last_toggle = now_ticks

        # Draw
        screen.fill(BLACK)
        if image:
            screen.blit(image, image_rect)
        else:
            draw_centered_text(screen, "fsociety", big_font, WHITE, image_rect.bottom - max(24, sh // 96))

        draw_centered_text(screen, "ALL YOUR FILES HAVE BEEN ENCRYPTED", big_font, RED, warning_y)

        remaining = end_time - datetime.now()
        total_sec = max(int(remaining.total_seconds()), 0)
        hours, rem = divmod(total_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        countdown_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        draw_centered_text(screen, countdown_str, med_font, WHITE, timer_y)

        # Access label + input box
        label = small_font.render("Access", True, WHITE)
        label_rect = label.get_rect(center=(sw // 2, input_box.y - 18))
        screen.blit(label, label_rect)

        pygame.draw.rect(screen, WHITE, input_box, width=2, border_radius=6)
        inner_rect = input_box.inflate(-4, -4)
        pygame.draw.rect(screen, BLACK, inner_rect, border_radius=6)

        available_w = inner_rect.width - padding_x * 2
        visible = user_text
        if visible and small_font.size(visible)[0] > available_w:
            start = 0
            L = len(visible)
            while start < L and small_font.size(visible[start:])[0] > available_w:
                start += 1
            visible = visible[start:]

        if user_text == "":
            ph_surf = small_font.render(PLACEHOLDER, True, WHITE)
            ph_surf.set_alpha(110)
            screen.blit(ph_surf, (inner_rect.x + padding_x, inner_rect.y + (inner_rect.height - ph_surf.get_height()) // 2))
            visible_width = 0
            text_surf = None
        else:
            text_surf = small_font.render(visible, True, WHITE)
            screen.blit(text_surf, (inner_rect.x + padding_x, inner_rect.y + (inner_rect.height - text_surf.get_height()) // 2))
            visible_width = text_surf.get_width()

        # Caret
        if now_ticks - caret_last_toggle >= 500:
            caret_visible = not caret_visible
            caret_last_toggle = now_ticks
        if caret_visible:
            caret_x = inner_rect.x + padding_x + visible_width
            text_h = (text_surf.get_height() if text_surf else small_font.get_height())
            caret_y = inner_rect.y + (inner_rect.height - text_h) // 2
            caret_h = max(12, text_h)
            pygame.draw.rect(screen, WHITE, (caret_x, caret_y, 2, caret_h))

        # Wrong token message
        if error_message and error_shown_at:
            if now_ticks - error_shown_at <= ERROR_SHOW_MS:
                err_surf = small_font.render(error_message, True, RED)
                err_rect = err_surf.get_rect(center=(sw // 2, input_box.y - 60))
                screen.blit(err_surf, err_rect)
            else:
                error_message = ""
                error_shown_at = None

        # Fake decrypt animation
        if decryption_started:
            elapsed = time.time() - decrypt_start_time
            screen.fill(BLACK)
            draw_centered_text(screen, "Decrypting files...", med_font, WHITE, sh // 2 - 24)
            pb_w = min(640, sw // 2)
            pb_h = 8
            pb_x = sw // 2 - pb_w // 2
            pb_y = sh // 2 + 12
            pygame.draw.rect(screen, WHITE, (pb_x, pb_y, pb_w, pb_h), border_radius=4, width=1)
            frac = min(elapsed / max(DECRYPT_FAKE_TIME, 0.001), 1.0)
            inner = int(pb_w * frac)
            if inner > 0:
                pygame.draw.rect(screen, WHITE, (pb_x + 1, pb_y + 1, max(1, inner - 2), pb_h - 2), border_radius=3)
            pygame.display.flip()
            if elapsed >= DECRYPT_FAKE_TIME:
                screen.fill(BLACK)
                draw_centered_text(screen, "All files are safe", med_font, WHITE, sh // 2)
                pygame.display.flip()
                pygame.time.delay(900)
                running = False
                continue

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    # Use sys.exit only when running as script; in .pyw double-click use return/exit gracefully
    try:
        sys.exit(0)
    except SystemExit:
        pass

# ----------------------------
# CLI Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fsociety prank (safe, fake decrypt)")
    parser.add_argument("--hours", type=float, default=DEFAULT_TIMER_HOURS, help="Countdown hours")
    parser.add_argument("--token-file", type=str, default=DEFAULT_TOKEN_FILENAME, help="Token filename (written next to script or inside exe temp)")
    parser.add_argument("--music", type=str, default=None, help="Music file name (use resource_path when bundling)")
    parser.add_argument("--no-fullscreen", dest="no_fullscreen", action="store_true", help="Run windowed instead of fullscreen")
    parser.add_argument("--logo", type=str, default=DEFAULT_LOGO, help="Logo file name (optional)")
    args = parser.parse_args()
    # Normalize music arg to use resource_path when provided
    args.music = args.music if args.music else DEFAULT_MUSIC_FILE if os.path.exists(resource_path(DEFAULT_MUSIC_FILE)) else None
    main(args)
SR = 8000          # bytebeat-style sample rate
DURATION = 2234234234320      # seconds
SAMPLES = int(SR * DURATION)

PI = math.pi
E = math.e
s = 0.999999999

# binary string used in expression
bin_str = '10110101010101011101010111010'

# arrays used in expression
arr1 = [4, 41, 34, 54, 7, 8]
arr2 = [12, 13]
arr3 = [1, 3, 5, 5, 1, 3, -2, -2]

# persistent state across samples
c = [0.0, 0.0]
d = 0.0

out = np.zeros(SAMPLES, dtype=np.float32)

# main per-sample loop
for idx in range(SAMPLES):
    t = idx  # integer time index
    i = 0    # resets each sample per original one-liner behaviour

    # inline function f: mutates c[i_index], returns (output, new_i)
    def f(pm, tm, pc, tc, o1, t1, o2, t2, i_index):
        # pm,tm,pc,tc,o1,t1,o2,t2 are floats; i_index is 0 or 1
        # update c at position i_index (wrap to 0/1 if needed)
        pos = i_index & 1
        c[pos] += s * pm * pc + (math.sin(pm * o1 * t * PI / 2424546.0) * t1 - math.sin(pm * o2 * t * PI / 256.0) * t2) * tm
        out_o = math.sin(c[pos] * PI / 256.0) * tc * tm
        return out_o, i_index + 1

    # --- term 1 ---
    idx1 = ((t >> 22) % 6)
    # pick 12 or 13 depending on ((t//3)>>12)&1
    shift_choice = ((t // 3) >> 12) & 1
    shift_val = arr2[shift_choice]
    x_mask = (t >> shift_val) & 3
    # compute pm1 safely
    pm1 = (2.0 ** (arr1[idx1] / 12.0)) * (2.0 ** (1.25 * float(x_mask)))
    tm1 = (t / 4096.0) % 1.0 - 2.0
    term1_o, i = f(pm1, tm1, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0, i)
    term1 = term1_o / 4.0 * (((t / 16384.0) % 1.0) + 0.5)

    # --- term 2 ---
    idx2 = (t >> 16) & 7
    pm2 = 2.0 ** (arr3[idx2] / 12.0)
    selector = (t >> 17) & 1
    denom = selector + 1
    # compute the tm2 part with safe floats
    temp = (t / float(denom)) % 65536.0
    tm2 = cbrt((temp * denom) / 65536.0) - 2.0
    term2_o, i = f(pm2, tm2, 2.0, 1.0, 3.0, 5.0, 5.0, 9.0, i)
    term2 = term2_o / 7.0

    # --- term 3: update d using binary string ---
    val3 = (t % 40496) / 3072.0
    val3 = min(cbrt(val3), 1.0) - 1.0
    bit_index = (t >> 12) & 15
    bit_val = int(bin_str[bit_index])
    d += val3 * (bit_val / 24.0)
    term3 = math.sin(d) / E

    # --- term 4 ---
    inner = ((t >> 1) ^ ((t >> 2) * (t >> 1)))
    term4 = math.sin((t >> 1) * math.sin(float(inner)))
    term4 *= (( (t / 4096.0) % 1.0 ) - 1.0) ** 2
    term4 *= (((t >> 12) & 3) - 0.5) / 24.0

    # --- term 5 ---
    if ((t >> 14) & 1):
        mult = ((t / 16384.0) % 1.0) - 1.0
    else:
        mult = 0.0
    inner5 = int(t) ^ int(t * t)
    term5 = math.sin(t * math.sin(float(inner5))) / 4.0 * (mult ** 4)

    total = term1 + term2 + term3 + term4 + term5
    out[idx] = total

# normalize safely
peak = np.max(np.abs(out))
if peak > 1e-12:
    out = out / peak * 0.95
else:
    out = out * 0.0

print("Playing... (SR=%d, duration=%ds)" % (SR, DURATION))
sd.play(out, samplerate=SR)
sd.wait()
print("Done.")

