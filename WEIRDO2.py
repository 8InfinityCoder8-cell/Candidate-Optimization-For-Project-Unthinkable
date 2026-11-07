#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font as tkfont
import subprocess
import threading
import sys
import os
import re
import json
import queue
import colorsys
import math
import platform
import tempfile
import textwrap
import zipfile
from xml.sax.saxutils import escape as xml_escape
import importlib.metadata as importlib_metadata
import urllib.request
import urllib.error
import urllib.parse


# ---------------------------------------------------------------------------
# Known "gotcha" aliases: what people type vs. the real pip/PyPI name
# ---------------------------------------------------------------------------
PACKAGE_ALIASES = {
    "pytorch": "torch",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "pil": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "discord": "discord.py",
    "tf": "tensorflow",
    "win32api": "pywin32",
    "serial": "pyserial",
    "mysql": "mysql-connector-python",
    "attr": "attrs",
    "jwt": "pyjwt",
}

# Curated, small list of "if the real package can't be built/installed on this
# Python, here's an actively-maintained drop-in replacement that provides the
# same import name". Used only as a last-resort auto-fix step, and always
# logged clearly so the user knows exactly what actually got installed.
PACKAGE_FALLBACKS = {
    "pygame": "pygame-ce",  # community fork, same `import pygame`, better new-Python support
}

# A handful of quick-pick "featured" packages shown as chips
FEATURED_PACKAGES = [
    "numpy", "pandas", "pulp", "torch", "tensorflow", "matplotlib",
    "requests", "flask", "django", "opencv-python", "scikit-learn",
    "pillow", "beautifulsoup4", "selenium", "pygame",
]

PYPI_JSON_URL = "https://pypi.org/pypi/{}/json"
PYPI_SEARCH_URL = "https://pypi.org/search/?q={}"
USER_AGENT = "Mozilla/5.0 (WeirdoInstaller/1.0)"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def hsv_hex(h, s=0.85, v=1.0):
    """Convert a hue (0-1) to a #RRGGBB hex color string."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def lerp_color(c1, c2, t):
    """Linearly interpolate between two #RRGGBB colors, t in [0,1]."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def ease_out_bounce(t):
    """Standard 'ease out bounce' easing curve, t in [0,1]."""
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


# ---------------------------------------------------------------------------
# Shared "old CRT terminal" widget helpers, used by the main window, the
# module manager, and the script preview so the whole app reads as one
# green-on-black computer rather than a set of separately-styled windows.
# ---------------------------------------------------------------------------
def term_button(parent, text, command, bg, accent, fg=None,
                 font=("Consolas", 10, "bold"), padx=12, pady=6, state="normal"):
    """A thin outlined, bracket-style button (no solid color fill) --
    a 1px accent border around a flat bg/fg button. Returns (wrap, button):
    pack/grid `wrap`, but keep `button` to .configure(state=..., text=...)."""
    fg = fg or accent
    wrap = tk.Frame(parent, bg=accent, bd=0, highlightthickness=0)
    btn = tk.Button(
        wrap, text=text, command=command, bg=bg, fg=fg,
        activebackground=bg, activeforeground=accent,
        relief="flat", bd=0, font=font, padx=padx, pady=pady, state=state,
        disabledforeground="#2f4a3c", cursor="hand2",
    )
    btn.pack(padx=1, pady=1)
    return wrap, btn


def term_panel(parent, bg, accent, thickness=1, **kwargs):
    """A borderless Frame with a thin accent-colored outline, standing in
    for the boxed panels on an old terminal screen."""
    return tk.Frame(
        parent, bg=bg, highlightbackground=accent, highlightcolor=accent,
        highlightthickness=thickness, bd=0, **kwargs
    )


def term_header_bar(parent, text, bg, accent, font=("Consolas", 11, "bold")):
    """A bordered header strip with a double rule underneath it, echoing
    the boxed 'SELECT AN ITEM' banner on the reference terminal screen."""
    outer = tk.Frame(parent, bg=bg)
    tk.Label(outer, text=text, bg=bg, fg=accent, font=font, anchor="w").pack(
        fill="x", padx=10, pady=(6, 2)
    )
    tk.Frame(outer, bg=accent, height=2).pack(fill="x", padx=0, pady=(0, 1))
    tk.Frame(outer, bg=accent, height=1).pack(fill="x", padx=0, pady=(0, 6))
    return outer


def package_exists_on_pypi(name):
    """Return True/False/None (None = couldn't reach PyPI to check)."""
    req = urllib.request.Request(PYPI_JSON_URL.format(name), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return None
    except Exception:
        return None


def search_pypi(query, limit=12):
    """
    Scrape PyPI's search page for package names + short descriptions.
    Returns a list of (name, summary) tuples. Returns [] on failure
    (e.g. offline, or PyPI changed their page layout).
    """
    url = PYPI_SEARCH_URL.format(urllib.parse.quote_plus(query))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    names = re.findall(
        r'package-snippet__name">\s*([^<]+?)\s*</span>', html
    )
    summaries = re.findall(
        r'package-snippet__description">\s*([^<]*?)\s*</p>', html
    )
    results = []
    for i, name in enumerate(names[:limit]):
        summary = summaries[i].strip() if i < len(summaries) else ""
        results.append((name.strip(), summary))
    return results


# ---------------------------------------------------------------------------
# Diagnosis engine: turns pip's raw error output into plain-English advice
# ---------------------------------------------------------------------------
def diagnose_failure(module_name, combined_output, return_code):
    """
    Look at pip's output and guess what went wrong.
    Returns (short_title, explanation, suggestion, fix_action)
    fix_action is one of: None, "use_alias:<name>", "retry_user", "retry", "search"
    """
    text = combined_output.lower()

    alias = PACKAGE_ALIASES.get(module_name.lower())
    if alias and ("no matching distribution" in text or "could not find a version" in text):
        return (
            "Package name looks off",
            f"PyPI has no package literally called '{module_name}'.",
            f"Did you mean '{alias}'? That's the real pip name for it.",
            f"use_alias:{alias}",
        )

    if "no matching distribution" in text or "could not find a version that satisfies" in text:
        return (
            "Package not found on PyPI",
            f"pip could not find any package named '{module_name}'.",
            "Double-check the spelling, or use Search to find the correct package name.",
            "search",
        )

    if "permission denied" in text or "access is denied" in text or "winerror 5" in text:
        return (
            "Permission error",
            "pip doesn't have permission to write to your Python install folder.",
            "Retry installing with the --user flag, or run this program as Administrator / with sudo.",
            "retry_user",
        )

    if any(k in text for k in ["temporary failure in name resolution", "connection aborted",
                                "connection timed out", "could not connect", "network is unreachable",
                                "max retries exceeded"]):
        return (
            "Network problem",
            "pip couldn't reach PyPI's servers.",
            "Check your internet connection (or proxy/VPN/firewall settings) and try again.",
            "retry",
        )

    if "microsoft visual c++" in text or "error: command 'cl.exe' failed" in text:
        return (
            "Missing build tools",
            "This package needs to compile some C/C++ code, and the compiler is missing.",
            "Install 'Microsoft C++ Build Tools' (Windows) or build-essential (Linux), then retry.",
            "retry",
        )

    if any(k in text for k in [
        "no module named 'distutils", "no module named 'setuptools._distutils",
        "no module named distutils", "msvccompiler",
    ]):
        return (
            "Package too old for this Python version",
            "This package is trying to build itself from source using a legacy build "
            "system that needs 'distutils', which was removed from the standard "
            "library in Python 3.12+ — and no ready-made wheel exists yet for your "
            "Python version, so pip fell back to a source build that can't succeed.",
            "Try forcing an existing prebuilt wheel, or upgrading pip/setuptools/wheel "
            "first; if the project has a maintained fork with newer-Python support, "
            "that's the most reliable fix.",
            "binary_only",
        )

    if "externally-managed-environment" in text:
        return (
            "System Python is protected",
            "Your OS blocks pip from installing into the system Python directly.",
            "Retry with --user, or better: create/use a virtual environment (python -m venv venv).",
            "retry_user",
        )

    if return_code != 0:
        return (
            "Install failed",
            "pip exited with an error that doesn't match a known pattern.",
            "Scroll the log above for the real reason, then try again — it may be transient.",
            "retry",
        )

    return ("Unknown issue", "Something unexpected happened.", "Please retry.", "retry")


# ---------------------------------------------------------------------------
# Auto-Fix ladders: a small, honest set of *known* remediations tried
# automatically, in order, before giving up and asking the user to decide.
# This is rule-based, not a general "AI reads any error" system — it covers
# the failure categories diagnose_failure() can actually name with confidence.
# ---------------------------------------------------------------------------
HEAL_PLANS = {
    "retry_user": [
        {"label": "Retrying with the --user flag (avoids the permission problem)",
         "args": ["--user"]},
    ],
    "retry": [
        {"label": "Retrying once more (in case it was a transient hiccup)",
         "args": []},
    ],
    "binary_only": [
        {"label": "Forcing an existing prebuilt wheel instead of building from source",
         "args": ["--only-binary=:all:"]},
        {"label": "Upgrading pip/setuptools/wheel, then retrying a normal install",
         "pre_cmd": [sys.executable, "-m", "pip", "install", "--upgrade",
                     "pip", "setuptools", "wheel"],
         "args": []},
    ],
}


# ---------------------------------------------------------------------------
# Installed-package inspection (uses only the stdlib importlib.metadata)
# ---------------------------------------------------------------------------
def list_installed_packages():
    """Return a sorted, de-duplicated [(name, version), ...] of everything
    installed in the current Python environment."""
    pkgs = []
    seen = set()
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        pkgs.append((name, dist.version or "?"))
    pkgs.sort(key=lambda t: t[0].lower())
    return pkgs


def list_installed_packages_detailed():
    """Return de-duplicated list of dicts with name, version, size_bytes,
    install_mtime (METADATA/PKG-INFO mtime as proxy), and file_count.
    Used by the module manager for multi-criteria sorting."""
    pkgs = []
    seen = set()
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        version = dist.version or "?"
        size_bytes = 0
        file_count = 0
        install_mtime = 0.0
        try:
            files = dist.files or []
            for f in files:
                try:
                    p = dist.locate_file(f)
                    if os.path.isfile(p):
                        size_bytes += os.path.getsize(p)
                        file_count += 1
                except Exception:
                    continue
            # Prefer METADATA / PKG-INFO mtime as an install-time proxy
            for f in files:
                s = str(f).replace("\\", "/")
                if s.endswith(".dist-info/METADATA") or s.endswith(".egg-info/PKG-INFO"):
                    try:
                        meta_path = dist.locate_file(f)
                        if os.path.isfile(meta_path):
                            install_mtime = os.path.getmtime(meta_path)
                            break
                    except Exception:
                        pass
        except Exception:
            pass

        pkgs.append({
            "name": name,
            "version": version,
            "size_bytes": size_bytes,
            "file_count": file_count,
            "install_mtime": install_mtime,
        })
    return pkgs


def human_readable_size(num_bytes):
    num = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def get_package_details(name):
    """Gather version, size-on-disk, location, summary, author, homepage
    for one installed package. Returns None if it isn't installed."""
    try:
        dist = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return None

    meta = dist.metadata
    size_bytes = 0
    file_count = 0
    size_known = False
    try:
        files = dist.files or []
        for f in files:
            try:
                p = dist.locate_file(f)
                if os.path.isfile(p):
                    size_bytes += os.path.getsize(p)
                    file_count += 1
            except Exception:
                continue
        size_known = file_count > 0
    except Exception:
        pass

    try:
        location = str(dist.locate_file(""))
    except Exception:
        location = "Unknown"

    homepage = meta.get("Home-page") or meta.get("Project-URL") or "Not listed"
    author = meta.get("Author") or meta.get("Author-email") or "Not listed"

    return {
        "name": meta.get("Name", name),
        "version": dist.version,
        "summary": meta.get("Summary") or "No description provided.",
        "author": author,
        "homepage": homepage,
        "location": location,
        "size_bytes": size_bytes,
        "size_known": size_known,
        "file_count": file_count,
    }


def generate_install_script(name, version=None):
    """A small, portable Python script that installs one package.
    Pin the version so the target environment gets the exact same build."""
    package_spec = f"{name}=={version}" if version else name
    return f'''#!/usr/bin/env python3
"""
Auto-generated by W.E.I.R.D.O. Installer
--------------------------------------------------
Installs: {package_spec}

Usage:
    python install_{name}.py

Just copy this file to any machine with Python + pip installed
and run it there to install the same package/version.
"""
import subprocess
import sys


def main():
    package = "{package_spec}"
    print(f"Installing {{package}} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("Done! '{name}' is now installed.")


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Dependency-free exporters: plain-text PDF and minimal .docx writers.
# No reportlab / python-docx needed - built by hand with only the stdlib.
# ---------------------------------------------------------------------------
def write_simple_pdf(path, title, body_text):
    """Write a small monospaced PDF containing `title` + `body_text`."""
    font_size = 10
    line_height = 14
    margin_top = 750
    margin_left = 50
    lines_per_page = 45

    def pdf_escape(s):
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    wrapped = []
    for raw_line in body_text.split("\n"):
        if not raw_line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(raw_line, width=95, replace_whitespace=False) or [""])

    pages = [wrapped[i:i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)] or [[]]

    font_obj_num = 3
    page_obj_nums, content_obj_nums = [], []
    n = 4
    for _ in pages:
        page_obj_nums.append(n); n += 1
        content_obj_nums.append(n); n += 1

    kids = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>",
        font_obj_num: "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    }

    for idx, page_lines in enumerate(pages):
        page_num, content_num = page_obj_nums[idx], content_obj_nums[idx]
        objs[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_num} 0 R >>"
        )
        stream = [f"BT /F1 {font_size} Tf {margin_left} {margin_top} Td {line_height} TL"]
        if idx == 0:
            stream.append(f"({pdf_escape(title)}) Tj T*")
            stream.append("T*")
        for line in page_lines:
            stream.append(f"({pdf_escape(line)}) Tj T*")
        stream.append("ET")
        content_stream = "\n".join(stream)
        objs[content_num] = f"<< /Length {len(content_stream)} >>\nstream\n{content_stream}\nendstream"

    buf = bytearray(b"%PDF-1.4\n")
    max_obj = max(objs.keys())
    offsets = {}
    for num in range(1, max_obj + 1):
        if num not in objs:
            continue
        offsets[num] = len(buf)
        buf += f"{num} 0 obj\n{objs[num]}\nendobj\n".encode("latin-1", errors="replace")

    xref_offset = len(buf)
    count = max_obj + 1
    buf += f"xref\n0 {count}\n".encode()
    buf += b"0000000000 65535 f \n"
    for num in range(1, max_obj + 1):
        offset = offsets.get(num)
        if offset is not None:
            buf += f"{offset:010} 00000 n \n".encode()
        else:
            buf += b"0000000000 00000 f \n"
    buf += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()

    with open(path, "wb") as f:
        f.write(buf)


def write_simple_docx(path, title, body_text):
    """Write a minimal but valid .docx (Word) file using only zipfile/xml."""
    paragraphs = []
    paragraphs.append(
        f'<w:p><w:r><w:rPr><w:b/><w:sz w:val="32"/></w:rPr>'
        f'<w:t xml:space="preserve">{xml_escape(title)}</w:t></w:r></w:p>'
    )
    for line in body_text.split("\n"):
        text = xml_escape(line) if line else ""
        paragraphs.append(
            '<w:p><w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
            f'<w:sz w:val="20"/></w:rPr><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
        )
    body_xml = "".join(paragraphs)

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)


def print_text_file(path):
    """Best-effort: hand a text file to the OS's default print flow.
    Returns (success, message)."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path, "print")  # noqa: [only exists on Windows]
            return True, "Sent to your default printer."
        elif system == "Darwin":
            subprocess.run(["lpr", path], check=True)
            return True, "Sent to your default printer via lpr."
        else:
            subprocess.run(["lpr", path], check=True)
            return True, "Sent to your default printer via lpr."
    except Exception as e:
        return False, (
            f"Could not print automatically ({e}).\n"
            "You can still export it as PDF/Word and print from there."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class WeirdoInstaller(tk.Tk):
    # -- Green-cyber / CRT terminal palette (the RGB border+glow effect
    #    layered on top of it is what stays "unique" — see _animate_glow) --
    BG = "#050f0a"
    PANEL_BG = "#0d1f15"
    TEXT_FG = "#cdffe0"
    MUTED_FG = "#5f9a7c"
    PHOSPHOR = "#8CFFC2"       # bright terminal-green accent
    PHOSPHOR_DIM = "#3f8a63"   # dimmer green for secondary text/lines

    ACRONYM_LETTERS = ["W", "E", "I", "R", "D", "O"]
    ACRONYM_DEFINITION = "WIZARD FOR EASY INSTALLATION OF REQUIRED DEPENDENCIES ONLINE"

    def __init__(self):
        super().__init__()
        self.title("W.E.I.R.D.O. — Wizard for Easy Installation of Required Dependencies Online")
        self.geometry("760x640")
        self.minsize(680, 560)
        self.configure(bg=self.BG)
        self.resizable(True, True)

        self.hue = 0.0
        self.pulse = 0.0
        self.install_proc = None
        self.log_queue = queue.Queue()
        self.last_failed_module = None
        self.glow_widgets = []
        self.active_heal = None  # tracks an in-progress auto-fix sequence, if any

        self._build_style()
        # Boot sequence first: loading bar -> falling W.E.I.R.D.O. letters
        # -> acronym definition -> hand off into the real UI below.
        self._run_boot_intro(on_complete=self._start_main_app)

    def _start_main_app(self):
        """Called once the intro finishes; builds the actual tool UI.
        Everything below this point is the same app/flow as before —
        only the intro screen, title, and color palette changed."""
        self._build_widgets()
        self._collect_glow_widgets(self)
        self._animate_glow()
        self._poll_log_queue()

    # ===================================================================
    #  BOOT INTRO — green-cyber CRT loading screen + falling-letter title
    #  (loading bar 0-100%  ->  W . E . I . R . D . O drop into place
    #   ->  acronym definition prints itself out, then intro clears)
    # ===================================================================
    def _run_boot_intro(self, on_complete):
        w, h = 760, 640
        self._boot_canvas = tk.Canvas(
            self, width=w, height=h, bg=self.BG, highlightthickness=0, bd=0
        )
        self._boot_canvas.pack(fill="both", expand=True)
        cv = self._boot_canvas

        # Faint static CRT scanlines across the whole boot screen.
        for y in range(0, h, 3):
            cv.create_line(0, y, w, y, fill="#08160f", width=1)
        # Soft vignette frame so the edges read as an old monitor.
        cv.create_rectangle(0, 0, w, h, outline=self.PHOSPHOR_DIM, width=2)

        cx, cy = w // 2, h // 2
        self._boot_state = {"cancelled": False}

        self._boot_loading_stage(cv, cx, cy, on_complete)

    def _boot_loading_stage(self, cv, cx, cy, on_complete):
        """Stage 1: a terminal-style progress bar counting 0% -> 100%."""
        label = cv.create_text(
            cx, cy - 70, text="INITIALIZING W.E.I.R.D.O. SYSTEM",
            font=("Consolas", 13, "bold"), fill=self.PHOSPHOR
        )
        bar_w, bar_h = 360, 22
        bar_x0, bar_y0 = cx - bar_w // 2, cy - bar_h // 2
        bar_x1, bar_y1 = cx + bar_w // 2, cy + bar_h // 2
        bar_outline = cv.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, outline=self.PHOSPHOR, width=2)
        fill_rect = cv.create_rectangle(bar_x0, bar_y0, bar_x0, bar_y1, fill=self.PHOSPHOR, width=0)
        pct_text = cv.create_text(
            cx, cy + 24, text="0%", font=("Consolas", 11, "bold"), fill=self.PHOSPHOR_DIM
        )
        status_lines = [
            "mounting pip subsystem...",
            "linking PyPI index...",
            "spinning up RGB core...",
            "calibrating phosphor glow...",
        ]
        status_text = cv.create_text(
            cx, cy + 48, text=status_lines[0], font=("Consolas", 9), fill=self.MUTED_FG
        )

        steps = 50
        delay_ms = 22
        # Every item drawn on this stage, so the whole loading bar (border
        # included) is actually wiped before the letters start falling --
        # nothing should be left floating on screen behind the intro text.
        stage_items = [label, bar_outline, fill_rect, pct_text, status_text]

        def tick(i=0):
            pct = int((i / steps) * 100)
            fill_x1 = bar_x0 + (bar_w * i / steps)
            cv.coords(fill_rect, bar_x0, bar_y0, fill_x1, bar_y1)
            cv.itemconfigure(pct_text, text=f"{pct}%")
            line_idx = min(len(status_lines) - 1, i // (steps // len(status_lines) + 1))
            cv.itemconfigure(status_text, text=status_lines[line_idx])
            if i < steps:
                self.after(delay_ms, lambda: tick(i + 1))
            else:
                self.after(250, lambda: self._boot_clear_stage(
                    cv, stage_items,
                    lambda: self._boot_letters_stage(cv, cx, cy, on_complete),
                ))

        tick()

    def _boot_clear_stage(self, cv, item_ids, next_stage):
        for item in item_ids:
            cv.delete(item)
        next_stage()

    def _boot_letters_stage(self, cv, cx, cy, on_complete):
        """Stage 2: W . E . I . R . D . O fall from the top and land one
        at a time, each one followed by its dot the instant it lands --
        the same cadence as the acronym reveal this title card is
        styled after."""
        big_font = tkfont.Font(family="Consolas", size=44, weight="bold")
        letters = self.ACRONYM_LETTERS
        dot_w = big_font.measure(".")
        letter_widths = [big_font.measure(l) for l in letters]
        total_w = sum(letter_widths) + dot_w * len(letters)
        start_x = cx - total_w / 2

        # Precompute each letter's and its trailing dot's centered x position
        # so the finished "W.E.I.R.D.O." sits centered once fully landed.
        positions = []
        x = start_x
        for lw in letter_widths:
            letter_cx = x + lw / 2
            x += lw
            dot_cx = x + dot_w / 2
            x += dot_w
            positions.append((letter_cx, dot_cx))
        end_x = start_x + total_w

        target_y = cy - 10
        underline_y = target_y + 34
        drop_start_y = cy - 260
        frames = 16
        frame_delay = 18

        def drop_letter(idx):
            if idx >= len(letters):
                cv.create_line(
                    start_x - 6, underline_y, end_x + 6, underline_y,
                    fill=self.PHOSPHOR, width=2
                )
                self.after(300, lambda: self._boot_definition_stage(
                    cv, cx, underline_y, on_complete
                ))
                return

            letter_cx, dot_cx = positions[idx]
            text_id = cv.create_text(
                letter_cx, drop_start_y, text=letters[idx],
                font=big_font, fill=self.PHOSPHOR
            )

            def animate(f=0):
                t = f / (frames - 1)
                eased = ease_out_bounce(t)
                y = drop_start_y + (target_y - drop_start_y) * eased
                cv.coords(text_id, letter_cx, y)
                if f < frames - 1:
                    self.after(frame_delay, lambda: animate(f + 1))
                else:
                    cv.coords(text_id, letter_cx, target_y)
                    cv.create_text(
                        dot_cx, target_y, text=".", font=big_font, fill=self.PHOSPHOR
                    )
                    self.after(150, lambda: drop_letter(idx + 1))

            animate()

        drop_letter(0)

    def _boot_definition_stage(self, cv, cx, underline_y, on_complete):
        """Stage 3: the acronym's meaning fades in underneath, phosphor-dim
        first and brightening slightly, like it's warming up on the tube."""
        def_id = cv.create_text(
            cx, underline_y + 26, text=self.ACRONYM_DEFINITION,
            font=("Consolas", 11, "bold"), fill=self.BG
        )
        steps = 12

        def fade(i=0):
            t = i / steps
            color = lerp_color(self.BG, self.PHOSPHOR_DIM, t)
            cv.itemconfigure(def_id, fill=color)
            if i < steps:
                self.after(35, lambda: fade(i + 1))
            else:
                self.after(900, lambda: self._boot_finish(cv, on_complete))

        fade()

    def _boot_finish(self, cv, on_complete):
        """Quick horizontal wipe transition out of the intro, then build
        the real tool UI in the same window."""
        w = int(cv.cget("width"))
        h = int(cv.cget("height"))
        wipe = cv.create_rectangle(0, 0, 0, h, fill=self.BG, outline="")
        steps = 14

        def wipe_step(i=0):
            x1 = int(w * (i / steps))
            cv.coords(wipe, 0, 0, x1, h)
            if i < steps:
                self.after(16, lambda: wipe_step(i + 1))
            else:
                cv.destroy()
                on_complete()

        wipe_step()

    # ---------------------------------------------------------- styling ---
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.PANEL_BG)
        style.configure("TLabel", background=self.PANEL_BG, foreground=self.TEXT_FG)
        style.configure(
            "RGB.Horizontal.TProgressbar",
            troughcolor="#0d1f15",
            background=self.PHOSPHOR,
            bordercolor="#0d1f15",
            lightcolor=self.PHOSPHOR,
            darkcolor=self.PHOSPHOR,
            thickness=22,
        )

    # --------------------------------------------------------- widgets ---
    def _build_widgets(self):
        # Outer frame acts as the animated RGB "border" -- this is the one
        # piece of the old rainbow identity we deliberately kept.
        self.border_frame = tk.Frame(self, bg=self.PHOSPHOR, bd=0)
        self.border_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.main = tk.Frame(self.border_frame, bg=self.BG)
        self.main.pack(fill="both", expand=True, padx=2, pady=2)

        # --- Title ---
        self.title_label = tk.Label(
            self.main, text="W.E.I.R.D.O.", font=("Consolas", 24, "bold"),
            bg=self.BG, fg=self.PHOSPHOR
        )
        self.title_label.pack(pady=(14, 2))

        subtitle = tk.Label(
            self.main, text="WIZARD FOR EASY INSTALLATION OF REQUIRED DEPENDENCIES ONLINE",
            font=("Consolas", 9), bg=self.BG, fg=self.MUTED_FG
        )
        subtitle.pack(pady=(0, 4))
        tk.Frame(self.main, bg=self.PHOSPHOR_DIM, height=1).pack(fill="x", padx=20, pady=(0, 10))

        # --- Input row (bordered entry + bracket-style terminal buttons) ---
        input_row = tk.Frame(self.main, bg=self.BG)
        input_row.pack(fill="x", padx=20, pady=(0, 6))

        tk.Label(input_row, text="MODULE:", bg=self.BG, fg=self.TEXT_FG,
                  font=("Consolas", 11, "bold")).pack(side="left")

        entry_border = term_panel(input_row, bg="#0d1f15", accent=self.PHOSPHOR_DIM)
        entry_border.pack(side="left", fill="x", expand=True, padx=8)
        self.module_entry = tk.Entry(
            entry_border, font=("Consolas", 12), bg="#0d1f15", fg=self.PHOSPHOR,
            insertbackground=self.PHOSPHOR, relief="flat", highlightthickness=0
        )
        self.module_entry.pack(fill="x", expand=True, padx=6, ipady=4)
        self.module_entry.bind("<Return>", lambda e: self.start_install())

        install_wrap, self.install_btn = term_button(
            input_row, "[ INSTALL ]", self.start_install, self.BG, self.PHOSPHOR
        )
        install_wrap.pack(side="left", padx=(0, 4))

        search_wrap, self.search_btn = term_button(
            input_row, "[ SEARCH ]", self.start_search, self.BG, self.PHOSPHOR
        )
        search_wrap.pack(side="left", padx=(4, 0))

        manage_wrap, self.manage_btn = term_button(
            input_row, "[ MANAGE ]", self.open_module_manager, self.BG, self.PHOSPHOR
        )
        manage_wrap.pack(side="left", padx=(4, 0))

        clear_wrap, self.clear_btn = term_button(
            input_row, "[ CLEAR ]", self.clear_interface, self.BG, self.PHOSPHOR_DIM
        )
        clear_wrap.pack(side="left", padx=(4, 0))

        # --- Auto-Fix toggle ---
        autofix_row = tk.Frame(self.main, bg=self.BG)
        autofix_row.pack(fill="x", padx=20, pady=(0, 4))
        self.auto_heal_var = tk.BooleanVar(value=True)
        autofix_check = tk.Checkbutton(
            autofix_row, text="AUTO-FIX FAILED INSTALLS", variable=self.auto_heal_var,
            bg=self.BG, fg=self.PHOSPHOR, selectcolor="#0d1f15", activebackground=self.BG,
            activeforeground=self.PHOSPHOR, font=("Consolas", 9, "bold"),
            highlightthickness=0, bd=0
        )
        autofix_check.pack(side="left")
        tk.Label(
            autofix_row,
            text="— tries known fixes automatically (correct name, --user, prebuilt "
                 "wheel, build-tool upgrade, then a compatible alternative if one exists)",
            bg=self.BG, fg=self.MUTED_FG, font=("Consolas", 8)
        ).pack(side="left", padx=(6, 0))

        # --- Featured chips ---
        chip_frame = tk.Frame(self.main, bg=self.BG)
        chip_frame.pack(fill="x", padx=20, pady=(2, 10))
        tk.Label(chip_frame, text="QUICK PICKS:", bg=self.BG, fg=self.MUTED_FG,
                  font=("Consolas", 9, "bold")).pack(anchor="w")
        chips_wrap = tk.Frame(chip_frame, bg=self.BG)
        chips_wrap.pack(fill="x")
        for i, pkg in enumerate(FEATURED_PACKAGES):
            b = tk.Button(
                chips_wrap, text=pkg, font=("Consolas", 9),
                bg="#0d1f15", fg=self.PHOSPHOR, relief="flat", padx=8, pady=2,
                activebackground="#173a28", activeforeground=self.PHOSPHOR,
                highlightbackground=self.PHOSPHOR_DIM, highlightthickness=1, bd=0,
                cursor="hand2",
                command=lambda p=pkg: self._fill_entry(p)
            )
            b.grid(row=i // 5, column=i % 5, padx=3, pady=3, sticky="w")

        # --- Search results list ---
        results_frame = tk.Frame(self.main, bg=self.BG)
        results_frame.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(results_frame, text="SEARCH RESULTS (double-click to fill the box above):",
                  bg=self.BG, fg=self.MUTED_FG, font=("Consolas", 9, "bold")).pack(anchor="w")
        results_border = term_panel(results_frame, bg="#0d1f15", accent=self.PHOSPHOR_DIM)
        results_border.pack(fill="x", pady=(2, 0))
        self.results_list = tk.Listbox(
            results_border, height=5, bg="#0d1f15", fg="#e8ffef",
            selectbackground=self.PHOSPHOR, selectforeground=self.BG,
            relief="flat", font=("Consolas", 10), highlightthickness=0, bd=0
        )
        self.results_list.pack(fill="x", padx=2, pady=2)
        self.results_list.bind("<Double-Button-1>", self._pick_result)

        # --- Progress bar ---
        prog_frame = tk.Frame(self.main, bg=self.BG)
        prog_frame.pack(fill="x", padx=20, pady=(4, 4))
        self.status_label = tk.Label(
            prog_frame, text="IDLE — ENTER A MODULE AND CLICK [ INSTALL ].",
            bg=self.BG, fg=self.MUTED_FG, font=("Consolas", 10)
        )
        self.status_label.pack(anchor="w")
        self.progress = ttk.Progressbar(
            prog_frame, style="RGB.Horizontal.TProgressbar", mode="determinate", maximum=100
        )
        self.progress.pack(fill="x", pady=(4, 0))

        # --- Output console ---
        console_frame = tk.Frame(self.main, bg=self.BG)
        console_frame.pack(fill="both", expand=True, padx=20, pady=(8, 8))
        tk.Label(console_frame, text="OUTPUT LOG:", bg=self.BG, fg=self.MUTED_FG,
                  font=("Consolas", 9, "bold")).pack(anchor="w")

        console_wrap = term_panel(console_frame, bg="#020a05", accent=self.PHOSPHOR_DIM)
        console_wrap.pack(fill="both", expand=True, pady=(2, 0))
        self.console = tk.Text(
            console_wrap, bg="#020a05", fg=self.PHOSPHOR, insertbackground=self.PHOSPHOR,
            font=("Consolas", 9), relief="flat", wrap="word", height=10, state="disabled",
            highlightthickness=0, bd=0
        )
        scroll = tk.Scrollbar(console_wrap, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side="left", fill="both", expand=True, padx=(2, 0), pady=2)
        scroll.pack(side="right", fill="y")

        # --- Fix-it bar (hidden until an error happens; red = warning accent) ---
        self.fixit_frame = term_panel(self.main, bg="#170a0d", accent="#ff5577")
        self.fixit_label = tk.Label(
            self.fixit_frame, text="", bg="#170a0d", fg="#ffb3c6",
            font=("Consolas", 10), justify="left", wraplength=500
        )
        self.fixit_label.pack(side="left", padx=10, pady=8, fill="x", expand=True)
        fixit_wrap, self.fixit_btn = term_button(
            self.fixit_frame, "[ RETRY ]", self._do_fix_action, "#170a0d", "#ff5577"
        )
        fixit_wrap.pack(side="right", padx=10)
        self._fix_action = None
        # not packed yet -- shown on demand via _show_fixit()

    def _fill_entry(self, text):
        self.module_entry.delete(0, tk.END)
        self.module_entry.insert(0, text)

    def open_module_manager(self):
        win = ModuleManagerWindow(self)
        # Fold this new window's background-colored widgets into the
        # same RGB "breathing" animation as the main window.
        self._collect_glow_widgets(win)

    def clear_interface(self):
        """Reset the main UI to a clean slate: clear the module entry,
        search results, console log, progress, status, and any fix-it /
        auto-heal temporary state. Safe to call at any time (does not
        interrupt a running install)."""
        if self._is_installing():
            messagebox.showinfo(
                "W.E.I.R.D.O.",
                "An install is still running. Wait for it to finish, then clear.",
            )
            return

        self.module_entry.delete(0, tk.END)
        self.results_list.delete(0, tk.END)
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.status_label.configure(text="Idle — enter a module and click Install.")
        self.install_btn.configure(state="normal", text="⬇ Install")
        self.search_btn.configure(state="normal")
        self._hide_fixit()
        self.active_heal = None
        self.last_failed_module = None
        self._log("Interface cleared. Ready for the next install.\n")

    def _pick_result(self, _event):
        sel = self.results_list.curselection()
        if not sel:
            return
        line = self.results_list.get(sel[0])
        name = line.split(" — ")[0].strip()
        self._fill_entry(name)

    # ------------------------------------------------------- RGB glow ----
    def _collect_glow_widgets(self, widget):
        """Recursively find every widget still using the plain background
        color, so the glow animation can pulse all of them together."""
        try:
            if str(widget.cget("bg")) == self.BG:
                self.glow_widgets.append(widget)
        except tk.TclError:
            pass  # widget has no 'bg' option (e.g. some ttk widgets)
        for child in widget.winfo_children():
            self._collect_glow_widgets(child)

    def _animate_glow(self):
        self.hue = (self.hue + 0.006) % 1.0
        self.pulse = self.pulse + 0.05
        c1 = hsv_hex(self.hue)
        c2 = hsv_hex((self.hue + 0.5) % 1.0)

        # Border + title: full brightness, fast-cycling accent color
        self.border_frame.configure(bg=c1)
        self.title_label.configure(fg=c2)

        # Whole background: same hue, but dark & gently "breathing" so
        # text stays readable while it still visibly glows.
        breathe = 0.13 + 0.05 * (0.5 + 0.5 * math.sin(self.pulse))
        bg_glow = hsv_hex(self.hue, s=0.55, v=breathe)
        for w in list(self.glow_widgets):
            try:
                w.configure(bg=bg_glow)
            except tk.TclError:
                try:
                    self.glow_widgets.remove(w)
                except ValueError:
                    pass

        if self.progress["mode"] == "indeterminate" or self._is_installing():
            style = ttk.Style(self)
            style.configure("RGB.Horizontal.TProgressbar", background=c1, lightcolor=c1, darkcolor=c1)
        self.after(40, self._animate_glow)

    def _is_installing(self):
        return self.install_proc is not None and self.install_proc.poll() is None

    # ------------------------------------------------------------ log ----
    def _log(self, line):
        self.console.configure(state="normal")
        self.console.insert("end", line if line.endswith("\n") else line + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                kind = item[0]
                if kind == "line":
                    self._log(item[1])
                elif kind == "done":
                    self._on_install_finished(item[1], item[2])
                elif kind == "progress":
                    self.progress.configure(mode="determinate", value=item[1])
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    # -------------------------------------------------------- install ----
    def start_install(self, override_name=None):
        module_name = override_name or self.module_entry.get().strip()
        if not module_name:
            messagebox.showwarning("W.E.I.R.D.O.", "Please type a module name first.")
            return
        if self._is_installing():
            messagebox.showinfo("W.E.I.R.D.O.", "An install is already running — please wait.")
            return

        if override_name is None:
            # A genuinely new, user-initiated request — start a clean slate.
            self.active_heal = None

        self._hide_fixit()
        self.install_btn.configure(state="disabled", text="Installing…")
        self.status_label.configure(text=f"Installing '{module_name}'…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._log(f"\n$ pip install {module_name}")

        thread = threading.Thread(target=self._install_worker, args=(module_name, []), daemon=True)
        thread.start()

    def _install_worker(self, module_name, extra_args, install_target=None, pre_cmd=None):
        """Runs pip in the background. `module_name` is what we report back
        under (for auto-fix tracking/logging); `install_target` is what
        actually gets passed to pip (differs only for fallback-package steps)."""
        target = install_target or module_name

        if pre_cmd:
            self.log_queue.put(("line", f"$ {' '.join(pre_cmd)}"))
            try:
                pre_result = subprocess.run(pre_cmd, capture_output=True, text=True, timeout=180)
                for line in (pre_result.stdout + pre_result.stderr).splitlines():
                    self.log_queue.put(("line", line))
            except Exception as e:
                self.log_queue.put(("line", f"(build-tool upgrade step failed non-fatally: {e})"))

        cmd = [sys.executable, "-m", "pip", "install", target] + extra_args
        self.log_queue.put(("line", f"$ {' '.join(cmd)}"))
        output_lines = []
        try:
            self.install_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in self.install_proc.stdout:
                output_lines.append(line)
                self.log_queue.put(("line", line.rstrip("\n")))
            self.install_proc.wait()
            rc = self.install_proc.returncode
        except FileNotFoundError:
            output_lines.append("pip executable not found.")
            rc = -1
        except Exception as e:
            output_lines.append(f"Unexpected error launching pip: {e}")
            rc = -1

        combined = "\n".join(output_lines)
        self.log_queue.put(("done", module_name, (rc, combined, target)))

    def _on_install_finished(self, module_name, result):
        rc, combined_output, actual_target = result
        self.progress.stop()

        if rc == 0:
            self.progress.configure(mode="determinate", value=100)
            self.install_btn.configure(state="normal", text="⬇ Install")
            if actual_target != module_name:
                self.status_label.configure(
                    text=f"✅ Installed '{actual_target}' — a working replacement for '{module_name}'."
                )
                self._log(
                    f"✔ SUCCESS: Installed '{actual_target}' instead of '{module_name}' "
                    f"(same `import {module_name}` interface, but this build actually works here).\n"
                )
            else:
                self.status_label.configure(text=f"✅ '{module_name}' installed successfully!")
                self._log(f"✔ SUCCESS: '{module_name}' is now installed.\n")
            self._hide_fixit()
            self.active_heal = None
            return

        # --- failure: diagnose, then maybe auto-fix ---
        title, explanation, suggestion, fix_action = diagnose_failure(
            module_name, combined_output, rc
        )
        self._log(f"✘ FAILED: {title} — {explanation}\n")
        self.last_failed_module = module_name

        if self.auto_heal_var.get():
            if self._try_advance_auto_heal(module_name, fix_action):
                return  # a new auto-fix attempt just started; keep the UI in "working" state

        # Auto-fix disabled, not applicable, or exhausted -> hand it to the user.
        self.install_btn.configure(state="normal", text="⬇ Install")
        self.progress.configure(mode="determinate", value=0)
        tried = self.active_heal["idx"] if self.active_heal else 0
        tried_note = f" Auto-Fix tried {tried} approach(es) automatically, no luck." if tried else ""
        self.status_label.configure(text=f"❌ Failed to install '{module_name}'.{tried_note}")
        self._show_fixit(title, explanation, suggestion, fix_action)
        self.active_heal = None

    def _try_advance_auto_heal(self, module_name, fix_action):
        """Returns True if a new auto-fix attempt was kicked off (caller
        should wait for it), False if there's nothing more to try."""

        # A wrong-but-correctable name is a sure thing — always worth an
        # immediate, transparent retry under the right name.
        if fix_action.startswith("use_alias:") and not (
            self.active_heal and self.active_heal.get("alias_tried")
        ):
            correct = fix_action.split(":", 1)[1]
            self._log(f"🔧 Auto-Fix: '{module_name}' isn't on PyPI, but '{correct}' is — retrying as that.\n")
            self.active_heal = {"module": module_name, "alias_tried": True, "idx": 0}
            self.start_install(override_name=correct)
            return True

        # Start (or continue) a step-by-step ladder for this failure category.
        if self.active_heal is None or self.active_heal.get("module") != module_name:
            ladder = [dict(step) for step in HEAL_PLANS.get(fix_action, [])]
            fallback = PACKAGE_FALLBACKS.get(module_name.lower())
            if fallback:
                ladder.append({
                    "label": f"Trying '{fallback}' — a maintained drop-in replacement for '{module_name}'",
                    "fallback_package": fallback,
                })
            self.active_heal = {"module": module_name, "ladder": ladder, "idx": 0}

        heal = self.active_heal
        if heal["idx"] >= len(heal["ladder"]):
            return False

        step = heal["ladder"][heal["idx"]]
        heal["idx"] += 1
        self._run_heal_step(module_name, step, heal["idx"], len(heal["ladder"]))
        return True

    def _run_heal_step(self, module_name, step, step_num, total_steps):
        label = step.get("label", "Retrying")
        self._log(f"\n🔧 Auto-Fix attempt {step_num}/{total_steps}: {label}")
        self.status_label.configure(text=f"🔧 Auto-Fix {step_num}/{total_steps}: {label}…")
        self.install_btn.configure(state="disabled", text="Auto-Fixing…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        fallback_pkg = step.get("fallback_package")
        install_target = fallback_pkg  # None means "use module_name itself"
        thread = threading.Thread(
            target=self._install_worker,
            args=(module_name, step.get("args", [])),
            kwargs={"install_target": install_target, "pre_cmd": step.get("pre_cmd")},
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------ fix-it bar ---
    def _show_fixit(self, title, explanation, suggestion, fix_action):
        self.fixit_label.configure(text=f"⚠ {title}: {explanation}\n💡 {suggestion}")
        self._fix_action = fix_action
        label_text = {
            "retry": "Retry",
            "retry_user": "Retry with --user",
            "search": "Search PyPI",
        }
        if isinstance(fix_action, str) and fix_action.startswith("use_alias:"):
            correct = fix_action.split(":", 1)[1]
            label_text_val = f"Install '{correct}' instead"
        else:
            label_text_val = label_text.get(fix_action, "Retry")
        self.fixit_btn.configure(text=label_text_val)
        if not self.fixit_frame.winfo_ismapped():
            self.fixit_frame.pack(fill="x", padx=20, pady=(0, 10))

    def _hide_fixit(self):
        self.fixit_frame.pack_forget()
        self._fix_action = None

    def _do_fix_action(self):
        action = self._fix_action
        module_name = self.last_failed_module
        self._hide_fixit()
        if not action or not module_name:
            return
        if action.startswith("use_alias:"):
            correct = action.split(":", 1)[1]
            self._fill_entry(correct)
            self.start_install(override_name=correct)
        elif action == "retry_user":
            self._log(f"\n$ pip install --user {module_name}")
            self.install_btn.configure(state="disabled", text="Installing…")
            self.status_label.configure(text=f"Retrying '{module_name}' with --user…")
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            threading.Thread(
                target=self._install_worker, args=(module_name, ["--user"]), daemon=True
            ).start()
        elif action == "search":
            self.module_entry.delete(0, tk.END)
            self.module_entry.insert(0, module_name)
            self.start_search()
        else:  # plain retry
            self.start_install(override_name=module_name)

    # --------------------------------------------------------- search ----
    def start_search(self):
        query = self.module_entry.get().strip()
        if not query:
            messagebox.showwarning("W.E.I.R.D.O.", "Type a keyword to search PyPI for.")
            return
        self.results_list.delete(0, tk.END)
        self.results_list.insert(tk.END, "Searching PyPI…")
        self.search_btn.configure(state="disabled")
        threading.Thread(target=self._search_worker, args=(query,), daemon=True).start()

    def _search_worker(self, query):
        results = search_pypi(query)
        self.after(0, lambda: self._show_search_results(query, results))

    def _show_search_results(self, query, results):
        self.results_list.delete(0, tk.END)
        self.search_btn.configure(state="normal")
        if not results:
            self.results_list.insert(
                tk.END, f"No results (offline, or PyPI layout changed) for '{query}'."
            )
            return
        for name, summary in results:
            line = f"{name} — {summary}" if summary else name
            self.results_list.insert(tk.END, line[:120])



# ---------------------------------------------------------------------------
# Installed-modules manager: browse, inspect, uninstall, generate a
# portable install script, and export/print that script.
# ---------------------------------------------------------------------------
class ModuleManagerWindow(tk.Toplevel):
    """Browse, sort, filter, inspect, and uninstall installed packages.

    Sorting criteria (via the Sort menu):
      • Name A–Z / Z–A
      • Version (lexicographic)
      • Size (largest / smallest)
      • Date installed (newest / oldest) — uses METADATA mtime as a proxy
    Search is live, case-insensitive, and matches partial names.
    """

    SORT_OPTIONS = [
        ("Name A–Z", "name_asc"),
        ("Name Z–A", "name_desc"),
        ("Version", "version"),
        ("Size (largest)", "size_desc"),
        ("Size (smallest)", "size_asc"),
        ("Date installed (newest)", "date_desc"),
        ("Date installed (oldest)", "date_asc"),
    ]

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("W.E.I.R.D.O. — Installed Modules")
        self.geometry("880x600")
        self.minsize(740, 520)
        self.configure(bg=app.BG)

        self.all_packages = []       # list of detailed dicts
        self.selected_name = None
        self.sort_key = "name_asc"

        self._build_widgets()
        self._reload_packages()

    # ------------------------------------------------------- widgets ----
    def _build_widgets(self):
        app = self.app
        BG, PHOSPHOR, PHOSPHOR_DIM, MUTED_FG = app.BG, app.PHOSPHOR, app.PHOSPHOR_DIM, app.MUTED_FG
        mono = ("Consolas", 10)
        mono_bold = ("Consolas", 10, "bold")

        # --- Header, styled after the "H.E.L.P.E.R." title card: a big
        #     monospace wordmark over a double green rule. ---
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 0))
        self.count_label = tk.Label(
            header, text="W.E.I.R.D.O. — SCANNING ENVIRONMENT...",
            font=("Consolas", 15, "bold"), bg=BG, fg=PHOSPHOR
        )
        self.count_label.pack(side="left")
        refresh_wrap, refresh_btn = term_button(
            header, "[ REFRESH ]", self._reload_packages, BG, PHOSPHOR_DIM
        )
        refresh_wrap.pack(side="right")
        tk.Frame(self, bg=PHOSPHOR, height=2).pack(fill="x", padx=16, pady=(8, 1))
        tk.Frame(self, bg=PHOSPHOR, height=1).pack(fill="x", padx=16, pady=(0, 10))

        # --- "SELECT A MODULE" banner, boxed like the reference screen ---
        select_bar = term_panel(self, bg="#0d1f15", accent=PHOSPHOR_DIM)
        select_bar.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(
            select_bar, text="SELECT A MODULE", font=mono_bold, bg="#0d1f15", fg=PHOSPHOR
        ).pack(anchor="w", padx=10, pady=6)

        # Search + Sort row (responsive filter + multi-criteria sort)
        control_row = tk.Frame(self, bg=BG)
        control_row.pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(control_row, text="SEARCH:", bg=BG, fg=MUTED_FG, font=mono_bold).pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._apply_filter_and_sort())
        filter_border = term_panel(control_row, bg="#0d1f15", accent=PHOSPHOR_DIM)
        filter_border.pack(side="left", fill="x", expand=True, padx=(6, 10))
        filter_entry = tk.Entry(
            filter_border, textvariable=self.filter_var, bg="#0d1f15", fg=PHOSPHOR,
            insertbackground=PHOSPHOR, relief="flat", font=mono, highlightthickness=0
        )
        filter_entry.pack(fill="x", expand=True, padx=6, ipady=4)
        # Clear search button (resets only the filter)
        clear_wrap, clear_search_btn = term_button(
            control_row, "[ X ]", self._clear_search, BG, PHOSPHOR_DIM, padx=8, pady=4
        )
        clear_wrap.pack(side="left", padx=(0, 12))

        tk.Label(control_row, text="SORT:", bg=BG, fg=MUTED_FG, font=mono_bold).pack(side="left")
        self.sort_var = tk.StringVar(value=self.SORT_OPTIONS[0][0])
        sort_menu = ttk.Combobox(
            control_row, textvariable=self.sort_var,
            values=[label for label, _ in self.SORT_OPTIONS],
            state="readonly", width=22, font=mono
        )
        sort_menu.pack(side="left", padx=(6, 0))
        sort_menu.bind("<<ComboboxSelected>>", self._on_sort_changed)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # --- Left: bordered package list, inverted (block-cursor) selection
        #     just like the highlighted row on the reference item screen ---
        left = term_panel(body, bg="#0d1f15", accent=PHOSPHOR_DIM)
        left.pack(side="left", fill="both", expand=True)
        self.pkg_list = tk.Listbox(
            left, bg="#0d1f15", fg="#d8ffe9", selectbackground=PHOSPHOR, selectforeground=BG,
            relief="flat", font=mono, activestyle="none", highlightthickness=0, bd=0
        )
        self.pkg_list.pack(side="left", fill="both", expand=True, padx=(2, 0), pady=2)
        self.pkg_list.bind("<<ListboxSelect>>", self._on_select_package)
        pkg_scroll = tk.Scrollbar(left, command=self.pkg_list.yview)
        self.pkg_list.configure(yscrollcommand=pkg_scroll.set)
        pkg_scroll.pack(side="right", fill="y", padx=(0, 2), pady=2)

        # --- Right: bordered details panel ---
        right = term_panel(body, bg="#0d1f15", accent=PHOSPHOR_DIM)
        right.pack(side="left", fill="both", padx=(12, 0))
        right.pack_propagate(False)
        right.configure(width=320)

        details_header = term_header_bar(right, "MODULE DETAILS", "#0d1f15", PHOSPHOR)
        details_header.pack(fill="x")

        self.details_text = tk.Text(
            right, bg="#0d1f15", fg="#d8ffe9", relief="flat", wrap="word",
            font=mono, width=36, height=14, state="disabled",
            highlightthickness=0, bd=0
        )
        self.details_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        btn_col = tk.Frame(right, bg="#0d1f15")
        btn_col.pack(fill="x", padx=12, pady=(0, 12))

        script_wrap, self.script_btn = term_button(
            btn_col, "[ GENERATE INSTALL SCRIPT ]", self._open_script_preview,
            "#0d1f15", PHOSPHOR, state="disabled"
        )
        script_wrap.pack(fill="x", pady=(0, 6))

        uninstall_wrap, self.uninstall_btn = term_button(
            btn_col, "[ UNINSTALL ]", self._uninstall_selected,
            "#0d1f15", "#ff5577", state="disabled"
        )
        uninstall_wrap.pack(fill="x")

        # --- Footer: dynamic status line + a static control-hint bar,
        #     echoing the "[W/S] CURSOR [A/D] DETAILS ..." legend strip ---
        self.status_label = tk.Label(
            self, text="", bg=BG, fg=MUTED_FG, font=mono, anchor="w"
        )
        self.status_label.pack(fill="x", padx=16, pady=(0, 2))

        hint_bar = term_panel(self, bg="#0d1f15", accent=PHOSPHOR_DIM)
        hint_bar.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(
            hint_bar,
            text="[CLICK] SELECT MODULE    [GENERATE INSTALL SCRIPT] EXPORT    [UNINSTALL] REMOVE",
            bg="#0d1f15", fg=PHOSPHOR_DIM, font=("Consolas", 9), anchor="w"
        ).pack(fill="x", padx=10, pady=6)

    # ------------------------------------------------------- loading ----
    def _reload_packages(self):
        self.count_label.configure(text="W.E.I.R.D.O. — SCANNING ENVIRONMENT...")
        self.pkg_list.delete(0, tk.END)
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.configure(state="disabled")
        self.script_btn.configure(state="disabled")
        self.uninstall_btn.configure(state="disabled")
        self.selected_name = None
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        packages = list_installed_packages_detailed()
        self.after(0, lambda: self._show_packages(packages))

    def _show_packages(self, packages):
        self.all_packages = packages
        self.count_label.configure(
            text=f"W.E.I.R.D.O. — {len(packages)} MODULES INSTALLED"
        )
        self._apply_filter_and_sort()

    def _clear_search(self):
        self.filter_var.set("")
        # _apply_filter_and_sort is triggered by the trace

    def _on_sort_changed(self, _event=None):
        label = self.sort_var.get()
        for lbl, key in self.SORT_OPTIONS:
            if lbl == label:
                self.sort_key = key
                break
        self._apply_filter_and_sort()

    def _sorted_packages(self, packages):
        key = self.sort_key
        if key == "name_asc":
            return sorted(packages, key=lambda p: p["name"].lower())
        if key == "name_desc":
            return sorted(packages, key=lambda p: p["name"].lower(), reverse=True)
        if key == "version":
            return sorted(packages, key=lambda p: (p["version"] or "").lower())
        if key == "size_desc":
            return sorted(packages, key=lambda p: p["size_bytes"], reverse=True)
        if key == "size_asc":
            return sorted(packages, key=lambda p: p["size_bytes"])
        if key == "date_desc":
            return sorted(packages, key=lambda p: p["install_mtime"], reverse=True)
        if key == "date_asc":
            return sorted(packages, key=lambda p: p["install_mtime"])
        return packages

    def _apply_filter_and_sort(self):
        """Live case-insensitive partial-match filter + current sort order."""
        query = self.filter_var.get().strip().lower()
        filtered = []
        for pkg in self.all_packages:
            if query and query not in pkg["name"].lower():
                continue
            filtered.append(pkg)

        ordered = self._sorted_packages(filtered)
        self.pkg_list.delete(0, tk.END)
        for pkg in ordered:
            size_hint = human_readable_size(pkg["size_bytes"]) if pkg["size_bytes"] else "—"
            self.pkg_list.insert(
                tk.END, f"{pkg['name']}  ({pkg['version']})  ·  {size_hint}"
            )

        # Update count to reflect filtered view when searching
        total = len(self.all_packages)
        shown = len(ordered)
        if query:
            self.count_label.configure(
                text=f"W.E.I.R.D.O. — {shown} OF {total} MODULES  (FILTER: \u201c{query}\u201d)"
            )
        else:
            self.count_label.configure(
                text=f"W.E.I.R.D.O. — {total} MODULES INSTALLED"
            )

    # ------------------------------------------------------ selection ----
    def _on_select_package(self, _event):
        sel = self.pkg_list.curselection()
        if not sel:
            return
        line = self.pkg_list.get(sel[0])
        # Format: "name  (version)  ·  size"
        name = line.split("  (")[0].strip()
        self.selected_name = name
        self.status_label.configure(text=f"Loading details for '{name}'…")
        threading.Thread(target=self._details_worker, args=(name,), daemon=True).start()

    def _details_worker(self, name):
        info = get_package_details(name)
        self.after(0, lambda: self._show_details(name, info))

    def _show_details(self, name, info):
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        if info is None:
            self.details_text.insert("end", f"Could not read details for '{name}'.")
            self.script_btn.configure(state="disabled")
            self.uninstall_btn.configure(state="disabled")
            self.status_label.configure(text="")
            self.details_text.configure(state="disabled")
            return

        size_text = (
            human_readable_size(info["size_bytes"]) if info["size_known"] else "Unknown"
        )
        lines = [
            f"Name:      {info['name']}",
            f"Version:   {info['version']}",
            f"Size:      {size_text}",
            f"Files:     {info['file_count']}",
            f"Location:  {info['location']}",
            f"Author:    {info['author']}",
            f"Homepage:  {info['homepage']}",
            "",
            "Summary:",
            textwrap.fill(info["summary"], width=38),
        ]
        self.details_text.insert("end", "\n".join(lines))
        self.details_text.configure(state="disabled")
        self.script_btn.configure(state="normal")
        self.uninstall_btn.configure(state="normal")
        self.status_label.configure(text="")

    # ------------------------------------------------------ uninstall ----
    def _uninstall_selected(self):
        name = self.selected_name
        if not name:
            return
        if not messagebox.askyesno(
            "Confirm uninstall",
            f"Uninstall '{name}' from this Python environment?\nThis cannot be undone.",
            parent=self,
        ):
            return
        self.uninstall_btn.configure(state="disabled", text="Uninstalling…")
        self.status_label.configure(text=f"Uninstalling '{name}'…")
        threading.Thread(target=self._uninstall_worker, args=(name,), daemon=True).start()

    def _uninstall_worker(self, name):
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", name]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            ok = result.returncode == 0
            output = result.stdout + result.stderr
        except Exception as e:
            ok, output = False, str(e)
        self.after(0, lambda: self._on_uninstall_done(name, ok, output))

    def _on_uninstall_done(self, name, ok, output):
        self.uninstall_btn.configure(state="normal", text="🗑 Uninstall")
        if ok:
            self.status_label.configure(text=f"✔ '{name}' uninstalled.")
            self.details_text.configure(state="normal")
            self.details_text.delete("1.0", "end")
            self.details_text.configure(state="disabled")
            self.script_btn.configure(state="disabled")
            self.uninstall_btn.configure(state="disabled")
            self.selected_name = None
            self._reload_packages()
        else:
            self.status_label.configure(text=f"✘ Failed to uninstall '{name}'.")
            messagebox.showerror(
                "Uninstall failed",
                f"Could not uninstall '{name}'.\n\n{output[-500:]}",
                parent=self,
            )

    # --------------------------------------------------- script preview --
    def _open_script_preview(self):
        if not self.selected_name:
            return
        info = get_package_details(self.selected_name)
        if info is None:
            return
        script_text = generate_install_script(info["name"], info["version"])
        ScriptPreviewWindow(self.app, info["name"], script_text)


# ---------------------------------------------------------------------------
# Script preview / export window: save .py, copy, print, PDF, Word.
# ---------------------------------------------------------------------------
class ScriptPreviewWindow(tk.Toplevel):
    def __init__(self, app, module_name, script_text):
        super().__init__(app)
        self.app = app
        self.module_name = module_name
        self.script_text = script_text

        self.title(f"W.E.I.R.D.O. — Install Script — {module_name}")
        self.geometry("640x540")
        self.minsize(480, 400)
        self.configure(bg=app.BG)

        tk.Label(
            self, text=f"PORTABLE INSTALL SCRIPT — '{module_name.upper()}'",
            font=("Consolas", 13, "bold"), bg=app.BG, fg=app.PHOSPHOR
        ).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Frame(self, bg=app.PHOSPHOR_DIM, height=1).pack(fill="x", padx=16, pady=(0, 6))

        tk.Label(
            self,
            text="Copy this file to any machine with Python + pip and run it there.",
            font=("Consolas", 9), bg=app.BG, fg=app.MUTED_FG
        ).pack(anchor="w", padx=16, pady=(0, 8))

        text_wrap = term_panel(self, bg="#020a05", accent=app.PHOSPHOR_DIM)
        text_wrap.pack(fill="both", expand=True, padx=16)
        self.text = tk.Text(
            text_wrap, bg="#020a05", fg=app.PHOSPHOR, insertbackground=app.PHOSPHOR,
            font=("Consolas", 10), relief="flat", wrap="none", highlightthickness=0, bd=0
        )
        self.text.insert("1.0", script_text)
        self.text.configure(state="normal")  # left editable in case user wants to tweak it
        self.text.pack(fill="both", expand=True, padx=2, pady=2)

        btn_row = tk.Frame(self, bg=app.BG)
        btn_row.pack(fill="x", padx=16, pady=12)

        buttons = [
            ("[ SAVE .PY ]", self._save_py, app.PHOSPHOR),
            ("[ COPY ]", self._copy, app.PHOSPHOR),
            ("[ PRINT ]", self._print, app.PHOSPHOR_DIM),
            ("[ EXPORT PDF ]", self._export_pdf, app.PHOSPHOR_DIM),
            ("[ EXPORT WORD ]", self._export_docx, app.PHOSPHOR_DIM),
        ]
        for text, cmd, accent in buttons:
            wrap, _btn = term_button(
                btn_row, text, cmd, app.BG, accent, padx=8, pady=6, font=("Consolas", 9, "bold")
            )
            wrap.pack(side="left", padx=(0, 6))

        self.status_label = tk.Label(self, text="", bg=app.BG, fg=app.MUTED_FG, font=("Consolas", 9))
        self.status_label.pack(fill="x", padx=16, pady=(0, 10))

    def _current_text(self):
        return self.text.get("1.0", "end-1c")

    def _save_py(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".py",
            initialfile=f"install_{self.module_name}.py",
            filetypes=[("Python script", "*.py"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._current_text())
        self.status_label.configure(text=f"Saved to {path}")

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._current_text())
        self.status_label.configure(text="Script copied to clipboard.")

    def _print(self):
        text = self._current_text()
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            tmp.write(text)
            tmp.close()
        except Exception as e:
            messagebox.showerror("Print failed", str(e), parent=self)
            return
        ok, message = print_text_file(tmp.name)
        self.status_label.configure(text=message)
        if not ok:
            messagebox.showwarning("Print", message, parent=self)

    def _export_pdf(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".pdf",
            initialfile=f"install_{self.module_name}.pdf",
            filetypes=[("PDF document", "*.pdf")],
        )
        if not path:
            return
        try:
            write_simple_pdf(path, f"Install script — {self.module_name}", self._current_text())
            self.status_label.configure(text=f"Exported PDF to {path}")
        except Exception as e:
            messagebox.showerror("PDF export failed", str(e), parent=self)

    def _export_docx(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".docx",
            initialfile=f"install_{self.module_name}.docx",
            filetypes=[("Word document", "*.docx")],
        )
        if not path:
            return
        try:
            write_simple_docx(path, f"Install script — {self.module_name}", self._current_text())
            self.status_label.configure(text=f"Exported Word doc to {path}")
        except Exception as e:
            messagebox.showerror("Word export failed", str(e), parent=self)


def main():
    app = WeirdoInstaller()
    app.mainloop()


if __name__ == "__main__":
    main()