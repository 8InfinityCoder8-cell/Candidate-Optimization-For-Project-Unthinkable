#!/usr/bin/env python3
"""
Simulated Annealing + Tiny NumPy Neural Network
Enhanced with Continuous Training + Animated Neural Network
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import math
import time
import threading
import queue
from collections import deque
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

PRINTABLE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-0123456789"
CHAR2IDX = {c: i for i, c in enumerate(PRINTABLE)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(PRINTABLE)

# ----------------------------------------------------------------------
# Tiny NumPy Neural Network
# ----------------------------------------------------------------------
class TinyNN:
    def __init__(self, context=3, hidden=64, lr=0.015):
        self.context = context
        self.hidden = hidden
        self.lr = lr
        self.W1 = np.random.randn(context * VOCAB_SIZE, hidden) * 0.08
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, VOCAB_SIZE) * 0.08
        self.b2 = np.zeros((1, VOCAB_SIZE))
        self.loss_history = []
        self.best_loss = float("inf")

    def _onehot(self, chars):
        x = np.zeros((1, self.context * VOCAB_SIZE))
        for i, c in enumerate(chars):
            if c in CHAR2IDX:
                x[0, i * VOCAB_SIZE + CHAR2IDX[c]] = 1.0
        return x

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs, h

    def train_step(self, ctx, tgt):
        x = self._onehot(ctx)
        t = CHAR2IDX.get(tgt, 0)
        probs, h = self.forward(x)
        loss = -np.log(probs[0, t] + 1e-8)

        dlogits = probs.copy()
        dlogits[0, t] -= 1.0
        dW2 = h.T @ dlogits
        db2 = dlogits
        dh = (dlogits @ self.W2.T) * (1 - h**2)
        dW1 = x.T @ dh
        db1 = dh

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return float(loss)

    def train_epoch(self, data):
        random.shuffle(data)
        total = 0.0
        for ctx, tgt in data:
            total += self.train_step(ctx, tgt)
        return total / max(1, len(data))

    def log_prob(self, text):
        if len(text) < self.context + 1:
            return -8.0
        text = text.lower()
        logp = 0.0
        n = 0
        for i in range(len(text) - self.context):
            ctx = text[i:i+self.context]
            tgt = text[i+self.context]
            if any(c not in CHAR2IDX for c in ctx + tgt):
                continue
            x = self._onehot(ctx)
            probs, _ = self.forward(x)
            logp += math.log(probs[0, CHAR2IDX[tgt]] + 1e-8)
            n += 1
        return logp / max(1, n)


NN = TinyNN(context=3, hidden=64, lr=0.015)

CORPUS = """
the quick brown fox jumps over the lazy dog. hello world this is a simple test.
we use simulated annealing with a neural network to create readable text.
python is great for artificial intelligence and machine learning projects.
good morning how are you today. the sun is shining and the weather is nice.
i love learning algorithms. numbers 42 and 123 are common. please generate english.
""".lower()

# Pre-train a little
text_clean = "".join(c for c in CORPUS if c in PRINTABLE)
DATA = [(text_clean[i:i+3], text_clean[i+3]) for i in range(len(text_clean)-3)]
for _ in range(6):
    NN.train_epoch(DATA)
print("Neural Network ready.")


def energy(text, target=""):
    if target:
        if len(text) != len(target):
            return 2000.0 + abs(len(text) - len(target)) * 80
        return float(sum(a != b for a, b in zip(text, target)))
    if not text:
        return 400.0
    score = -NN.log_prob(text) * 14.0
    score += abs(len(text) - 14) * 0.7
    score -= text.count(" ") * 1.5
    for c in text:
        if c not in PRINTABLE:
            score += 8.0
    return score


# ----------------------------------------------------------------------
# Simulated Annealing
# ----------------------------------------------------------------------
class SimulatedAnnealing:
    def __init__(self, binary_seed, target="", initial_temp=32.0, cooling_rate=0.9975, max_iter=18000):
        self.target = target
        self.T = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.min_temp = 0.002
        self.absolute_max_iter = 500000

        seed = int(binary_seed, 2) if binary_seed else random.randint(1, 10**9)
        self.rng = random.Random(seed % (2**32))

        if target:
            self.current = self._noisy(target, 0.55)
        else:
            length = max(9, min(24, len(binary_seed)//3 + 5))
            self.current = "".join(self.rng.choice(PRINTABLE) for _ in range(length))

        self.current_energy = energy(self.current, target)
        self.best = self.current
        self.best_energy = self.current_energy

        self.iteration = 0
        self.stopped = False
        self.finished = False
        self.log_queue = queue.Queue()
        self.last_delta = 0.0
        self.last_prob = 0.0
        self.last_accepted = False
        self.last_nn = NN.log_prob(self.current) if not target else 0.0

        self.history_iter = deque(maxlen=900)
        self.history_current = deque(maxlen=900)
        self.history_best = deque(maxlen=900)
        self.history_temp = deque(maxlen=900)
        self.stuck = 0

    def _noisy(self, text, ratio=0.55):
        chars = list(text)
        for i in self.rng.sample(range(len(chars)), max(1, int(len(chars)*ratio))):
            chars[i] = self.rng.choice([c for c in PRINTABLE if c != chars[i]])
        return "".join(chars)

    def _neighbor(self, s):
        if not s:
            return self.rng.choice(PRINTABLE)
        s = list(s)
        if self.target:
            wrong = [i for i in range(len(s)) if s[i] != self.target[i]]
            if not wrong:
                return "".join(s)
            i = self.rng.choice(wrong) if self.rng.random() < 0.75 else self.rng.randrange(len(s))
            progress = 1.0 - self.best_energy / max(1, len(self.target))
            if self.rng.random() < 0.20 + 0.55 * progress:
                s[i] = self.target[i]
            else:
                s[i] = self.rng.choice(PRINTABLE)
        else:
            act = self.rng.choice(["r"]*6 + ["s", "i", "d"])
            if act == "r":
                i = self.rng.randrange(len(s))
                s[i] = self.rng.choice(PRINTABLE)
            elif act == "s" and len(s) > 1:
                i, j = self.rng.sample(range(len(s)), 2)
                s[i], s[j] = s[j], s[i]
            elif act == "i" and len(s) < 28:
                s.insert(self.rng.randrange(len(s)+1), self.rng.choice(PRINTABLE))
            elif act == "d" and len(s) > 5:
                s.pop(self.rng.randrange(len(s)))
        return "".join(s)

    def run_step(self):
        if self.target and self.best == self.target:
            self.finished = True
            self.log_queue.put(("END", f"=== PERFECT MATCH === '{self.best}'"))
            return True
        if self.iteration >= self.absolute_max_iter or self.stopped:
            self.finished = True
            return True
        if not self.target and (self.iteration >= self.max_iter or self.T < self.min_temp):
            self.finished = True
            return True
        if self.target and self.T < self.min_temp:
            self.stuck += 1
            self.T = self.initial_temp * (0.65 if self.stuck < 8 else 0.85)
            self.log_queue.put(("STEP", f"--- Reheat T={self.T:.3f} ---"))
        else:
            self.stuck = 0

        cand = self._neighbor(self.current)
        ce = energy(cand, self.target)
        delta = ce - self.current_energy
        self.last_delta = delta
        self.last_nn = NN.log_prob(cand) if not self.target else 0.0

        accept = False
        if delta < 0:
            accept = True
            self.last_prob = 1.0
            reason = f"✓ Better ΔE={delta:.1f}"
        else:
            p = math.exp(-delta / max(self.T, 1e-9))
            self.last_prob = p
            if self.rng.random() < p:
                accept = True
                reason = f"✓ Accept P={p:.3f}"
            else:
                reason = f"✗ Reject P={p:.3f}"

        self.last_accepted = accept
        if accept:
            self.current = cand
            self.current_energy = ce
            if ce < self.best_energy:
                self.best = cand
                self.best_energy = ce
                self.log_queue.put(("BEST", f"★ '{self.best}'"))

        self.iteration += 1
        self.T *= self.cooling_rate
        self.history_iter.append(self.iteration)
        self.history_current.append(self.current_energy)
        self.history_best.append(self.best_energy)
        self.history_temp.append(self.T)

        if self.iteration % 20 == 0 or accept or self.iteration < 50:
            self.log_queue.put(("STEP", f"Iter {self.iteration:6d} | T={self.T:6.3f} | {reason}"))
        return False

    def stop(self):
        self.stopped = True
        self.finished = True


# ----------------------------------------------------------------------
# Continuous Training + Animated Neural Network Popup
# ----------------------------------------------------------------------
class ContinuousTrainingWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Continuous Neural Network Training + Animation")
        self.geometry("1050x780")
        self.configure(bg="#0a0e18")
        self.transient(parent)
        self.grab_set()

        self.running = True
        self.epoch = 0
        self.current_loss = 0.0
        self.best_loss = float("inf")
        self.no_improve = 0
        self.phase = 0.0          # for animation

        # Title
        tk.Label(self, text="🧠 Continuous Training Mode – Neural Network is learning non-stop",
                 bg="#0a0e18", fg="#00d4ff", font=("Segoe UI", 13, "bold")).pack(pady=8)

        # Animated network canvas
        self.fig = Figure(figsize=(9.8, 4.5), dpi=100, facecolor="#0a0e18")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0a0e18")
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)

        # Status
        self.status_lbl = tk.Label(self, text="Starting continuous training...",
                                   bg="#0a0e18", fg="#f9e2af", font=("Consolas", 12, "bold"))
        self.status_lbl.pack(pady=3)

        self.loss_lbl = tk.Label(self, text="Current Loss: —   |   Best Loss: —",
                                 bg="#0a0e18", fg="#a6e3a1", font=("Consolas", 11))
        self.loss_lbl.pack()

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=750)
        self.progress.pack(pady=6)
        self.progress.start(12)

        # Log
        logf = ttk.LabelFrame(self, text="Continuous Training Log", padding=4)
        logf.pack(fill="both", expand=True, padx=12, pady=6)
        self.log = scrolledtext.ScrolledText(logf, height=9, state="disabled",
                                             font=("Consolas", 9), bg="#111622", fg="#cdd6f4")
        self.log.pack(fill="both", expand=True)

        # Buttons
        btnf = tk.Frame(self, bg="#0a0e18")
        btnf.pack(pady=8)
        self.stop_btn = ttk.Button(btnf, text="■ Stop Continuous Training",
                                   command=self.stop_training)
        self.stop_btn.pack(side="left", padx=8)
        self.close_btn = ttk.Button(btnf, text="Close Window", state="disabled",
                                    command=self.destroy)
        self.close_btn.pack(side="left", padx=8)

        self.log_msg("=== Continuous Training Started ===")
        self.log_msg("Training will continue until loss stops improving significantly.")
        self.log_msg("Watch the nodes and lines animate while the network learns...\n")

        # Start animation + training
        self.after(100, self.animate)
        threading.Thread(target=self.train_loop, daemon=True).start()

    def log_msg(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def draw_animated_network(self):
        self.ax.clear()
        self.ax.set_facecolor("#0a0e18")
        self.ax.set_xlim(0, 14)
        self.ax.set_ylim(0, 8)
        self.ax.axis("off")

        # Layer definitions
        layers = [
            (1.5, 7, "#ffaa00", "INPUT"),
            (4.0, 9, "#00d4ff", "HIDDEN 1"),
            (6.5, 9, "#00d4ff", "HIDDEN 2"),
            (9.0, 7, "#00d4ff", "HIDDEN 3"),
            (11.5, 6, "#ffd43b", "OUTPUT")
        ]

        node_positions = []
        for idx, (x, n, color, label) in enumerate(layers):
            ys = np.linspace(1.4, 6.6, n)
            pos = []
            for i, y in enumerate(ys):
                # Animation: slight movement + pulsing size
                offset_x = 0.08 * math.sin(self.phase + i * 0.7 + idx)
                offset_y = 0.06 * math.cos(self.phase * 1.3 + i * 0.5)
                size = 0.20 + 0.04 * math.sin(self.phase * 2 + i)
                alpha = 0.75 + 0.25 * abs(math.sin(self.phase + i * 0.4))

                self.ax.add_patch(Circle((x + offset_x, y + offset_y), size,
                                         color=color, alpha=alpha, zorder=3,
                                         ec="white", linewidth=0.7))
                pos.append((x + offset_x, y + offset_y))
            node_positions.append(pos)
            self.ax.text(x, 0.55, label, ha="center", color=color, fontsize=8, fontweight="bold")

        # Animated connections
        for i in range(len(node_positions) - 1):
            for j, (x1, y1) in enumerate(node_positions[i][::2]):
                for k, (x2, y2) in enumerate(node_positions[i+1][::2]):
                    alpha = 0.25 + 0.35 * abs(math.sin(self.phase * 1.5 + j + k))
                    lw = 0.5 + 0.6 * abs(math.sin(self.phase + j * 0.3))
                    self.ax.plot([x1, x2], [y1, y2],
                                 color="#4a7a9a", lw=lw, alpha=alpha, zorder=1)

        title = f"Continuous Training  •  Epoch {self.epoch}  •  Loss {self.current_loss:.4f}"
        self.ax.set_title(title, color="#cdd6f4", fontsize=11, pad=8)
        self.canvas.draw_idle()

    def animate(self):
        if not self.running:
            return
        self.phase += 0.18
        self.draw_animated_network()
        self.after(50, self.animate)   # ~20 FPS animation

    def train_loop(self):
        """Continuous training until loss plateaus"""
        while self.running:
            self.epoch += 1
            loss = NN.train_epoch(DATA)
            self.current_loss = loss

            improved = False
            if loss < self.best_loss - 0.0008:          # meaningful improvement
                self.best_loss = loss
                self.no_improve = 0
                improved = True
            else:
                self.no_improve += 1

            msg = f"Epoch {self.epoch:4d}  →  Loss: {loss:.4f}"
            if improved:
                msg += "  ★ improved"
            self.after(0, lambda m=msg: self.log_msg(m))
            self.after(0, lambda: self.loss_lbl.config(
                text=f"Current Loss: {self.current_loss:.4f}   |   Best Loss: {self.best_loss:.4f}"))
            self.after(0, lambda: self.status_lbl.config(
                text=f"Continuous Training... Epoch {self.epoch}"))

            # Stop conditions
            if self.no_improve >= 18:                   # no improvement for 18 epochs
                self.after(0, lambda: self.log_msg(
                    f"\n=== STOPPED === Loss plateaued at {self.best_loss:.4f}"))
                self.after(0, self.finish)
                break
            if self.best_loss < 1.15:                    # very good loss reached
                self.after(0, lambda: self.log_msg(
                    f"\n=== STOPPED === Excellent loss reached: {self.best_loss:.4f}"))
                self.after(0, self.finish)
                break
            if self.epoch >= 400:                       # safety limit
                self.after(0, lambda: self.log_msg(
                    f"\n=== STOPPED === Safety limit reached (400 epochs)"))
                self.after(0, self.finish)
                break

            time.sleep(0.08)

    def stop_training(self):
        self.running = False
        self.log_msg("\n*** Continuous training stopped by user ***")
        self.finish()

    def finish(self):
        self.running = False
        self.progress.stop()
        self.status_lbl.config(text="Continuous Training Finished", fg="#00ff9f")
        self.stop_btn.config(state="disabled")
        self.close_btn.config(state="normal")
        self.draw_animated_network()


# ----------------------------------------------------------------------
# Other dialogs (same as before)
# ----------------------------------------------------------------------
class LargeBinaryDialog(tk.Toplevel):
    def __init__(self, parent, initial=""):
        super().__init__(parent)
        self.title("Edit Full Binary Seed")
        self.geometry("920x520")
        self.configure(bg="#0b0e17")
        self.transient(parent)
        self.grab_set()
        tk.Label(self, text="Paste / edit long binary seed (0 and 1 only)",
                 bg="#0b0e17", fg="#00d4ff", font=("Segoe UI", 11, "bold")).pack(pady=8)
        self.txt = scrolledtext.ScrolledText(self, font=("Consolas", 12), bg="#111622",
                                             fg="#cdd6f4", insertbackground="#00d4ff", height=18)
        self.txt.pack(fill="both", expand=True, padx=12, pady=6)
        self.txt.insert("1.0", initial)
        self.txt.focus_set()
        fr = tk.Frame(self, bg="#0b0e17")
        fr.pack(pady=10)
        ttk.Button(fr, text="✓ Apply", style="Accent.TButton", command=self.ok).pack(side="left", padx=8)
        ttk.Button(fr, text="Cancel", command=self.destroy).pack(side="left", padx=8)
        self.result = None
        self.wait_window()

    def ok(self):
        c = self.txt.get("1.0", "end").strip().replace(" ", "").replace("\n", "")
        if c and all(x in "01" for x in c):
            self.result = c
            self.destroy()
        else:
            messagebox.showerror("Error", "Only 0 and 1 allowed", parent=self)


class LargeTextDialog(tk.Toplevel):
    def __init__(self, parent, title="Text", initial=""):
        super().__init__(parent)
        self.title(title)
        self.geometry("720x300")
        self.configure(bg="#0b0e17")
        self.transient(parent)
        self.grab_set()
        tk.Label(self, text=title, bg="#0b0e17", fg="#00d4ff", font=("Segoe UI", 12, "bold")).pack(pady=10)
        self.txt = scrolledtext.ScrolledText(self, font=("Consolas", 13), bg="#111622",
                                             fg="#cdd6f4", insertbackground="#00d4ff", height=9)
        self.txt.pack(fill="both", expand=True, padx=15, pady=5)
        self.txt.insert("1.0", initial)
        self.txt.focus_set()
        fr = tk.Frame(self, bg="#0b0e17")
        fr.pack(pady=12)
        ttk.Button(fr, text="✓ Save", style="Accent.TButton", command=self.ok).pack(side="left", padx=8)
        ttk.Button(fr, text="Cancel", command=self.destroy).pack(side="left", padx=8)
        self.result = None
        self.wait_window()

    def ok(self):
        c = self.txt.get("1.0", "end").strip()
        if c:
            self.result = c
            self.destroy()
        else:
            messagebox.showwarning("Empty", "Enter some text", parent=self)


# ----------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Annealing + NumPy Neural Network (Continuous Training)")
        self.root.geometry("1420x1020")
        self.root.minsize(1250, 920)

        style = ttk.Style()
        style.theme_use("clam")
        bg, fg, accent, surface = "#0b0e17", "#cdd6f4", "#00d4ff", "#141a26"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=surface)
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", background=bg, foreground=accent)
        style.configure("TLabelframe.Label", background=bg, foreground=accent, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=surface, foreground=fg, padding=6)
        style.map("TButton", background=[("active", accent)])
        style.configure("Accent.TButton", background=accent, foreground="#0b0e17")
        style.configure("TEntry", fieldbackground=surface, foreground=fg, padding=4)
        style.configure("Horizontal.TProgressbar", troughcolor=surface, background=accent)
        style.configure("TRadiobutton", background=bg, foreground=fg)

        self.targets = ["Yohan Gwapo", "Hello World", "I love Python",
                        "I am you! and you are me..", "Simulated Annealing is powerful"]
        self.sa = None
        self.running = False
        self.start_time = 0.0
        self.active_target = ""
        self.mode = tk.StringVar(value="free")
        self.target_visible = True
        self.binary_var = tk.StringVar()

        self._build()
        self._refresh_list()
        self._on_mode_change()
        self._poll()

    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=6)
        ttk.Label(top, text="Mode:", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Radiobutton(top, text="Free Exploration (SA + Neural Network)", variable=self.mode,
                        value="free", command=self._on_mode_change).pack(side="left", padx=6)
        ttk.Radiobutton(top, text="Target Exact Match", variable=self.mode,
                        value="target", command=self._on_mode_change).pack(side="left", padx=6)
        self.hide_btn = ttk.Button(top, text="Hide Target List", command=self.toggle_target_list)
        self.hide_btn.pack(side="right")

        self.target_frame = ttk.LabelFrame(self.root, text="  Target Text List  ", padding=6)
        self.target_frame.pack(fill="x", padx=10, pady=3)
        lf = ttk.Frame(self.target_frame)
        lf.pack(side="left", fill="both", expand=True)
        self.listbox = tk.Listbox(lf, height=3, font=("Consolas", 12), bg="#111622", fg="#00d4ff",
                                  selectbackground="#00d4ff", selectforeground="#0b0e17", activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        bt = ttk.Frame(self.target_frame)
        bt.pack(side="left", padx=8)
        ttk.Button(bt, text="➕ Add", width=14, command=self.add_target).pack(fill="x", pady=2)
        ttk.Button(bt, text="✏️ Edit", width=14, command=self.edit_target).pack(fill="x", pady=2)
        ttk.Button(bt, text="🗑️ Remove", width=14, command=self.remove_target).pack(fill="x", pady=2)
        self.prepare_btn = ttk.Button(bt, text="➡️ PREPARE TARGET", style="Accent.TButton",
                                      width=14, command=self.prepare_target)
        self.prepare_btn.pack(fill="x", pady=5)

        bf = ttk.LabelFrame(self.root, text="  Binary Seed + Parameters  ", padding=6)
        bf.pack(fill="x", padx=10, pady=3)
        br = ttk.Frame(bf)
        br.pack(fill="x", pady=3)
        ttk.Label(br, text="Binary preview:").pack(side="left")
        ttk.Entry(br, textvariable=self.binary_var, font=("Consolas", 11)).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(br, text="📝 Edit Full Binary", style="Accent.TButton",
                   command=self.open_binary_editor).pack(side="left", padx=4)

        pr = ttk.Frame(bf)
        pr.pack(fill="x", pady=4)
        ttk.Label(pr, text="Init T:").grid(row=0, column=0)
        self.temp_var = tk.StringVar(value="32.0")
        ttk.Entry(pr, textvariable=self.temp_var, width=9).grid(row=0, column=1, padx=4)
        ttk.Label(pr, text="Cooling:").grid(row=0, column=2, padx=(10,0))
        self.cool_var = tk.StringVar(value="0.9975")
        ttk.Entry(pr, textvariable=self.cool_var, width=9).grid(row=0, column=3, padx=4)
        ttk.Label(pr, text="Max Iter:").grid(row=0, column=4, padx=(10,0))
        self.iter_var = tk.StringVar(value="18000")
        ttk.Entry(pr, textvariable=self.iter_var, width=9).grid(row=0, column=5, padx=4)

        br2 = ttk.Frame(bf)
        br2.pack(fill="x", pady=4)
        self.start_btn = ttk.Button(br2, text="▶ START ANNEALING", style="Accent.TButton", width=20, command=self.start)
        self.start_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(br2, text="■ Stop", width=10, command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=3)
        self.reset_btn = ttk.Button(br2, text="↺ Reset", width=10, command=self.reset)
        self.reset_btn.pack(side="left", padx=3)

        # NEW: Continuous Training button
        ttk.Button(br2, text="🧠 Continuous Training (Animated)", style="Accent.TButton",
                   command=self.open_continuous_training).pack(side="left", padx=12)

        sf = ttk.LabelFrame(self.root, text="  LIVE CANDIDATE  ", padding=6)
        sf.pack(fill="x", padx=10, pady=3)
        self.current_lbl = tk.Label(sf, text="Ready – press START", font=("Consolas", 16, "bold"),
                                    bg="#0a0e18", fg="#00d4ff", pady=8, wraplength=1300, justify="left")
        self.current_lbl.pack(fill="x")
        br3 = tk.Frame(sf, bg="#0a0e18")
        br3.pack(fill="x")
        tk.Label(br3, text="BEST →", font=("Segoe UI", 9, "bold"), bg="#0a0e18", fg="#00ff9f").pack(side="left", padx=5)
        self.best_lbl = tk.Label(br3, text="—", font=("Consolas", 13, "bold"), bg="#0a0e18", fg="#00ff9f",
                                 wraplength=1200, justify="left")
        self.best_lbl.pack(side="left")
        info = ttk.Frame(sf)
        info.pack(fill="x", pady=2)
        self.energy_lbl = ttk.Label(info, text="Energy: —")
        self.energy_lbl.pack(side="left", padx=4)
        self.target_lbl = ttk.Label(info, text="Mode: Free (SA + NN)", foreground="#00d4ff")
        self.target_lbl.pack(side="left", padx=10)
        self.math_lbl = ttk.Label(info, text="", foreground="#f9e2af")
        self.math_lbl.pack(side="left", padx=10)
        self.nn_lbl = ttk.Label(info, text="", foreground="#a6e3a1")
        self.nn_lbl.pack(side="left", padx=10)

        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=4)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        st = ttk.LabelFrame(left, text="Status", padding=5)
        st.pack(fill="x")
        self.temp_lbl = ttk.Label(st, text="Temperature: —")
        self.temp_lbl.pack(anchor="w")
        self.iter_lbl = ttk.Label(st, text="Iteration: 0")
        self.iter_lbl.pack(anchor="w")
        self.time_lbl = ttk.Label(st, text="Elapsed: 0.00 s")
        self.time_lbl.pack(anchor="w")
        self.progress = ttk.Progressbar(st, mode="determinate")
        self.progress.pack(fill="x", pady=3)

        logf = ttk.LabelFrame(left, text="Main Log", padding=3)
        logf.pack(fill="both", expand=True, pady=(4, 0))
        self.log = scrolledtext.ScrolledText(logf, height=11, state="disabled",
                                             font=("Consolas", 9), bg="#0a0e18", fg="#cdd6f4")
        self.log.pack(fill="both", expand=True)

        right = ttk.Frame(mid)
        right.pack(side="right", fill="both", expand=True)

        pf1 = ttk.LabelFrame(right, text="Energy", padding=2)
        pf1.pack(fill="both", expand=True)
        self.fig1 = Figure(figsize=(5.4, 2.0), dpi=100, facecolor="#0b0e17")
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor("#0a0e18")
        self.ax1.tick_params(colors="#cdd6f4", labelsize=6)
        self.line_c, = self.ax1.plot([], [], color="#ff6b9d", lw=1.1, label="Current")
        self.line_b, = self.ax1.plot([], [], color="#00ff9f", lw=1.5, label="Best")
        self.ax1.legend(facecolor="#141a26", labelcolor="#cdd6f4", fontsize=6)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=pf1)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

        pf2 = ttk.LabelFrame(right, text="Temperature", padding=2)
        pf2.pack(fill="both", expand=True, pady=(3, 0))
        self.fig2 = Figure(figsize=(5.4, 1.8), dpi=100, facecolor="#0b0e17")
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor("#0a0e18")
        self.ax2.tick_params(colors="#cdd6f4", labelsize=6)
        self.line_t, = self.ax2.plot([], [], color="#00d4ff", lw=1.4)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=pf2)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

    def open_continuous_training(self):
        ContinuousTrainingWindow(self.root)

    def open_binary_editor(self):
        d = LargeBinaryDialog(self.root, self.binary_var.get())
        if d.result is not None:
            self.binary_var.set(d.result)
            self._log(f"Binary updated ({len(d.result)} bits)")

    def add_target(self):
        d = LargeTextDialog(self.root, "Add Target")
        if d.result:
            self.targets.append(d.result)
            self._refresh_list()
            self.listbox.selection_set(tk.END)

    def edit_target(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a target")
            return
        idx = sel[0]
        d = LargeTextDialog(self.root, "Edit Target", self.targets[idx])
        if d.result:
            self.targets[idx] = d.result
            self._refresh_list()
            self.listbox.selection_set(idx)

    def remove_target(self):
        sel = self.listbox.curselection()
        if sel:
            del self.targets[sel[0]]
            self._refresh_list()

    def prepare_target(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a target")
            return
        t = self.targets[sel[0]]
        binary = "".join(format(ord(c), "08b") for c in t)
        self.binary_var.set(binary)
        self.active_target = t
        self.target_lbl.config(text=f"Target: \"{t[:50]}{'...' if len(t)>50 else ''}\"")
        self.current_lbl.config(text="Ready – press START")
        self._log(f"Prepared: \"{t}\" ({len(binary)} bits)")

    def _on_mode_change(self):
        if self.mode.get() == "free":
            self.prepare_btn.config(state="disabled")
            self.target_lbl.config(text="Mode: Free Exploration (SA + Neural Network)")
            self.active_target = ""
        else:
            self.prepare_btn.config(state="normal")

    def toggle_target_list(self):
        if self.target_visible:
            self.target_frame.pack_forget()
            self.hide_btn.config(text="Show Target List")
            self.target_visible = False
        else:
            self.target_frame.pack(fill="x", padx=10, pady=3, after=self.root.winfo_children()[0])
            self.hide_btn.config(text="Hide Target List")
            self.target_visible = True

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for t in self.targets:
            self.listbox.insert(tk.END, t)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_plots(self):
        if not self.sa or not self.sa.history_iter:
            return
        x = list(self.sa.history_iter)
        yc = list(self.sa.history_current)
        yb = list(self.sa.history_best)
        yt = list(self.sa.history_temp)
        m = min(len(x), len(yc), len(yb), len(yt))
        if m == 0:
            return
        x, yc, yb, yt = x[:m], yc[:m], yb[:m], yt[:m]
        self.line_c.set_data(x, yc)
        self.line_b.set_data(x, yb)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.draw_idle()
        self.line_t.set_data(x, yt)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas2.draw_idle()

    def _poll(self):
        try:
            while True:
                k, msg = self.sa.log_queue.get_nowait() if self.sa else (None, None)
                if k is None:
                    break
                self._log(msg)
        except queue.Empty:
            pass

        if self.running and self.sa:
            self.current_lbl.config(text=self.sa.current)
            self.best_lbl.config(text=self.sa.best)
            self.energy_lbl.config(text=f"Energy: {self.sa.current_energy:.1f} (best {self.sa.best_energy:.1f})")
            self.temp_lbl.config(text=f"Temperature: {self.sa.T:.4f}")
            self.iter_lbl.config(text=f"Iteration: {self.sa.iteration}")
            self.time_lbl.config(text=f"Elapsed: {time.perf_counter()-self.start_time:.1f} s")
            mx = self.sa.max_iter if not self.sa.target else self.sa.absolute_max_iter
            self.progress["value"] = min(100, 100*self.sa.iteration/max(1, mx))
            d, p = self.sa.last_delta, self.sa.last_prob
            acc = "ACCEPTED" if self.sa.last_accepted else "REJECTED"
            self.math_lbl.config(text=f"ΔE={d:.1f}  P={p:.3f} → {acc}")
            if not self.sa.target:
                self.nn_lbl.config(text=f"NN logP: {self.sa.last_nn:.3f}")
            else:
                self.nn_lbl.config(text="")
            if self.sa.iteration % 12 == 0:
                self._update_plots()
        elif self.sa and self.sa.finished:
            self.current_lbl.config(text=self.sa.best)
            self.best_lbl.config(text=self.sa.best)
            self.math_lbl.config(text="Finished")

        self.root.after(30, self._poll)

    def start(self):
        binary = self.binary_var.get().strip()
        mode = self.mode.get()
        if mode == "target" and not self.active_target:
            messagebox.showerror("Error", "Select target and PREPARE first")
            return
        if mode == "free" and (not binary or any(c not in "01" for c in binary)):
            binary = bin(random.randint(10**10, 10**16))[2:]
            self.binary_var.set(binary)
        try:
            T0 = float(self.temp_var.get())
            cool = float(self.cool_var.get())
            mit = int(self.iter_var.get())
        except:
            messagebox.showerror("Error", "Invalid parameters")
            return

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        target = self.active_target if mode == "target" else ""
        self.sa = SimulatedAnnealing(binary, target, T0, cool, mit)
        self.running = True
        self.start_time = time.perf_counter()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        if target:
            self._log("=== Target Exact Match ===")
            self._log(f"Target: \"{target}\"")
        else:
            self._log("=== Free Exploration (SA + Neural Network) ===")
            self._log("Neural Network is scoring the candidates...")
        self._log(f"Start: '{self.sa.current}'")

        def worker():
            while self.running and not self.sa.stopped:
                if self.sa.run_step():
                    break
                time.sleep(0.002)
            self.running = False
            self.sa.finished = True
            if not (self.sa.target and self.sa.best == self.sa.target):
                self.sa.log_queue.put(("END", f"=== FINISHED === '{self.sa.best}'"))

        threading.Thread(target=worker, daemon=True).start()

        def check():
            if self.running:
                self.root.after(200, check)
            else:
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
                self._update_plots()
                if self.sa:
                    self.current_lbl.config(text=self.sa.best)
                    self.best_lbl.config(text=self.sa.best)
        self.root.after(200, check)

    def stop(self):
        if self.sa:
            self.sa.stop()
        self.running = False
        self._log("*** Stopped ***")

    def reset(self):
        self.stop()
        self.sa = None
        self.current_lbl.config(text="Ready – press START")
        self.best_lbl.config(text="—")
        self.energy_lbl.config(text="Energy: —")
        self.temp_lbl.config(text="Temperature: —")
        self.iter_lbl.config(text="Iteration: 0")
        self.time_lbl.config(text="Elapsed: 0.00 s")
        self.math_lbl.config(text="")
        self.nn_lbl.config(text="")
        self.progress["value"] = 0
        self.line_c.set_data([], [])
        self.line_b.set_data([], [])
        self.line_t.set_data([], [])
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")



if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()