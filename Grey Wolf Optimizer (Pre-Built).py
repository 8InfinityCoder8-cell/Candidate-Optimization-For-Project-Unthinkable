"""
Crisis Budget Optimizer - Grey Wolf Optimization Laboratory   (SINGLE-FILE EDITION)

An educational desktop simulator that runs the real, standard Grey Wolf Optimizer (GWO)
on a crisis budget allocation problem, with live charts and a mathematical log.
NOT official emergency guidance or an approved budget.

Install :  pip install numpy matplotlib
Run GUI :  python crisis_budget_optimizer.py
Run tests: python crisis_budget_optimizer.py --test

File layout (search for the banner lines):
  1. BUDGET MODEL      fitness, constraint repair, verification, presets
  2. GWO ALGORITHM     the optimizer, snapshots, run control
  3. VALIDATION        user-input checks
  4. EXPORT            CSV + JSON
  5. VISUALIZATIONS    Matplotlib charts + mathematical log text
  6. EDUCATION         text of the "Learn GWO" tab
  7. GUI               Tkinter application
  8. TESTS             unittest suite (run with --test)
"""
from __future__ import annotations
import csv
import json
import math
import os
import queue
import sys
import textwrap
import threading
import time
import unittest




from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ==========================================================================
# 1. BUDGET MODEL
# ==========================================================================

# Absolute tolerance in pesos for floating-point comparisons (constraints).
TOL = 1e-6


class RepairError(Exception):
    """Raised when a candidate cannot be turned into a feasible allocation."""


@dataclass(frozen=True)
class Sector:
    name: str
    target: float      # di  target funding need (pesos)
    weight: float      # wi  priority weight
    min_alloc: float   # li
    max_alloc: float   # ui


def peso(value: float) -> str:
    """Consistent currency formatting used everywhere in the application."""
    return f"\u20b1{value:,.2f}"


class BudgetProblem:
    """Holds the problem data as NumPy arrays and implements fitness + repair."""

    def __init__(self, budget: float, sectors: List[Sector]):
        self.budget = float(budget)
        self.sectors = tuple(sectors)
        self.names = [s.name for s in sectors]
        self.n = len(sectors)
        self.targets = np.array([s.target for s in sectors], dtype=float)
        self.weights = np.array([s.weight for s in sectors], dtype=float)
        self.lower = np.array([s.min_alloc for s in sectors], dtype=float)
        self.upper = np.array([s.max_alloc for s in sectors], dtype=float)
        # Cap at target (no credit above it) but never below the user's minimum.
        self.effective_upper = np.maximum(
            self.lower, np.minimum(self.upper, self.targets))

    # ------------------------------------------------------------------ fitness
    def coverage(self, x) -> np.ndarray:
        """Per-sector coverage min(xi/di, 1) in [0, 1]. Works on 1-D or 2-D input."""
        x = np.asarray(x, dtype=float)
        safe = np.where(self.targets > 0, self.targets, 1.0)
        cov = np.where(self.targets > 0, x / safe, 1.0)
        return np.clip(cov, 0.0, 1.0)

    def fitness(self, x) -> np.ndarray | float:
        """F(X) = sum wi * min(xi/di, 1). 2-D input (pop, n) returns a (pop,) array."""
        result = np.sum(self.weights * self.coverage(x), axis=-1)
        return float(result) if np.ndim(result) == 0 else result

    @property
    def max_fitness(self) -> float:
        """Upper bound of F: every sector fully covered (= sum of weights)."""
        return float(self.weights.sum())

    def normalized_fitness(self, fitness: float) -> float:
        """Weighted coverage in [0,1]. Same ranking as F, easier to interpret."""
        return float(fitness) / self.max_fitness if self.max_fitness > 0 else 0.0

    # ----------------------------------------------------------- feasibility
    def inputs_feasible(self) -> bool:
        return bool(np.all(self.lower <= self.upper + TOL)
                    and self.lower.sum() <= self.budget + TOL)

    def is_feasible(self, x) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(np.all(np.isfinite(x))
                    and np.all(x >= -TOL)
                    and np.all(x >= self.lower - TOL)
                    and np.all(x <= self.upper + TOL)
                    and x.sum() <= self.budget + TOL)

    # ----------------------------------------------------------------- repair
    def repair(self, x) -> np.ndarray:
        """
        Turn an arbitrary real vector into a feasible allocation.

        1. Clip every variable into [li, hi_i]          (bounds, nonnegativity)
        2. If the total exceeds the budget B, shrink only the part ABOVE each
           minimum, proportionally:
               xi = li + (xi - li) * (B - sum(l)) / (sum(x) - sum(l))
           Because the minimums alone fit in B, the result sums to exactly B and
           every xi stays >= li and <= hi_i. (Plain clipping alone would NOT keep
           the sum within B -- hence this step.)
        3. A tiny floating-point guard removes any residual excess.
        4. Final verification; raises RepairError instead of returning a
           candidate that is not feasible.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise RepairError(f"Expected a vector of length {self.n}, got {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise RepairError("Candidate contains NaN or infinite values.")
        if not self.inputs_feasible():
            raise RepairError("Problem is infeasible: minimum allocations exceed the budget.")

        lo, hi = self.lower, self.effective_upper
        y = np.clip(x, lo, hi)
        if y.sum() > self.budget:
            allowed = self.budget - lo.sum()          # >= 0 (inputs feasible)
            spare = y - lo
            spare_total = spare.sum()
            y = lo + spare * (allowed / spare_total) if spare_total > 0 else lo.copy()
        for _ in range(3):                             # floating-point guard
            excess = y.sum() - self.budget
            if excess <= 0:
                break
            spare = y - lo
            s = spare.sum()
            if s <= 0:
                break
            y = y - spare * min(1.0, excess / s)
        y = np.clip(y, lo, hi)
        if not self.is_feasible(y):
            raise RepairError("Repair failed to produce a feasible allocation.")
        return y

    def repair_population(self, X) -> np.ndarray:
        return np.vstack([self.repair(row) for row in np.asarray(X, dtype=float)])

    # ------------------------------------------------------- verification
    def verify(self, x, reported_fitness: Optional[float] = None) -> List[Tuple[str, bool, str]]:
        """Real constraint checks for the results checklist: (label, passed, detail)."""
        x = np.asarray(x, dtype=float)
        checks = [
            ("Every allocation is at or above its minimum",
             bool(np.all(x >= self.lower - TOL)), ""),
            ("Every allocation is at or below its maximum",
             bool(np.all(x <= self.upper + TOL)), ""),
            ("Total allocation is within the available budget",
             bool(x.sum() <= self.budget + TOL),
             f"{peso(x.sum())} of {peso(self.budget)}"),
            ("Every allocation is nonnegative", bool(np.all(x >= -TOL)), ""),
        ]
        if reported_fitness is not None:
            recomputed = self.fitness(x)
            checks.append(("Displayed values match the selected best candidate",
                           bool(abs(recomputed - reported_fitness) <= 1e-9),
                           f"recomputed F = {recomputed:.6f}"))
        return checks

    # ----------------------------------------------------------- reporting
    def sector_report(self, x) -> List[dict]:
        """One row per sector (shared by GUI table, CSV and JSON exports)."""
        x = np.asarray(x, dtype=float)
        total = float(x.sum())
        cov = self.coverage(x)
        rows = []
        for i, s in enumerate(self.sectors):
            rows.append({
                "sector": s.name, "target": s.target, "weight": s.weight,
                "min": s.min_alloc, "max": s.max_alloc,
                "allocation": float(x[i]),
                "coverage_pct": float(cov[i] * 100.0),
                "share_of_allocated_pct": float(x[i] / total * 100.0) if total > 0 else 0.0,
                "shortfall": float(max(s.target - x[i], 0.0)),
            })
        return rows


# --------------------------------------------------------------------- presets
# Educational simulation presets (NOT official response plans). Fully editable.
# Row format: (name, target need, priority weight, minimum, maximum), in pesos.
_STD = [
    ("Food and water", 1_500_000, 9, 300_000, 1_800_000),
    ("Medical supplies", 1_200_000, 10, 250_000, 1_500_000),
    ("Emergency shelter", 1_800_000, 8, 200_000, 2_200_000),
    ("Evacuation and transportation", 900_000, 7, 100_000, 1_200_000),
    ("Contingency reserve", 600_000, 4, 100_000, 800_000),
]

PRESETS = {
    "Typhoon and flood response": (5_000_000, _STD),
    "Earthquake emergency relief": (8_000_000, [
        ("Search and rescue", 2_000_000, 10, 400_000, 2_500_000),
        ("Medical supplies", 2_200_000, 10, 500_000, 2_800_000),
        ("Emergency shelter", 2_800_000, 8, 400_000, 3_200_000),
        ("Food and water", 1_800_000, 9, 300_000, 2_200_000),
        ("Contingency reserve", 1_000_000, 4, 150_000, 1_200_000)]),
    "Hospital emergency allocation": (3_000_000, [
        ("Emergency room supplies", 900_000, 10, 150_000, 1_100_000),
        ("Medicines and oxygen", 1_000_000, 10, 200_000, 1_200_000),
        ("Staff overtime and support", 800_000, 7, 100_000, 1_000_000),
        ("Equipment repair", 500_000, 5, 50_000, 600_000),
        ("Contingency reserve", 300_000, 3, 50_000, 400_000)]),
    "Food and water shortage": (2_500_000, [
        ("Food and water", 1_600_000, 10, 300_000, 1_800_000),
        ("Medical supplies", 400_000, 7, 50_000, 500_000),
        ("Distribution and transportation", 700_000, 8, 100_000, 900_000),
        ("Storage and logistics", 300_000, 5, 50_000, 400_000),
        ("Contingency reserve", 300_000, 4, 50_000, 400_000)]),
    "Custom crisis scenario": (4_000_000, _STD),
}


# ==========================================================================
# 2. GWO ALGORITHM
# ==========================================================================

IMPROVEMENT_EPS = 1e-12   # smallest fitness gain counted as an improvement


@dataclass(frozen=True)
class GWOConfig:
    population_size: int
    max_iterations: int
    seed: Optional[int] = None      # None -> nondeterministic run
    patience: int = 0               # stop early after this many iterations without
                                    # improvement of best-so-far; 0 disables it


def _ro(array) -> np.ndarray:
    """Read-only copy so snapshots are immutable and safe to share across threads."""
    out = np.array(array, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def rank_wolves(fitness) -> np.ndarray:
    """Indices sorted best-first for MAXIMISATION. Ties: lower index wins (deterministic)."""
    f = np.asarray(fitness, dtype=float)
    return np.lexsort((np.arange(f.size), -f))


@dataclass(frozen=True)
class IterationSnapshot:
    iteration: int
    a: Optional[float]                 # None for iteration 0
    positions: np.ndarray              # (pop, n) repaired positions after this step
    fitness: np.ndarray                # (pop,)
    alpha_index: int
    beta_index: int
    delta_index: int
    alpha_fitness: float
    beta_fitness: float
    delta_fitness: float
    alpha_position: np.ndarray
    beta_position: np.ndarray
    delta_position: np.ndarray
    best_so_far_fitness: float
    best_so_far_position: np.ndarray
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    feasible_count: int
    all_feasible: bool
    details: Optional[Dict[str, np.ndarray]]   # update internals (None at iteration 0)


@dataclass
class GWOResult:
    status: str                         # "completed" | "early_stop" | "stopped"
    best_position: np.ndarray
    best_fitness: float
    best_iteration: int
    initial_best_fitness: float
    iterations_completed: int           # updates actually performed
    runtime_seconds: float
    snapshots: List[IterationSnapshot]
    config: GWOConfig
    stop_reason: str

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "early_stop")


class RunControl:
    """Thread-safe pause / stop / speed control shared by the GUI and the worker."""

    def __init__(self, delay: float = 0.0):
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self.delay = delay            # seconds slept after each iteration (visual pacing)

    def stop(self):
        self._stop.set()
        self._running.set()           # release a paused worker so it can exit

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self) -> bool:
        """Called by the worker after each snapshot. Returns False if it must stop."""
        while not self._running.wait(0.05):
            if self._stop.is_set():
                return False
        waited = 0.0
        while waited < self.delay and not self._stop.is_set():
            time.sleep(0.01)
            waited += 0.01
        return not self._stop.is_set()


class GreyWolfOptimizer:
    def __init__(self, problem: BudgetProblem, config: GWOConfig):
        if config.population_size < 3:
            raise ValueError("GWO needs at least 3 wolves (alpha, beta, delta).")
        if config.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if not problem.inputs_feasible():
            raise ValueError("Problem inputs are infeasible; refusing to optimise.")
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.early_stopped = False

    # ---------------------------------------------------------------- helpers
    def _snapshot(self, it, a, pos, fit, order, best_fit, best_pos, details):
        a_i, b_i, d_i = (int(k) for k in order[:3])
        feasible = sum(self.problem.is_feasible(row) for row in pos)
        return IterationSnapshot(
            iteration=it, a=a, positions=_ro(pos), fitness=_ro(fit),
            alpha_index=a_i, beta_index=b_i, delta_index=d_i,
            alpha_fitness=float(fit[a_i]), beta_fitness=float(fit[b_i]),
            delta_fitness=float(fit[d_i]),
            alpha_position=_ro(pos[a_i]), beta_position=_ro(pos[b_i]),
            delta_position=_ro(pos[d_i]),
            best_so_far_fitness=float(best_fit), best_so_far_position=_ro(best_pos),
            mean_fitness=float(fit.mean()), std_fitness=float(fit.std()),
            min_fitness=float(fit.min()), max_fitness=float(fit.max()),
            feasible_count=int(feasible), all_feasible=bool(feasible == len(pos)),
            details=None if details is None else {k: _ro(v) for k, v in details.items()},
        )

    # -------------------------------------------------------------- main loop
    def iterate(self):
        """Generator yielding one IterationSnapshot per step (0 = initialisation)."""
        p, cfg, rng = self.problem, self.config, self.rng
        pop, n, T = cfg.population_size, p.n, cfg.max_iterations

        # Step 1: random initial population inside [l, hi], then repaired -> feasible.
        raw = p.lower + rng.random((pop, n)) * (p.effective_upper - p.lower)
        pos = p.repair_population(raw)
        # Step 2-3: evaluate and rank.
        fit = p.fitness(pos)
        order = rank_wolves(fit)
        best_fit, best_pos = float(fit[order[0]]), pos[order[0]].copy()
        stagnant = 0
        yield self._snapshot(0, None, pos, fit, order, best_fit, best_pos, None)

        for t in range(T):                                   # t is zero-indexed
            a = 2.0 - 2.0 * t / T                            # Step 4
            leaders = pos[order[:3]].copy()                  # alpha, beta, delta (current)
            leader_fit = fit[order[:3]].copy()
            A_all = np.empty((3, pop, n)); C_all = np.empty((3, pop, n))
            D_all = np.empty((3, pop, n)); X_all = np.empty((3, pop, n))
            for k in range(3):
                r1 = rng.random((pop, n))
                r2 = rng.random((pop, n))
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                D = np.abs(C * leaders[k] - pos)             # Step 5 (all wolves at once)
                A_all[k], C_all[k], D_all[k] = A, C, D
                X_all[k] = leaders[k] - A * D
            X_new = X_all.mean(axis=0)                       # (X1 + X2 + X3) / 3
            try:
                X_feas = p.repair_population(X_new)          # Step 6
            except RepairError as exc:
                raise RuntimeError(f"Iteration {t + 1}: {exc}") from exc

            prev_pos, prev_fit = pos, fit
            pos = X_feas
            fit = p.fitness(pos)                             # Step 7
            order = rank_wolves(fit)

            if fit[order[0]] > best_fit + IMPROVEMENT_EPS:   # best-so-far archive
                best_fit, best_pos = float(fit[order[0]]), pos[order[0]].copy()
                stagnant = 0
            else:
                stagnant += 1

            details = {
                "previous_positions": prev_pos, "previous_fitness": prev_fit,
                "leader_positions": leaders, "leader_fitness": leader_fit,
                "A": A_all, "C": C_all, "D": D_all, "X_guided": X_all,
                "X_new": X_new, "X_feasible": X_feas,
            }
            yield self._snapshot(t + 1, a, pos, fit, order, best_fit, best_pos, details)

            if cfg.patience > 0 and stagnant >= cfg.patience:   # optional stopping rule
                self.early_stopped = True
                return

    def run(self, callback: Optional[Callable[[IterationSnapshot], None]] = None,
            control: Optional[RunControl] = None) -> GWOResult:
        """Run to completion (or until `control` asks to stop). Blocking."""
        start = time.perf_counter()
        snaps: List[IterationSnapshot] = []
        status, reason = None, ""
        for snap in self.iterate():
            snaps.append(snap)
            if callback:
                callback(snap)
            if control is not None and not control.checkpoint():
                status, reason = "stopped", "Stopped by user before the iteration limit."
                break
        if status is None:
            if self.early_stopped:
                status = "early_stop"
                reason = f"No improvement for {self.config.patience} iterations (patience rule)."
            else:
                status, reason = "completed", "Reached the maximum number of iterations."
        last = snaps[-1]
        best_iter = next(s.iteration for s in snaps
                         if s.best_so_far_fitness >= last.best_so_far_fitness - IMPROVEMENT_EPS)
        return GWOResult(
            status=status, best_position=np.array(last.best_so_far_position),
            best_fitness=last.best_so_far_fitness, best_iteration=best_iter,
            initial_best_fitness=snaps[0].best_so_far_fitness,
            iterations_completed=last.iteration,
            runtime_seconds=time.perf_counter() - start,
            snapshots=snaps, config=self.config, stop_reason=reason)


# ==========================================================================
# 3. VALIDATION
# ==========================================================================

POP_MIN, POP_MAX = 5, 100
ITER_MIN, ITER_MAX = 5, 500
MIN_SECTORS, MAX_SECTORS = 2, 12
SEED_MAX = 2**32 - 1


def parse_number(text, field_name: str) -> float:
    """Parse a finite number; accepts '1,500,000', '₱1500000'. Raises ValueError."""
    raw = str(text).strip().replace(",", "").replace("\u20b1", "").replace("PHP", "").strip()
    if raw == "":
        raise ValueError(f"{field_name}: a value is required.")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{field_name}: '{text}' is not a valid number.") from None
    if not math.isfinite(value):
        raise ValueError(f"{field_name}: value must be finite (not NaN or infinity).")
    return value


def validate_budget(text) -> float:
    value = parse_number(text, "Available budget")
    if value <= 0:
        raise ValueError("Available budget: must be greater than zero.")
    return value


def parse_int(text, field_name: str, lo: int, hi: int) -> int:
    value = parse_number(text, field_name)
    if value != int(value):
        raise ValueError(f"{field_name}: must be a whole number.")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{field_name}: must be between {lo} and {hi}.")
    return int(value)


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    problem: Optional[BudgetProblem] = None
    config: Optional[GWOConfig] = None

    @property
    def ok(self) -> bool:
        return not self.errors and self.problem is not None and self.config is not None


def validate_sectors(rows: Sequence[Sequence], budget: Optional[float],
                     errors: List[str], warnings: List[str]) -> List[Sector]:
    """rows: (name, target, weight, min, max) as raw text/number values."""
    sectors: List[Sector] = []
    if not MIN_SECTORS <= len(rows) <= MAX_SECTORS:
        errors.append(f"Number of sectors must be between {MIN_SECTORS} and {MAX_SECTORS} "
                      f"(currently {len(rows)}).")
    seen = set()
    for idx, row in enumerate(rows, start=1):
        name = str(row[0]).strip()
        label = f"Sector {idx} ('{name}')" if name else f"Sector {idx}"
        if not name:
            errors.append(f"Sector {idx}: name must not be empty.")
        elif name.lower() in seen:
            errors.append(f"{label}: duplicate sector name.")
        seen.add(name.lower())
        try:
            target = parse_number(row[1], f"{label} target need")
            weight = parse_number(row[2], f"{label} priority weight")
            lo = parse_number(row[3], f"{label} minimum")
            hi = parse_number(row[4], f"{label} maximum")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        ok = True
        if target < 0:
            errors.append(f"{label}: target need cannot be negative."); ok = False
        if weight <= 0:
            errors.append(f"{label}: priority weight must be greater than zero."); ok = False
        if lo < 0:
            errors.append(f"{label}: minimum allocation cannot be negative."); ok = False
        if hi < 0:
            errors.append(f"{label}: maximum allocation cannot be negative."); ok = False
        if ok and lo > hi:
            errors.append(f"{label}: minimum ({peso(lo)}) exceeds maximum ({peso(hi)})."); ok = False
        if ok and target == 0:
            warnings.append(f"{label}: target need is 0, so it is treated as fully covered.")
        if ok and lo > target > 0:
            warnings.append(f"{label}: minimum exceeds the target need; the minimum still applies.")
        if ok:
            sectors.append(Sector(name, target, weight, lo, hi))

    if budget is not None and len(sectors) == len(rows) and sectors:
        total_min = sum(s.min_alloc for s in sectors)
        if total_min > budget:
            errors.append(
                f"Infeasible: the minimum allocations add up to {peso(total_min)}, which is more "
                f"than the available budget of {peso(budget)}. Lower some minimums or raise the budget.")
        total_target = sum(s.target for s in sectors)
        if budget >= total_target:
            warnings.append("The budget covers every target need; some funds may stay unallocated.")
        else:
            warnings.append(f"The budget ({peso(budget)}) is below the total target need "
                            f"({peso(total_target)}); trade-offs will be made.")
    return sectors


def validate_config(pop, iters, seed, patience, errors: List[str]) -> Optional[GWOConfig]:
    n_err = len(errors)
    try:
        population = parse_int(pop, "Population size", POP_MIN, POP_MAX)
    except ValueError as e:
        errors.append(str(e)); population = None
    try:
        iterations = parse_int(iters, "Maximum iterations", ITER_MIN, ITER_MAX)
    except ValueError as e:
        errors.append(str(e)); iterations = None
    seed_value = None
    if str(seed).strip() != "":
        try:
            seed_value = parse_int(seed, "Random seed", 0, SEED_MAX)
        except ValueError as e:
            errors.append(str(e))
    try:
        patience_value = parse_int(patience if str(patience).strip() else "0",
                                   "Early-stop patience", 0, ITER_MAX)
    except ValueError as e:
        errors.append(str(e)); patience_value = None
    if len(errors) > n_err:
        return None
    return GWOConfig(population, iterations, seed_value, patience_value)


def validate_all(budget_text, sector_rows, pop, iters, seed, patience) -> ValidationResult:
    result = ValidationResult()
    budget = None
    try:
        budget = validate_budget(budget_text)
    except ValueError as exc:
        result.errors.append(str(exc))
    sectors = validate_sectors(sector_rows, budget, result.errors, result.warnings)
    result.config = validate_config(pop, iters, seed, patience, result.errors)
    if not result.errors:
        result.problem = BudgetProblem(budget, sectors)
    return result


# ==========================================================================
# 4. EXPORT (CSV + JSON)
# ==========================================================================

def export_results(directory: str, scenario: str, problem: BudgetProblem, result: GWOResult):
    """Write both files into `directory`; returns (csv_path, json_path)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(directory, f"allocation_report_{stamp}.csv")
    json_path = os.path.join(directory, f"gwo_run_{stamp}.json")
    x = result.best_position
    rows = problem.sector_report(x)
    total = float(x.sum())

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel-friendly
        w = csv.writer(fh)
        w.writerow(["Sector", "Target need (PHP)", "Priority weight", "Minimum (PHP)",
                    "Maximum (PHP)", "Final allocation (PHP)", "Target coverage (%)",
                    "Share of allocated funds (%)", "Shortfall (PHP)"])
        for r in rows:
            w.writerow([r["sector"], f'{r["target"]:.2f}', r["weight"], f'{r["min"]:.2f}',
                        f'{r["max"]:.2f}', f'{r["allocation"]:.2f}',
                        f'{r["coverage_pct"]:.2f}', f'{r["share_of_allocated_pct"]:.2f}',
                        f'{r["shortfall"]:.2f}'])
        w.writerow([])
        w.writerow(["Available budget (PHP)", f"{problem.budget:.2f}"])
        w.writerow(["Total allocated (PHP)", f"{total:.2f}"])
        w.writerow(["Unallocated balance (PHP)", f"{problem.budget - total:.2f}"])
        w.writerow(["Best fitness F(X)", f"{result.best_fitness:.6f}"])
        w.writerow(["Weighted coverage (F / sum of weights)",
                    f"{problem.normalized_fitness(result.best_fitness):.6f}"])
        w.writerow(["Note", "Educational simulation only; not an official or approved budget."])

    payload = {
        "application": "Crisis Budget Optimizer - Grey Wolf Optimization Laboratory",
        "disclaimer": "Educational simulation; not an official emergency budget.",
        "scenario": scenario,
        "budget_php": problem.budget,
        "sectors": [dict(name=s.name, target=s.target, weight=s.weight,
                         min_alloc=s.min_alloc, max_alloc=s.max_alloc) for s in problem.sectors],
        "gwo_config": {"population_size": result.config.population_size,
                       "max_iterations": result.config.max_iterations,
                       "seed": result.config.seed, "patience": result.config.patience,
                       "a_schedule": "a = 2 - 2t/T, t zero-indexed"},
        "result": {
            "status": result.status, "stop_reason": result.stop_reason,
            "best_fitness": result.best_fitness,
            "weighted_coverage": problem.normalized_fitness(result.best_fitness),
            "best_found_at_iteration": result.best_iteration,
            "iterations_completed": result.iterations_completed,
            "runtime_seconds": result.runtime_seconds,
            "total_allocated": total, "unallocated": problem.budget - total,
            "allocation": rows,
        },
        "history": [{"iteration": s.iteration, "a": s.a, "alpha_fitness": s.alpha_fitness,
                     "beta_fitness": s.beta_fitness, "delta_fitness": s.delta_fitness,
                     "best_so_far_fitness": s.best_so_far_fitness,
                     "mean_fitness": s.mean_fitness, "std_fitness": s.std_fitness,
                     "alpha_allocation": s.alpha_position.tolist(),
                     "all_feasible": s.all_feasible} for s in result.snapshots],
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return csv_path, json_path


# ==========================================================================
# 5. VISUALIZATIONS
# ==========================================================================

# Theme colours (shared with the GUI)
BG, PANEL, CARD = "#0b1220", "#141c2f", "#1b2640"
FG, MUTED, GRID = "#e5ecf6", "#8b9ab5", "#2c3a57"
CYAN, BLUE, GREEN, AMBER, RED = "#22d3ee", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444"
GOLD, SILVER, BRONZE = "#fbbf24", "#e2e8f0", "#fb923c"
PALETTE = ["#22d3ee", "#3b82f6", "#a78bfa", "#22c55e", "#f59e0b", "#f472b6",
           "#fb923c", "#2dd4bf", "#818cf8", "#facc15", "#34d399", "#f87171"]


def short_peso(v: float) -> str:
    if abs(v) >= 1e6:
        return f"\u20b1{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"\u20b1{v / 1e3:.0f}K"
    return f"\u20b1{v:.0f}"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)


def _legend(ax, **kw):
    leg = ax.legend(facecolor=CARD, edgecolor=GRID, fontsize=7, **kw)
    for t in leg.get_texts():
        t.set_color(FG)


def _wrap(names, width=12):
    return [textwrap.fill(n, width) for n in names]


# ----------------------------------------------------------------- live charts
def draw_convergence(ax, snaps, title="Convergence (maximisation)"):
    _style(ax, title, "Iteration", "Fitness F(X)")
    if not snaps:
        return
    its = [s.iteration for s in snaps]
    ax.plot(its, [s.alpha_fitness for s in snaps], color=AMBER, lw=1, ls="--",
            label="Current alpha")
    ax.plot(its, [s.best_so_far_fitness for s in snaps], color=CYAN, lw=2,
            label="Best so far")
    _legend(ax, loc="lower right")


def draw_population(ax, problem: BudgetProblem, snap: IterationSnapshot, i: int, j: int):
    """Two chosen sector dimensions of the real current population."""
    names = problem.names
    _style(ax, f"Wolves in 2 of {problem.n} dimensions - iteration {snap.iteration}",
           f"{names[i]} (\u20b1)", f"{names[j]} (\u20b1)")
    pos = snap.positions
    ax.scatter(pos[:, i], pos[:, j], s=26, color=BLUE, alpha=0.75, label="Wolves (omega)")
    ax.scatter(*snap.delta_position[[i, j]], s=150, marker="^", color=BRONZE,
               edgecolor="white", linewidth=0.6, zorder=4, label="Delta")
    ax.scatter(*snap.beta_position[[i, j]], s=150, marker="D", color=SILVER,
               edgecolor="white", linewidth=0.6, zorder=5, label="Beta")
    ax.scatter(*snap.alpha_position[[i, j]], s=240, marker="*", color=GOLD,
               edgecolor="white", linewidth=0.6, zorder=6, label="Alpha")
    for axis_set, k in ((ax.set_xlim, i), (ax.set_ylim, j)):
        lo, hi = problem.lower[k], problem.effective_upper[k]
        pad = max((hi - lo) * 0.06, 1.0)
        axis_set(lo - pad, hi + pad)
    _legend(ax, loc="best")


def draw_allocation_evolution(ax, problem: BudgetProblem, first: IterationSnapshot,
                              current: IterationSnapshot):
    total = float(current.alpha_position.sum())
    _style(ax, f"Alpha allocation: iteration {first.iteration} vs {current.iteration}"
               f"  |  total {peso(total)} of budget {peso(problem.budget)}", "", "Allocation (\u20b1)")
    x = np.arange(problem.n)
    w = 0.38
    ax.bar(x - w / 2, first.alpha_position, w, color=MUTED, alpha=0.6,
           label=f"Alpha at iteration {first.iteration}")
    bars = ax.bar(x + w / 2, current.alpha_position, w, color=CYAN,
                  label=f"Alpha at iteration {current.iteration}")
    for rect, v in zip(bars, current.alpha_position):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), short_peso(v),
                ha="center", va="bottom", color=FG, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap(problem.names), fontsize=7)
    _legend(ax, loc="upper right")


# --------------------------------------------------------------- result charts
def draw_doughnut(ax, problem: BudgetProblem, alloc):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_title("Final budget distribution", color=FG, fontsize=10, fontweight="bold")
    unalloc = max(problem.budget - float(np.sum(alloc)), 0.0)
    values = [v for v in alloc if v > 0]
    labels = [n for n, v in zip(problem.names, alloc) if v > 0]
    colors = [PALETTE[k % len(PALETTE)] for k, v in enumerate(alloc) if v > 0]
    if unalloc > 1e-6:
        values.append(unalloc); labels.append("Unallocated"); colors.append("#475569")
    wedges, _, autotexts = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                                  autopct="%1.1f%%", pctdistance=0.78,
                                  wedgeprops=dict(width=0.42, edgecolor=PANEL))
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(7)
    leg = ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.95, 0.5),
                    facecolor=CARD, edgecolor=GRID, fontsize=7)
    for t in leg.get_texts():
        t.set_color(FG)
    ax.text(0, 0, "% of\nbudget", ha="center", va="center", color=MUTED, fontsize=8)


def draw_alloc_vs_target(ax, problem: BudgetProblem, alloc):
    _style(ax, "Allocation versus target need", "", "Pesos (\u20b1)")
    x = np.arange(problem.n)
    w = 0.38
    ax.bar(x - w / 2, problem.targets, w, color=AMBER, alpha=0.85, label="Target need")
    ax.bar(x + w / 2, alloc, w, color=GREEN, label="Final allocation")
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap(problem.names, 10), fontsize=7)
    _legend(ax, loc="upper right")


def draw_initial_vs_final(ax, problem: BudgetProblem, initial_best, final_best):
    _style(ax, "Initial vs final best fitness", "", "Fitness F(X)")
    bars = ax.bar(["Initial best\n(iteration 0)", "Final best"], [initial_best, final_best],
                  color=[MUTED, CYAN], width=0.5)
    for r, v in zip(bars, [initial_best, final_best]):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                color=FG, fontsize=8)
    ax.axhline(problem.max_fitness, color=GREEN, ls=":", lw=1,
               label=f"Upper bound = sum of weights ({problem.max_fitness:g})")
    ax.set_ylim(0, problem.max_fitness * 1.08)
    _legend(ax, loc="lower right")


# ------------------------------------------------------------ calculation log
def _vec(v, decimals=2, money=True):
    body = ", ".join((f"{x:,.{decimals}f}" if money else f"{x:.{decimals}f}") for x in v)
    return f"[{body}]"


def format_iteration_log(problem: BudgetProblem, snap: IterationSnapshot, wolf: int = 0) -> str:
    """Real computed values for one iteration and one selected wolf."""
    L = [f"ITERATION {snap.iteration}", "=" * 46,
         "Sector order: " + " | ".join(f"{k + 1}:{n}" for k, n in enumerate(problem.names)), ""]
    pop = snap.positions.shape[0]
    d = snap.details
    if d is None:
        L += ["Initialisation: random population, then constraint repair.",
              f"Feasible wolves: {snap.feasible_count}/{pop}",
              f"Fitness  min/mean/max = {snap.min_fitness:.4f} / {snap.mean_fitness:.4f} / "
              f"{snap.max_fitness:.4f}",
              "", "Leaders chosen from this population:",
              f"  Alpha fitness = {snap.alpha_fitness:.6f}",
              f"  Beta  fitness = {snap.beta_fitness:.6f}",
              f"  Delta fitness = {snap.delta_fitness:.6f}",
              "", "Alpha allocation:", "  " + _vec(snap.alpha_position)]
        return "\n".join(L)

    wolf = int(min(max(wolf, 0), pop - 1))
    L += ["Control parameter:",
          f"  a = {snap.a:.6f}    (a = 2 - 2t/T, t = {snap.iteration - 1} zero-indexed)", "",
          "Leaders used for THIS update (previous population):",
          f"  Alpha fitness = {d['leader_fitness'][0]:.6f}",
          f"  Beta  fitness = {d['leader_fitness'][1]:.6f}",
          f"  Delta fitness = {d['leader_fitness'][2]:.6f}", "",
          f"Selected wolf: #{wolf}   (previous fitness {d['previous_fitness'][wolf]:.6f})",
          "  X = " + _vec(d["previous_positions"][wolf])]
    for k, name in enumerate(("alpha", "beta", "delta")):
        L += ["", f"--- guided by {name} ---",
              f"  A_{name} = " + _vec(d["A"][k][wolf], 4, False),
              f"  C_{name} = " + _vec(d["C"][k][wolf], 4, False),
              f"  D_{name} = |C*X_{name} - X| = " + _vec(d["D"][k][wolf]),
              f"  X{k + 1} = X_{name} - A*D = " + _vec(d["X_guided"][k][wolf])]
    changed = float(np.max(np.abs(d["X_feasible"][wolf] - d["X_new"][wolf])))
    feasible = problem.is_feasible(snap.positions[wolf])
    L += ["", "Updated position BEFORE repair, X_new = (X1+X2+X3)/3:",
          "  " + _vec(d["X_new"][wolf]), "",
          "Updated position AFTER repair, X_feasible:",
          "  " + _vec(d["X_feasible"][wolf]),
          f"  (largest change made by repair: {peso(changed)})", "",
          f"New fitness: {snap.fitness[wolf]:.6f}",
          f"Feasibility: {'PASS' if feasible else 'FAIL'}  "
          f"(total {peso(snap.positions[wolf].sum())} of {peso(problem.budget)})", "",
          "Population after this update:",
          f"  Alpha / Beta / Delta fitness = {snap.alpha_fitness:.6f} / "
          f"{snap.beta_fitness:.6f} / {snap.delta_fitness:.6f}",
          f"  Fitness min/mean/max = {snap.min_fitness:.4f} / {snap.mean_fitness:.4f} / "
          f"{snap.max_fitness:.4f}",
          f"  Feasible wolves: {snap.feasible_count}/{pop}",
          f"  Best so far: {snap.best_so_far_fitness:.6f}", "",
          "[Illustrative note - not computed] |A|>1 pushes wolves away from the leaders "
          "(exploration); |A|<1 pulls them toward the leaders (exploitation)."]
    return "\n".join(L)


# ==========================================================================
# 6. EDUCATION
# ==========================================================================

EDUCATION_TEXT = """\
## What is the Grey Wolf Optimizer (GWO)?
GWO is a population-based search heuristic inspired by how grey wolves hunt in a pack. It does NOT
guarantee the global optimum; it gives a good solution found by guided random search.

## What does each wolf represent here?
One wolf = one complete candidate budget:  X = [x1, x2, ..., xn],  xi = pesos given to sector i.
With 20 wolves, the program is testing 20 different budgets at the same time.

## Alpha, Beta, Delta, Omega
After every evaluation the wolves are ranked by fitness (higher is better):
  Alpha  = best wolf,  Beta = second best,  Delta = third best,  Omega = everyone else.
Leaders can change from iteration to iteration. Ties are broken by wolf index (deterministic).

## Fitness
F(X) = sum_i  wi * min(xi / di, 1)
  wi = priority weight (code: problem.weights),  di = target need (problem.targets),
  xi = allocation. Money above a target earns no extra credit, so unneeded money stays unallocated.
This is an educational model of funding coverage. It is NOT a measure of lives saved, clinical
outcomes, or real disaster-response quality.

## The control parameter a
a = 2 - 2t/T    (t counts updates starting at 0, T = maximum iterations)
a shrinks from 2 toward 0 and controls how boldly wolves move.

## Coefficient vectors A and C, distance vector D
  A = 2*a*r1 - a     (r1 random in [0,1], one number per sector)  -> A lies in [-a, a]
  C = 2*r2           (r2 random in [0,1])                         -> C lies in [0, 2]
  D_alpha = | C1*X_alpha - X |       (similarly D_beta, D_delta)
Fresh random vectors are drawn for every wolf and every leader.

## The position update
  X1 = X_alpha - A1*D_alpha,   X2 = X_beta - A2*D_beta,   X3 = X_delta - A3*D_delta
  X_new = (X1 + X2 + X3) / 3
The new wolf position is the average of three "steps" guided by the three leaders.
Code: gwo.py, method GreyWolfOptimizer.iterate().

## Exploration vs exploitation
When |A| > 1 (early, a is large) wolves move away from leaders and search widely = exploration.
When |A| < 1 (late, a is small) wolves close in on the leaders = exploitation.

## Why constraint repair is necessary
The formulas above know nothing about budgets: X_new can be negative, above a maximum, or add up to
more than the budget. repair() (budget_model.py) therefore
  1. clips each sector into [minimum, effective maximum],
  2. if the total exceeds the budget, shrinks only the money above each minimum, proportionally,
  3. verifies the result and raises an error instead of returning an infeasible candidate.
Clipping each variable alone is not enough - the total could still exceed the budget.

## Why keep a best-so-far archive?
Current leaders can get worse after an update. The archive remembers the best feasible candidate ever
seen, so the reported answer never gets worse. It is used for REPORTING only; the position updates
always use the current population's leaders, as in standard GWO.

## Feasible vs globally optimal
Feasible = obeys all constraints. Globally optimal = the best feasible solution that exists.
Every reported allocation is feasible (checked); the program does not claim it is globally optimal.

## Disclaimer
This is a classroom simulation. Presets are sample data, not official response plans, and results
are not an approved emergency budget or operational guidance.
"""


# ==========================================================================
# 7. GUI (Tkinter)
# ==========================================================================

SPEEDS = {"Fast (no delay)": 0.0, "Normal (40 ms)": 0.04, "Slow (150 ms)": 0.15,
          "Step-by-step (600 ms)": 0.6}
CHECK, CROSS = "\u2714", "\u2718"
TAB_SETUP, TAB_LIVE, TAB_RESULTS, TAB_HISTORY, TAB_LEARN = range(5)


# ======================================================================= dialogs
class SectorDialog(tk.Toplevel):
    """Add / edit one sector row."""
    FIELDS = ["Sector name", "Target funding need (\u20b1)", "Priority weight",
              "Minimum allocation (\u20b1)", "Maximum allocation (\u20b1)"]

    def __init__(self, master, values, on_ok):
        super().__init__(master)
        self.title("Sector"); self.configure(bg=BG); self.transient(master); self.on_ok = on_ok
        self.vars = [tk.StringVar(value=str(v)) for v in values]
        for r, (label, var) in enumerate(zip(self.FIELDS, self.vars)):
            ttk.Label(self, text=label).grid(row=r, column=0, sticky="w", padx=12, pady=5)
            ttk.Entry(self, textvariable=var, width=30).grid(row=r, column=1, padx=12, pady=5)
        bar = ttk.Frame(self); bar.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(bar, text="Save", command=self._save).pack(side="left", padx=6)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        self.grab_set()

    def _save(self):
        vals = [v.get().strip() for v in self.vars]
        if not vals[0]:
            messagebox.showerror("Sector", "Sector name must not be empty.", parent=self); return
        try:
            for label, v in zip(self.FIELDS[1:], vals[1:]):
                parse_number(v, label)
        except ValueError as exc:
            messagebox.showerror("Sector", str(exc), parent=self); return
        self.on_ok(vals); self.destroy()


class LogPanel(ttk.Frame):
    """Scrollable mathematical log with iteration and wolf selectors (real data only)."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app; self._busy = False
        top = ttk.Frame(self); top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Iteration").pack(side="left")
        self.scale = ttk.Scale(top, from_=0, to=1, command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=6)
        self.iter_lbl = ttk.Label(top, text="0", width=5); self.iter_lbl.pack(side="left")
        ttk.Label(top, text="Wolf #").pack(side="left", padx=(8, 2))
        self.wolf_var = tk.StringVar(value="0")
        self.spin = tk.Spinbox(top, from_=0, to=POP_MAX - 1, width=4, textvariable=self.wolf_var,
                               command=self.render, bg=CARD, fg=FG, buttonbackground=CARD,
                               insertbackground=FG, relief="flat")
        self.spin.pack(side="left"); self.spin.bind("<KeyRelease>", lambda e: self.render())
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Follow live", variable=self.follow).pack(side="left", padx=8)
        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="none", bg=PANEL, fg=FG, font=("Consolas", 9),
                            relief="flat", state="disabled", width=60)
        ys = ttk.Scrollbar(body, command=self.text.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y"); self.text.pack(side="left", fill="both", expand=True)
        xs.pack(fill="x")

    def reset(self):
        self._set_text("Waiting for optimisation data...")
        self.scale.configure(to=1); self.iter_lbl.configure(text="0")

    def _set_text(self, s):
        self.text.configure(state="normal"); self.text.delete("1.0", "end")
        self.text.insert("1.0", s); self.text.configure(state="disabled")

    def on_new_snapshot(self):
        n = len(self.app.history)
        self.scale.configure(to=max(1, n - 1))
        if self.follow.get():
            self.show_iteration(n - 1)

    def show_iteration(self, k):
        self._busy = True
        self.scale.set(k); self._busy = False
        self.render()

    def _on_scale(self, _v):
        if not self._busy:
            self.follow.set(False); self.render()

    def render(self):
        hist = self.app.history
        if not hist or self.app.run_problem is None:
            return
        k = min(int(round(float(self.scale.get()))), len(hist) - 1)
        try:
            wolf = int(self.wolf_var.get())
        except ValueError:
            wolf = 0
        self.iter_lbl.configure(text=str(hist[k].iteration))
        self._set_text(format_iteration_log(self.app.run_problem, hist[k], wolf))


# ========================================================================== app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crisis Budget Optimizer - Grey Wolf Optimization Laboratory")
        self.geometry("1440x880"); self.minsize(1180, 740); self.configure(bg=BG)
        self.msgq: queue.Queue = queue.Queue()
        self.run_id = 0; self.running = False; self.control = None
        self.history = []; self.result = None
        self.run_problem = self.run_config = None; self.run_scenario = ""
        self._last_draw = 0.0
        self._style(); self._header()
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tabs = [ttk.Frame(self.nb) for _ in range(5)]
        for tab, name in zip(self.tabs, ["1  Scenario Setup", "2  Live Laboratory",
                                         "3  Final Results", "4  Optimization History",
                                         "5  Learn GWO"]):
            self.nb.add(tab, text=name)
        self._build_setup(); self._build_live(); self._build_results_shell()
        self._build_history(); self._build_learn()
        self.load_preset(list(PRESETS)[0])
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(40, self.poll_queue)

    # ------------------------------------------------------------------ style
    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=CARD, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("CardTitle.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        s.configure("CardValue.TLabel", background=CARD, foreground=CYAN,
                    font=("Segoe UI", 13, "bold"))
        s.configure("Title.TLabel", font=("Segoe UI", 17, "bold"), foreground=CYAN)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("TButton", background=BLUE, foreground="white", padding=7, borderwidth=0)
        s.map("TButton", background=[("active", "#2563eb"), ("disabled", "#334155")],
              foreground=[("disabled", "#64748b")])
        s.configure("Run.TButton", background=GREEN, foreground="#04210e",
                    font=("Segoe UI", 12, "bold"), padding=11)
        s.map("Run.TButton", background=[("active", "#16a34a"), ("disabled", "#334155")])
        s.configure("Danger.TButton", background="#b91c1c")
        s.map("Danger.TButton", background=[("active", RED), ("disabled", "#334155")])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(14, 7))
        s.map("TNotebook.Tab", background=[("selected", BLUE)], foreground=[("selected", "white")])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=26)
        s.configure("Treeview.Heading", background=CARD, foreground=CYAN, font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", BLUE)])
        s.configure("TEntry", fieldbackground=CARD, foreground=FG)
        s.configure("TCombobox", fieldbackground=CARD, foreground=FG, arrowcolor=FG)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.configure("Horizontal.TProgressbar", background=CYAN, troughcolor=CARD)
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", FG)

    def _header(self):
        h = ttk.Frame(self); h.pack(fill="x", padx=14, pady=(10, 6))
        ttk.Label(h, text="Crisis Budget Optimizer", style="Title.TLabel").pack(side="left")
        ttk.Label(h, text="   Grey Wolf Optimization Laboratory  |  educational simulation, "
                          "not official emergency guidance", style="Muted.TLabel").pack(side="left")

    # ------------------------------------------------------------------ setup
    def _build_setup(self):
        f = self.tabs[TAB_SETUP]
        cards = ttk.Frame(f); cards.pack(fill="x", padx=10, pady=10)
        self.card_vars = {}
        for i, title in enumerate(["Scenario", "Available budget", "Sectors", "Population size",
                                   "Max iterations", "Optimization status"]):
            c = ttk.Frame(cards, style="Card.TFrame", padding=10)
            c.grid(row=0, column=i, sticky="nsew", padx=4); cards.columnconfigure(i, weight=1)
            ttk.Label(c, text=title, style="CardTitle.TLabel").pack(anchor="w")
            v = tk.StringVar(value="-"); self.card_vars[title] = v
            ttk.Label(c, textvariable=v, style="CardValue.TLabel", wraplength=190).pack(anchor="w")
        self.card_vars["Optimization status"].set("Ready")

        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=10)
        left = ttk.Frame(body, padding=(0, 0, 14, 0)); left.pack(side="left", fill="y")
        self.scenario_var = tk.StringVar(); self.budget_var = tk.StringVar()
        self.pop_var = tk.StringVar(value="20"); self.iter_var = tk.StringVar(value="100")
        self.seed_var = tk.StringVar(); self.patience_var = tk.StringVar(value="0")
        ttk.Label(left, text="Crisis scenario preset").pack(anchor="w")
        cb = ttk.Combobox(left, textvariable=self.scenario_var, values=list(PRESETS),
                          state="readonly", width=34); cb.pack(anchor="w", pady=(0, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.scenario_var.get()))
        ttk.Label(left, text="Available budget (\u20b1)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.budget_var, width=36).pack(anchor="w", pady=(0, 12))
        ttk.Label(left, text="GWO configuration", foreground=CYAN,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for label, var, hint in [("Population size", self.pop_var, f"{POP_MIN}-{POP_MAX}"),
                                 ("Maximum iterations", self.iter_var, f"{ITER_MIN}-{ITER_MAX}"),
                                 ("Random seed (optional)", self.seed_var, "blank = random"),
                                 ("Early-stop patience", self.patience_var, "0 = off")]:
            r = ttk.Frame(left); r.pack(fill="x", pady=2)
            ttk.Label(r, text=label, width=22).pack(side="left")
            ttk.Entry(r, textvariable=var, width=9).pack(side="left")
            ttk.Label(r, text=f" {hint}", style="Muted.TLabel").pack(side="left")
        ttk.Label(left, style="Muted.TLabel", wraplength=330, justify="left",
                  text="Control parameter: a = 2 - 2t/T, with t counted from 0. Presets are "
                       "editable sample data, not official plans. Early-stop patience stops the "
                       "run if the best-so-far fitness does not improve for that many iterations."
                  ).pack(anchor="w", pady=10)
        for v in (self.scenario_var, self.budget_var, self.pop_var, self.iter_var):
            v.trace_add("write", lambda *a: self.refresh_cards())

        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Budget sectors (double-click a row to edit)").pack(anchor="w")
        cols = ("name", "target", "weight", "min", "max")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c, t, w in zip(cols, ["Sector", "Target need (\u20b1)", "Priority weight",
                                  "Minimum (\u20b1)", "Maximum (\u20b1)"], [250, 130, 110, 120, 120]):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="w" if c == "name" else "e")
        self.tree.pack(fill="both", expand=True, pady=4)
        self.tree.bind("<Double-1>", lambda e: self.edit_sector())
        bar = ttk.Frame(right); bar.pack(fill="x")
        ttk.Button(bar, text="Add sector", command=self.add_sector).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Edit selected", command=self.edit_sector).pack(side="left", padx=6)
        ttk.Button(bar, text="Remove selected", command=self.remove_sector).pack(side="left", padx=6)

        bottom = ttk.Frame(f); bottom.pack(fill="x", padx=10, pady=10)
        self.msg_var = tk.StringVar()
        self.msg_label = ttk.Label(bottom, textvariable=self.msg_var, wraplength=1000, justify="left")
        self.msg_label.pack(side="left", fill="x", expand=True)
        self.run_btn = ttk.Button(bottom, text="RUN OPTIMIZATION", style="Run.TButton",
                                  command=self.on_run_clicked)
        self.run_btn.pack(side="right")

    def load_preset(self, name):
        budget, rows = PRESETS[name]
        self.scenario_var.set(name); self.budget_var.set(f"{budget:,}")
        self.tree.delete(*self.tree.get_children())
        for n, t, w, lo, hi in rows:
            self.tree.insert("", "end", values=(n, f"{t:,}", w, f"{lo:,}", f"{hi:,}"))
        self.refresh_cards()

    def refresh_cards(self):
        self.card_vars["Scenario"].set(self.scenario_var.get() or "-")
        try:
            self.card_vars["Available budget"].set(peso(parse_number(self.budget_var.get(), "b")))
        except ValueError:
            self.card_vars["Available budget"].set("invalid")
        self.card_vars["Sectors"].set(str(len(self.tree.get_children())))
        self.card_vars["Population size"].set(self.pop_var.get() or "-")
        self.card_vars["Max iterations"].set(self.iter_var.get() or "-")

    def add_sector(self):
        if len(self.tree.get_children()) >= MAX_SECTORS:
            messagebox.showinfo("Sectors", f"At most {MAX_SECTORS} sectors are supported."); return
        SectorDialog(self, ["New sector", "500,000", "5", "0", "500,000"],
                     lambda v: (self.tree.insert("", "end", values=v), self.refresh_cards()))

    def edit_sector(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Sectors", "Select a sector first."); return
        SectorDialog(self, self.tree.item(sel[0], "values"),
                     lambda v: self.tree.item(sel[0], values=v))

    def remove_sector(self):
        for i in self.tree.selection():
            self.tree.delete(i)
        self.refresh_cards()

    def set_message(self, text, color=MUTED):
        self.msg_var.set(text); self.msg_label.configure(foreground=color)

    def validate_current(self):
        rows = [self.tree.item(i, "values") for i in self.tree.get_children()]
        return validate_all(self.budget_var.get(), rows, self.pop_var.get(), self.iter_var.get(),
                            self.seed_var.get(), self.patience_var.get())

    # ------------------------------------------------------ run request/review
    def on_run_clicked(self):
        if self.running:
            return
        res = self.validate_current()
        if not res.ok:
            self.set_message("Cannot start:\n- " + "\n- ".join(res.errors), RED)
            self.nb.select(TAB_SETUP)
            return
        self.set_message("Inputs valid." + (" Notes: " + " ".join(res.warnings) if res.warnings else ""),
                         GREEN if not res.warnings else AMBER)
        self.show_review(res)

    def show_review(self, res):
        p, c = res.problem, res.config
        dlg = tk.Toplevel(self); dlg.title("Review before execution"); dlg.configure(bg=BG)
        dlg.transient(self); dlg.geometry("720x560")
        lines = [f"Scenario: {self.scenario_var.get()}", f"Available budget: {peso(p.budget)}",
                 f"Total target need: {peso(p.targets.sum())}", "",
                 "Sectors (target | weight | min | max):"]
        for s in p.sectors:
            lines.append(f"  - {s.name}: {peso(s.target)} | w={s.weight:g} | "
                         f"{peso(s.min_alloc)} | {peso(s.max_alloc)}")
        lines += ["", "Objective (maximise):  F(X) = sum_i wi * min(xi/di, 1)",
                  "Constraints: li <= xi <= ui,  sum(xi) <= budget,  xi >= 0",
                  "Money above a target gives no credit, so it stays unallocated.", "",
                  f"Population size: {c.population_size}", f"Maximum iterations: {c.max_iterations}",
                  f"Random seed: {c.seed if c.seed is not None else 'none (random)'}",
                  f"Early-stop patience: {c.patience or 'off'}",
                  "Control parameter: a = 2 - 2t/T  (t from 0)"]
        if res.warnings:
            lines += ["", "Notes:"] + [f"  * {w}" for w in res.warnings]
        t = tk.Text(dlg, bg=PANEL, fg=FG, font=("Consolas", 10), relief="flat", wrap="word")
        t.insert("1.0", "\n".join(lines)); t.configure(state="disabled")
        t.pack(fill="both", expand=True, padx=12, pady=12)
        bar = ttk.Frame(dlg); bar.pack(pady=(0, 12))
        def go():
            dlg.destroy(); self.start_run(p, c, self.scenario_var.get())
        ttk.Button(bar, text="START RUN", style="Run.TButton", command=go).pack(side="left", padx=8)
        ttk.Button(bar, text="Back to setup", command=dlg.destroy).pack(side="left", padx=8)
        dlg.grab_set()

    # ---------------------------------------------------------- run lifecycle
    def start_run(self, problem, config, scenario):
        """Start a NEW run from scratch (never a resume). Safe to call while another runs."""
        if self.control is not None:
            self.control.stop()                      # old worker exits; its output is ignored
        self.run_id += 1; rid = self.run_id
        self.run_problem, self.run_config, self.run_scenario = problem, config, scenario
        self.history = []; self.result = None
        self.control = RunControl(SPEEDS[self.speed_var.get()]); ctl = self.control
        self.running = True
        self._clear_results(); self.hist_tree.delete(*self.hist_tree.get_children())
        self.live_log.reset(); self.hist_log.reset(); self.live_log.follow.set(True)
        for cb in (self.axis_x, self.axis_y):
            cb.configure(values=problem.names)
        self.axis_x.current(0); self.axis_y.current(min(1, problem.n - 1))
        self.progress.configure(maximum=config.max_iterations, value=0)
        self.pause_btn.configure(text="PAUSE")
        self._set_status("Running"); self._sync_buttons()
        self.nb.select(TAB_LIVE)
        opt = GreyWolfOptimizer(problem, config)

        def worker():
            try:
                res = opt.run(callback=lambda s: self.msgq.put((rid, "snap", s)), control=ctl)
                self.msgq.put((rid, "done", res))
            except Exception as exc:                  # surfaced to the GUI thread
                self.msgq.put((rid, "error", repr(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll_queue(self):
        try:
            new, done, error = False, None, None
            try:
                while True:
                    rid, kind, payload = self.msgq.get_nowait()
                    if rid != self.run_id:
                        continue                      # stale output from an older run
                    if kind == "snap":
                        self.history.append(payload); self._add_history_row(payload); new = True
                    elif kind == "done":
                        done = payload
                    elif kind == "error":
                        error = payload
            except queue.Empty:
                pass
            if new:
                last = self.history[-1]
                self.progress.configure(value=last.iteration)
                self.progress_lbl.configure(
                    text=f"iteration {last.iteration}/{self.run_config.max_iterations}   "
                         f"best so far {last.best_so_far_fitness:.4f}")
                now = time.perf_counter()
                if now - self._last_draw > 0.08 or done is not None:
                    self.redraw_live(); self.live_log.on_new_snapshot(); self.hist_log.on_new_snapshot()
                    self._last_draw = now
            if error is not None:
                self.running = False; self._set_status("Error"); self._sync_buttons()
                messagebox.showerror("Optimization error", error)
            if done is not None:
                self.on_run_done(done)
        finally:
            self.after(40, self.poll_queue)

    def on_run_done(self, result):
        self.running = False; self.result = result
        self.redraw_live(); self.live_log.on_new_snapshot(); self.hist_log.on_new_snapshot()
        self._sync_buttons()
        if result.is_complete:
            self._set_status("Completed" if result.status == "completed" else "Completed (early stop)")
            self.show_results(result); self.nb.select(TAB_RESULTS)
        else:
            self._set_status("Stopped - INCOMPLETE run")
            self.progress_lbl.configure(text=self.progress_lbl.cget("text") + "   (stopped by user)")

    def _set_status(self, s):
        self.card_vars["Optimization status"].set(s); self.status_var.set(s)

    def _sync_buttons(self):
        self.run_btn.configure(state="disabled" if self.running else "normal")
        self.start_btn.configure(state="disabled" if self.running else "normal")
        self.pause_btn.configure(state="normal" if self.running else "disabled")
        self.stop_btn.configure(state="normal" if self.running else "disabled")
        self.restart_btn.configure(state="normal" if self.run_problem is not None else "disabled")

    def on_pause(self):
        if not (self.running and self.control):
            return
        if self.control.paused:
            self.control.resume(); self.pause_btn.configure(text="PAUSE"); self._set_status("Running")
        else:
            self.control.pause(); self.pause_btn.configure(text="RESUME"); self._set_status("Paused")

    def on_stop(self):
        if self.running and self.control:
            self.control.stop()

    def on_restart(self):
        if self.run_problem is not None:
            self.start_run(self.run_problem, self.run_config, self.run_scenario)

    def on_speed(self, _e=None):
        if self.control:
            self.control.delay = SPEEDS[self.speed_var.get()]

    def on_close(self):
        if self.control:
            self.control.stop()
        self.destroy()

    # ------------------------------------------------------------------- live
    def _build_live(self):
        f = self.tabs[TAB_LIVE]
        bar = ttk.Frame(f); bar.pack(fill="x", padx=10, pady=8)
        self.start_btn = ttk.Button(bar, text="START", style="Run.TButton", command=self.on_run_clicked)
        self.pause_btn = ttk.Button(bar, text="PAUSE", command=self.on_pause)
        self.stop_btn = ttk.Button(bar, text="STOP", style="Danger.TButton", command=self.on_stop)
        self.restart_btn = ttk.Button(bar, text="RESTART", command=self.on_restart)
        for b in (self.start_btn, self.pause_btn, self.stop_btn, self.restart_btn):
            b.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status_var, foreground=AMBER,
                  font=("Segoe UI", 10, "bold")).pack(side="left", padx=14)
        self.speed_var = tk.StringVar(value="Normal (40 ms)")
        sp = ttk.Combobox(bar, textvariable=self.speed_var, values=list(SPEEDS), state="readonly", width=20)
        sp.pack(side="right"); sp.bind("<<ComboboxSelected>>", self.on_speed)
        ttk.Label(bar, text="Viewing speed:").pack(side="right", padx=4)
        pbar = ttk.Frame(f); pbar.pack(fill="x", padx=14)
        self.progress = ttk.Progressbar(pbar, maximum=100); self.progress.pack(side="left", fill="x", expand=True)
        self.progress_lbl = ttk.Label(pbar, text="no run yet", style="Muted.TLabel", width=48)
        self.progress_lbl.pack(side="left", padx=8)
        axbar = ttk.Frame(f); axbar.pack(fill="x", padx=10, pady=(6, 0))
        ttk.Label(axbar, text="Population chart axes -  X:").pack(side="left")
        self.axis_x = ttk.Combobox(axbar, state="readonly", width=26); self.axis_x.pack(side="left", padx=4)
        ttk.Label(axbar, text="Y:").pack(side="left")
        self.axis_y = ttk.Combobox(axbar, state="readonly", width=26); self.axis_y.pack(side="left", padx=4)
        for cb in (self.axis_x, self.axis_y):
            cb.bind("<<ComboboxSelected>>", lambda e: self.redraw_live())
        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=10, pady=8)
        self.live_fig = Figure(figsize=(8, 7), facecolor=BG, constrained_layout=True)
        gs = self.live_fig.add_gridspec(2, 2)
        self.ax_conv = self.live_fig.add_subplot(gs[0, 0])
        self.ax_pop = self.live_fig.add_subplot(gs[0, 1])
        self.ax_alloc = self.live_fig.add_subplot(gs[1, :])
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=body)
        self.live_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.live_log = LogPanel(body, self); self.live_log.pack(side="left", fill="y", padx=(8, 0))
        self.live_log.reset()
        self._sync_buttons(); self.pause_btn.configure(state="disabled"); self.stop_btn.configure(state="disabled")
        self.restart_btn.configure(state="disabled")

    def redraw_live(self):
        if not self.history or self.run_problem is None:
            return
        p, last = self.run_problem, self.history[-1]
        i = max(self.axis_x.current(), 0); j = max(self.axis_y.current(), 0)
        draw_convergence(self.ax_conv, self.history)
        draw_population(self.ax_pop, p, last, i, j)
        draw_allocation_evolution(self.ax_alloc, p, self.history[0], last)
        self.live_canvas.draw_idle()

    # ----------------------------------------------------------------- results
    def _build_results_shell(self):
        self.results_host = ttk.Frame(self.tabs[TAB_RESULTS])
        self.results_host.pack(fill="both", expand=True)
        self._clear_results()

    def _clear_results(self):
        for w in self.results_host.winfo_children():
            w.destroy()
        ttk.Label(self.results_host, style="Muted.TLabel", font=("Segoe UI", 12),
                  text="No completed optimisation to show.\nRun an optimisation to completion "
                       "(a stopped run is not a completed result).").pack(expand=True)

    def show_results(self, r):
        for w in self.results_host.winfo_children():
            w.destroy()
        p, x = self.run_problem, r.best_position
        rows = p.sector_report(x); total = float(x.sum())
        left = ttk.Frame(self.results_host); left.pack(side="left", fill="y", padx=10, pady=10)
        right = ttk.Frame(self.results_host); right.pack(side="left", fill="both", expand=True, pady=10)

        grid = ttk.Frame(left, style="Card.TFrame", padding=10); grid.pack(fill="x")
        seed = r.config.seed if r.config.seed is not None else "none"
        items = [("Best fitness F(X)", f"{r.best_fitness:.4f}  (of max {p.max_fitness:g})"),
                 ("Weighted coverage", f"{p.normalized_fitness(r.best_fitness) * 100:.2f}%"),
                 ("Available budget", peso(p.budget)), ("Total allocated", peso(total)),
                 ("Unallocated balance", peso(p.budget - total)),
                 ("Total target need", peso(p.targets.sum())),
                 ("Sectors / population", f"{p.n} / {r.config.population_size}"),
                 ("Iterations completed", f"{r.iterations_completed} of {r.config.max_iterations}"),
                 ("Best found at iteration", str(r.best_iteration)),
                 ("Random seed", str(seed)), ("Runtime", f"{r.runtime_seconds:.2f} s"),
                 ("Run status", r.stop_reason)]
        for k, (a, b) in enumerate(items):
            ttk.Label(grid, text=a, style="CardTitle.TLabel").grid(row=k, column=0, sticky="w", pady=1)
            ttk.Label(grid, text=b, background=CARD, foreground=FG, wraplength=250,
                      justify="left").grid(row=k, column=1, sticky="w", padx=10)
        ttk.Label(left, text="Best Alpha allocation vector", foreground=CYAN).pack(anchor="w", pady=(8, 0))
        ttk.Label(left, text="[" + ", ".join(f"{v:,.2f}" for v in x) + "]", wraplength=380,
                  style="Muted.TLabel", justify="left").pack(anchor="w")

        ttk.Label(left, text="Constraint verification", foreground=CYAN).pack(anchor="w", pady=(10, 2))
        for label, ok, detail in p.verify(x, r.best_fitness):
            ttk.Label(left, text=f"{CHECK if ok else CROSS}  {label}" + (f"  ({detail})" if detail else ""),
                      foreground=GREEN if ok else RED, wraplength=390, justify="left").pack(anchor="w")
        ttk.Label(left, text="Unmet needs", foreground=CYAN).pack(anchor="w", pady=(10, 2))
        short = [q for q in rows if q["shortfall"] > 1e-6]
        if short:
            for q in short:
                ttk.Label(left, text=f"\u25b2  {q['sector']}: short by {peso(q['shortfall'])} "
                                     f"({q['coverage_pct']:.1f}% covered)", foreground=AMBER,
                          wraplength=390, justify="left").pack(anchor="w")
        else:
            ttk.Label(left, foreground=GREEN, wraplength=390, justify="left",
                      text=f"All modeled target needs are covered. Remaining funds: "
                           f"{peso(p.budget - total)}.").pack(anchor="w")
        ttk.Label(left, style="Muted.TLabel", wraplength=390, justify="left",
                  text="Educational simulation: this is not an officially approved emergency "
                       "budget. The fitness score is model-specific, not a universal measure of "
                       "response quality. GWO does not guarantee the global optimum.").pack(anchor="w", pady=8)

        cols = ("sector", "target", "weight", "min", "max", "alloc", "cov", "share")
        tv = ttk.Treeview(right, columns=cols, show="headings", height=min(p.n, 8)); tv.pack(fill="x")
        for c, t, w in zip(cols, ["Sector", "Target", "Weight", "Min", "Max", "Final allocation",
                                  "% of target covered", "% of allocated funds"],
                           [170, 100, 55, 95, 100, 110, 110, 110]):
            tv.heading(c, text=t); tv.column(c, width=w, anchor="w" if c == "sector" else "e")
        for q in rows:
            tv.insert("", "end", values=(q["sector"], peso(q["target"]), f"{q['weight']:g}",
                      peso(q["min"]), peso(q["max"]), peso(q["allocation"]),
                      f"{q['coverage_pct']:.1f}%", f"{q['share_of_allocated_pct']:.1f}%"))
        fig = Figure(figsize=(8, 5), facecolor=BG, constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        draw_doughnut(fig.add_subplot(gs[0, 0]), p, x)
        draw_alloc_vs_target(fig.add_subplot(gs[0, 1]), p, x)
        draw_convergence(fig.add_subplot(gs[1, 0]), r.snapshots, "Final convergence")
        draw_initial_vs_final(fig.add_subplot(gs[1, 1]), p, r.initial_best_fitness, r.best_fitness)
        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=6); canvas.draw()

        bar = ttk.Frame(right); bar.pack(fill="x")
        for text, cmd in [("RUN AGAIN", self.on_restart), ("EDIT INPUTS", lambda: self.nb.select(TAB_SETUP)),
                          ("NEW SCENARIO", self.on_new_scenario), ("EXPORT RESULTS", self.on_export),
                          ("VIEW OPTIMIZATION HISTORY", lambda: self.nb.select(TAB_HISTORY))]:
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=4)
        if r.config.seed is not None:
            ttk.Label(right, style="Muted.TLabel",
                      text=f"Seed {r.config.seed}: RUN AGAIN with the same inputs should reproduce this result."
                      ).pack(anchor="w", pady=2)

    def on_new_scenario(self):
        if messagebox.askyesno("New scenario", "Reset the setup form to an editable preset?\n"
                                               "(Current inputs will be replaced; results stay until a new run.)"):
            self.load_preset("Custom crisis scenario"); self.nb.select(TAB_SETUP)

    def on_export(self):
        if not (self.result and self.result.is_complete):
            messagebox.showinfo("Export", "There is no completed run to export."); return
        d = filedialog.askdirectory(title="Choose a folder for the CSV and JSON files")
        if not d:
            return
        try:
            c, j = export_results(d, self.run_scenario, self.run_problem, self.result)
            messagebox.showinfo("Export complete", f"Saved:\n{c}\n{j}")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    # ----------------------------------------------------------------- history
    def _build_history(self):
        f = self.tabs[TAB_HISTORY]
        top = ttk.Frame(f); top.pack(fill="x", padx=10, pady=8)
        cols = ("it", "a", "alpha", "beta", "delta", "best", "mean", "feas")
        self.hist_tree = ttk.Treeview(top, columns=cols, show="headings", height=11)
        for c, t, w in zip(cols, ["Iteration", "a", "Alpha fitness", "Beta fitness", "Delta fitness",
                                  "Best so far", "Mean fitness", "Feasible wolves"],
                           [80, 80, 110, 110, 110, 110, 110, 120]):
            self.hist_tree.heading(c, text=t); self.hist_tree.column(c, width=w, anchor="e")
        ys = ttk.Scrollbar(top, command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=ys.set)
        self.hist_tree.pack(side="left", fill="both", expand=True); ys.pack(side="left", fill="y")
        self.hist_tree.bind("<<TreeviewSelect>>", self._on_hist_select)
        self.hist_log = LogPanel(f, self); self.hist_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.hist_log.follow.set(False); self.hist_log.reset()

    def _add_history_row(self, s):
        self.hist_tree.insert("", "end", iid=str(s.iteration), values=(
            s.iteration, "-" if s.a is None else f"{s.a:.4f}", f"{s.alpha_fitness:.4f}",
            f"{s.beta_fitness:.4f}", f"{s.delta_fitness:.4f}", f"{s.best_so_far_fitness:.4f}",
            f"{s.mean_fitness:.4f}", f"{s.feasible_count}/{len(s.fitness)}"))

    def _on_hist_select(self, _e):
        sel = self.hist_tree.selection()
        if sel and self.history:
            self.hist_log.follow.set(False)
            self.hist_log.show_iteration(min(int(sel[0]), len(self.history) - 1))

    # ------------------------------------------------------------------- learn
    def _build_learn(self):
        t = tk.Text(self.tabs[TAB_LEARN], wrap="word", bg=PANEL, fg=FG, font=("Segoe UI", 10),
                    relief="flat", padx=16, pady=12, spacing3=3)
        ys = ttk.Scrollbar(self.tabs[TAB_LEARN], command=t.yview); t.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y"); t.pack(fill="both", expand=True)
        t.tag_configure("h", foreground=CYAN, font=("Segoe UI", 12, "bold"), spacing1=10)
        t.tag_configure("eq", foreground=AMBER, font=("Consolas", 10))
        for line in EDUCATION_TEXT.splitlines():
            if line.startswith("## "):
                t.insert("end", line[3:] + "\n", "h")
            elif line.startswith("  ") and any(s in line for s in ("=", "|")):
                t.insert("end", line + "\n", "eq")
            else:
                t.insert("end", line + "\n")
        t.configure(state="disabled")




# ==========================================================================
# 8. TESTS (python crisis_budget_optimizer.py --test)
# ==========================================================================

def small_problem(budget=300.0):
    return BudgetProblem(budget, [Sector("A", 100, 2, 0, 150), Sector("B", 200, 1, 0, 250)])


class TestFitness(unittest.TestCase):
    def test_hand_checked_value(self):
        # F = 2*min(50/100,1) + 1*min(300/200,1) = 1.0 + 1.0 = 2.0 (B is over target: capped at 1)
        self.assertAlmostEqual(small_problem().fitness([50, 300]), 2.0)

    def test_fully_covered_equals_sum_of_weights(self):
        p = small_problem()
        self.assertAlmostEqual(p.fitness([100, 200]), 3.0)
        self.assertAlmostEqual(p.normalized_fitness(3.0), 1.0)

    def test_no_credit_above_target(self):
        p = small_problem()
        self.assertAlmostEqual(p.fitness([100, 200]), p.fitness([150, 250]))

    def test_zero_target_is_handled(self):
        p = BudgetProblem(100, [Sector("Z", 0, 3, 0, 10), Sector("B", 50, 1, 0, 50)])
        self.assertAlmostEqual(p.fitness([0, 25]), 3.0 + 0.5)   # no ZeroDivisionError

    def test_population_input(self):
        out = small_problem().fitness(np.array([[50, 300], [0, 0]]))
        np.testing.assert_allclose(out, [2.0, 0.0])


class TestRepair(unittest.TestCase):
    def test_repair_always_feasible(self):
        rng = np.random.default_rng(1)
        for name in PRESETS:
            p = preset_problem(name)
            for _ in range(300):
                x = rng.normal(0, 5e6, p.n) * rng.choice([1, 100, 1e-3])
                y = p.repair(x)
                self.assertTrue(np.all(y >= p.lower - TOL))          # minimums
                self.assertTrue(np.all(y <= p.upper + TOL))          # maximums
                self.assertTrue(np.all(y >= 0))                      # nonnegative
                self.assertLessEqual(y.sum(), p.budget + TOL)        # budget

    def test_clipping_alone_would_break_budget_but_repair_does_not(self):
        p = preset_problem()
        huge = p.upper.copy()
        self.assertGreater(np.clip(huge, p.lower, p.upper).sum(), p.budget)
        self.assertLessEqual(p.repair(huge).sum(), p.budget + TOL)

    def test_nan_and_wrong_shape_rejected(self):
        p = preset_problem()
        with self.assertRaises(RepairError):
            p.repair([np.nan] * p.n)
        with self.assertRaises(RepairError):
            p.repair([1.0, 2.0])

    def test_infeasible_problem_not_repaired(self):
        p = BudgetProblem(100, [Sector("A", 100, 1, 80, 100), Sector("B", 100, 1, 80, 100)])
        with self.assertRaises(RepairError):
            p.repair([80, 80])

    def test_verify_checklist_passes_and_fails_honestly(self):
        p = preset_problem()
        good = p.repair(p.upper)
        self.assertTrue(all(ok for _, ok, _ in p.verify(good, p.fitness(good))))
        bad = p.upper.copy()                       # exceeds the budget
        self.assertFalse(all(ok for _, ok, _ in p.verify(bad)))
        self.assertFalse(p.verify(good, reported_fitness=123.0)[-1][1])


class TestValidation(unittest.TestCase):
    rows = [("A", "100", "1", "10", "100"), ("B", "100", "1", "10", "100")]

    def test_invalid_numbers_rejected(self):
        for bad in ["abc", "", "nan", "inf", "-5", "0"]:
            with self.assertRaises(ValueError):
                validate_budget(bad)
        self.assertEqual(validate_budget("\u20b11,500,000"), 1_500_000)

    def test_minimums_exceeding_budget_detected(self):
        rows = [("A", "100", "1", "80", "100"), ("B", "100", "1", "80", "100")]
        r = validate_all("100", rows, "20", "50", "", "0")
        self.assertFalse(r.ok)
        self.assertTrue(any("Infeasible" in e for e in r.errors))

    def test_min_greater_than_max_and_negative_detected(self):
        rows = [("A", "100", "1", "90", "50"), ("B", "-5", "1", "0", "10")]
        r = validate_all("1000", rows, "20", "50", "", "0")
        self.assertFalse(r.ok)
        self.assertEqual(len(r.errors), 2)

    def test_bad_config_rejected(self):
        for args in [("2", "50"), ("20", "0"), ("20.5", "50"), ("x", "50")]:
            r = validate_all("1000", self.rows, args[0], args[1], "", "0")
            self.assertFalse(r.ok)

    def test_valid_input_builds_problem(self):
        r = validate_all("1000", self.rows, "20", "50", "42", "0")
        self.assertTrue(r.ok)
        self.assertEqual(r.config.seed, 42)


def preset_problem(name="Typhoon and flood response", budget=None):
    b, rows = PRESETS[name]
    return BudgetProblem(budget or b, [Sector(*r) for r in rows])


def run_gwo(problem, seed=7, pop=20, iters=40, **kw):
    return GreyWolfOptimizer(problem, GWOConfig(pop, iters, seed, **kw)).run()


class TestGWO(unittest.TestCase):
    def test_ranking_for_maximisation_with_deterministic_ties(self):
        order = rank_wolves([1.0, 5.0, 3.0, 5.0, 0.5])
        self.assertEqual(list(order), [1, 3, 2, 0, 4])        # tie 5.0: lower index first

    def test_snapshot_leaders_are_the_top_three(self):
        res = run_gwo(preset_problem())
        for s in res.snapshots:
            top3 = sorted(s.fitness, reverse=True)[:3]
            self.assertAlmostEqual(s.alpha_fitness, top3[0])
            self.assertAlmostEqual(s.beta_fitness, top3[1])
            self.assertAlmostEqual(s.delta_fitness, top3[2])
            self.assertGreaterEqual(s.alpha_fitness, s.beta_fitness)
            self.assertGreaterEqual(s.beta_fitness, s.delta_fitness)

    def test_best_so_far_never_decreases(self):
        res = run_gwo(preset_problem(), iters=80)
        best = [s.best_so_far_fitness for s in res.snapshots]
        self.assertTrue(all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:])))

    def test_seed_reproducibility(self):
        p = preset_problem()
        r1, r2, r3 = run_gwo(p, seed=123), run_gwo(p, seed=123), run_gwo(p, seed=124)
        np.testing.assert_array_equal(r1.best_position, r2.best_position)
        self.assertEqual(r1.best_fitness, r2.best_fitness)
        self.assertFalse(np.array_equal(r1.snapshots[0].positions, r3.snapshots[0].positions))

    def test_every_candidate_and_best_are_feasible(self):
        p = preset_problem()
        res = run_gwo(p)
        self.assertTrue(p.is_feasible(res.best_position))
        for s in res.snapshots:
            self.assertTrue(s.all_feasible)
            self.assertEqual(s.feasible_count, 20)
            self.assertTrue(np.all(s.positions >= 0))
            self.assertTrue(np.all(s.positions.sum(axis=1) <= p.budget + TOL))

    def test_budget_below_total_need(self):
        p = preset_problem(budget=3_000_000)                  # total need is 6.0M
        res = run_gwo(p, iters=80)
        self.assertTrue(p.is_feasible(res.best_position))
        self.assertLessEqual(res.best_position.sum(), p.budget + TOL)
        self.assertLess(p.normalized_fitness(res.best_fitness), 1.0)
        self.assertGreater(res.best_fitness, res.initial_best_fitness - 1e-9)

    def test_budget_above_total_need_leaves_funds_unused(self):
        p = preset_problem(budget=20_000_000)
        res = run_gwo(p, iters=60)
        self.assertTrue(p.is_feasible(res.best_position))
        self.assertLessEqual(res.best_position.sum(), p.targets.sum() + TOL)   # no pointless spending
        self.assertGreater(p.budget - res.best_position.sum(), 0)
        self.assertGreater(p.normalized_fitness(res.best_fitness), 0.9)

    def test_early_stop_and_user_stop(self):
        p = preset_problem()
        res = run_gwo(p, iters=300, patience=5)
        self.assertEqual(res.status, "early_stop")
        self.assertLess(res.iterations_completed, 300)
        ctl = RunControl(); ctl.stop()
        res2 = GreyWolfOptimizer(p, GWOConfig(20, 100, 1)).run(control=ctl)
        self.assertEqual(res2.status, "stopped")
        self.assertFalse(res2.is_complete)

    def test_update_details_follow_the_equations(self):
        res = run_gwo(preset_problem(), iters=5)
        s = res.snapshots[1]
        d = s.details
        a = s.a
        self.assertAlmostEqual(a, 2.0)                        # t = 0 -> a = 2
        self.assertTrue(np.all(np.abs(d["A"]) <= a + 1e-12))  # |A| <= a
        self.assertTrue(np.all((d["C"] >= 0) & (d["C"] <= 2)))
        np.testing.assert_allclose(d["X_new"], d["X_guided"].mean(axis=0))
        k = 0; w = 3
        Xk = d["leader_positions"][k] - d["A"][k][w] * np.abs(
            d["C"][k][w] * d["leader_positions"][k] - d["previous_positions"][w])
        np.testing.assert_allclose(d["X_guided"][k][w], Xk)


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main(module=sys.modules[__name__], verbosity=2)
    else:
        App().mainloop()


if __name__ == "__main__":
    main()