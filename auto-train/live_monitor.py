"""
Live training monitor for the terminal.
No port forwarding needed.

Usage:
    python live_monitor.py
"""
import json, time, os, sys
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent
METRICS = ROOT / "training_engine/logs/training_metrics.jsonl"
LOOP_LOG = "/tmp/loop_lipmode8.log"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR = "\033[H\033[J"


def sparkline(values, width=40):
    if not values:
        return "─" * width
    chars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1
    out = ""
    step = max(1, len(values) // width)
    samples = values[::step][-width:]
    for v in samples:
        idx = int((v - lo) / rng * (len(chars) - 1))
        out += chars[max(0, min(len(chars) - 1, idx))]
    return out


def progress_bar(value, target=0.95, width=50):
    pct = min(1.0, value / target)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if value >= target else (YELLOW if value >= target * 0.9 else CYAN)
    return f"{color}{bar}{RESET} {value*100:5.2f}% / {target*100:.1f}%"


def get_loop_status():
    if not os.path.exists(LOOP_LOG):
        return "log not found"
    try:
        with open(LOOP_LOG) as f:
            lines = f.readlines()
        for line in reversed(lines[-200:]):
            line = line.strip()
            if not line:
                continue
            if "[LOOP]" in line:
                stage = line.split("[LOOP]", 1)[-1].strip()
                if stage and not stage.startswith("="):
                    return stage[:80]
        return "starting..."
    except Exception as e:
        return f"err: {e}"


def main():
    print(CLEAR, end="")
    history = deque(maxlen=200)
    last_size = 0

    while True:
        try:
            print(CLEAR, end="")
            print(f"{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}{CYAN}║          LP TRAINING — LIVE MONITOR                            ║{RESET}")
            print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════╝{RESET}\n")

            # Loop status
            loop_status = get_loop_status()
            print(f"{BOLD}Loop:{RESET} {loop_status}\n")

            # Read metrics
            if METRICS.exists():
                size = METRICS.stat().st_size
                if size != last_size:
                    history.clear()
                    with open(METRICS) as f:
                        for line in f:
                            try:
                                history.append(json.loads(line))
                            except Exception:
                                pass
                    last_size = size

            if not history:
                print(f"{DIM}Waiting for first training step...{RESET}")
                print(f"{DIM}(metrics file: {METRICS}){RESET}\n")
                print(f"{DIM}Refreshes every 3s — Ctrl+C to quit{RESET}")
                time.sleep(3)
                continue

            latest = history[-1]
            step = latest.get("step", 0)
            arc = latest.get("arcface_score", latest.get("identity", 0))
            total_loss = latest.get("total", 0)
            expr_loss = latest.get("expression", 0)
            perc_loss = latest.get("perceptual", 0)
            pixel_loss = latest.get("pixel", 0)

            print(f"{BOLD}Step:{RESET}    {step}")
            print(f"{BOLD}Identity:{RESET}")
            print(f"  {progress_bar(arc, target=0.95)}\n")

            print(f"{BOLD}Losses (latest):{RESET}")
            print(f"  total       {total_loss:.4f}")
            print(f"  expression  {expr_loss:.4f}")
            print(f"  perceptual  {perc_loss:.4f}")
            print(f"  pixel       {pixel_loss:.4f}\n")

            # Sparklines from history
            arc_history = [h.get("arcface_score", h.get("identity", 0)) for h in history]
            loss_history = [h.get("total", 0) for h in history]

            print(f"{BOLD}Trend (last {len(arc_history)} evals):{RESET}")
            print(f"  identity ↑ {GREEN}{sparkline(arc_history)}{RESET}")
            print(f"  loss     ↓ {YELLOW}{sparkline(loss_history)}{RESET}\n")

            # Best so far
            best = max(arc_history) if arc_history else 0
            best_step_idx = arc_history.index(best) if arc_history else 0
            best_step = history[best_step_idx].get("step", 0)
            print(f"{BOLD}Best identity so far:{RESET} {GREEN}{best*100:.2f}%{RESET} @ step {best_step}\n")

            print(f"{DIM}Refreshes every 3s — Ctrl+C to quit{RESET}")
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break
        except Exception as e:
            print(f"\nMonitor error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
