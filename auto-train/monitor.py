"""
Live Training Monitor
Run: python monitor.py
Refreshes every 10 seconds. Press Ctrl+C to exit.
"""
import os, sys, time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
LOG  = "/tmp/loop.log"
HIST = ROOT / "training_engine/logs/loop_history.json"

def clear(): os.system("clear")

def count_images(folder):
    p = Path(folder)
    if not p.exists(): return 0
    return len([f for f in p.iterdir() if f.suffix in ('.jpg','.jpeg','.png')])

def tail_log(path, n=200):
    try:
        lines = Path(path).read_text().splitlines()
        return lines[-n:]
    except: return []

def get_current_step():
    lines = tail_log(LOG)
    for line in reversed(lines):
        if "arcface_score" in line or "identity_sim" in line or "cosine_sim" in line:
            try:
                d = json.loads(line)
                step  = d.get("step", 0)
                score = d.get("arcface_score", d.get("identity_sim", d.get("cosine_sim", 0)))
                loss  = d.get("total", d.get("identity_loss", 0))
                return step, float(score), float(loss)
            except: pass
    return 0, 0.0, 0.0

def get_loop_stage():
    lines = tail_log(LOG)
    stage  = "unknown"
    cycle  = 1
    pairs  = 0
    val_id = 0.0
    best   = 0.0

    for line in reversed(lines):
        if "[LOOP]" not in line: continue
        if "SCRAPE →"      in line and stage == "unknown": stage = "SCRAPING"
        if "CLEAN →"       in line and stage == "unknown": stage = "CLEANING"
        if "PAIR →"        in line and stage == "unknown": stage = "PAIRING"
        if "TRAIN →"       in line and stage == "unknown": stage = "TRAINING"
        if "VALIDATE →"    in line and stage == "unknown": stage = "VALIDATING"
        if "pairs so far"  in line and pairs == 0:
            try: pairs = int(line.split()[3])
            except: pass
        if "PAIR done:"    in line and pairs == 0:
            try: pairs = int(line.split()[4])
            except: pass
        if "CYCLE"         in line and cycle == 1:
            try: cycle = int(line.split("CYCLE")[1].split("/")[0].strip())
            except: pass
        if "avg_identity=" in line and val_id == 0:
            try: val_id = float(line.split("avg_identity=")[1].split()[0])
            except: pass
        if "New best:"     in line and best == 0:
            try: best = float(line.split("New best:")[1].strip())
            except: pass
        if stage != "unknown": break

    return stage, cycle, pairs, val_id, best

def get_identity_history():
    """Get identity score trend from all training runs."""
    scores = []
    # From training metrics
    for mfile in ["training_metrics.jsonl", "training_scraped.jsonl", "eval_metrics.jsonl"]:
        p = ROOT / "training_engine/logs" / mfile
        if not p.exists(): continue
        try:
            lines = p.read_text().splitlines()
            for l in lines[-50:]:
                d = json.loads(l)
                s = d.get("arcface_score", d.get("identity_sim", d.get("mean_identity", 0)))
                if s > 0: scores.append(s)
        except: pass
    return scores

def bar(val, total=1.0, width=30, fill="█", empty="░"):
    filled = int(val / total * width)
    return fill * filled + empty * (width - filled)

def render():
    clear()
    now = datetime.now().strftime("%H:%M:%S")

    stage, cycle, pairs, val_id, best = get_loop_stage()
    train_step, train_score, train_loss = get_current_step()
    id_history = get_identity_history()

    # Dataset counts
    smile_raw   = count_images(ROOT / "raw_data/model_photos/smile")
    neutral_raw = count_images(ROOT / "raw_data/model_photos/neutral")
    smile_clean = count_images(ROOT / "raw_data/cleaned/smile")
    neut_clean  = count_images(ROOT / "raw_data/cleaned/neutral")
    n_pairs     = max(pairs, len(list((ROOT / "raw_data/pairs").glob("*_src.jpg"))) if (ROOT / "raw_data/pairs").exists() else 0)

    # Best ever
    best_ever = max(id_history + [best, val_id, 0.9809])
    target    = 0.995
    gap       = target - best_ever

    print(f"{'='*58}")
    print(f"  PHOTOSHOOT AI — LIVE MONITOR          {now}")
    print(f"{'='*58}")

    # Overall progress
    pct = min(best_ever / target, 1.0)
    print(f"\n  IDENTITY PROGRESS")
    print(f"  [{bar(pct)}] {best_ever*100:.2f}%")
    print(f"  Target: {target*100:.1f}%   Gap: {gap*100:.2f}%   Best ever: {best_ever*100:.2f}%")

    # Current stage
    stage_icons = {
        "SCRAPING": "🔍", "CLEANING": "🧹", "PAIRING": "🔗",
        "TRAINING": "🏋", "VALIDATING": "📊", "unknown": "⏳"
    }
    icon = stage_icons.get(stage, "⏳")
    print(f"\n  LOOP STATUS")
    print(f"  Cycle   : {cycle} / 20")
    print(f"  Stage   : {icon}  {stage}")

    # Dataset
    print(f"\n  DATASET")
    print(f"  Smile   : {smile_raw} raw  →  {smile_clean} clean")
    print(f"  Neutral : {neutral_raw} raw  →  {neut_clean} clean")
    print(f"  Pairs   : {n_pairs}  (identity-verified ≥ 0.90)")

    # Training
    if train_step > 0:
        print(f"\n  TRAINING  (cycle {cycle})")
        print(f"  Step    : {train_step} / 5000")
        prog = train_step / 5000
        print(f"  [{bar(prog)}] {prog*100:.0f}%")
        print(f"  ArcFace : {train_score:.4f}  ({train_score*100:.1f}%)")
        print(f"  Loss    : {train_loss:.4f}")
    elif stage == "PAIRING":
        print(f"\n  PAIRING")
        print(f"  Pairs generated : {n_pairs}")
        prog = min(n_pairs / 150, 1.0)
        print(f"  [{bar(prog)}] {prog*100:.0f}% of target")

    # API
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8001/health", timeout=1) as r:
            api_status = "✓ LIVE"
    except:
        api_status = "✗ DOWN"
    print(f"\n  SERVICES")
    print(f"  Production API (port 8001) : {api_status}")
    print(f"  Training loop              : {'✓ RUNNING' if stage != 'unknown' else '✗ STOPPED'}")

    # Score history sparkline
    if id_history:
        recent = id_history[-20:]
        mn, mx = min(recent), max(recent)
        sparks = "▁▂▃▄▅▆▇█"
        line = ""
        for v in recent:
            idx = int((v - mn) / max(mx - mn, 0.001) * 7)
            line += sparks[idx]
        print(f"\n  IDENTITY TREND (last {len(recent)} evals)")
        print(f"  {line}  {recent[-1]*100:.1f}%")

    print(f"\n{'='*58}")
    print(f"  Refreshing every 10s  |  Ctrl+C to exit")
    print(f"{'='*58}")

if __name__ == "__main__":
    print("Starting monitor... (Ctrl+C to stop)")
    try:
        while True:
            render()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
