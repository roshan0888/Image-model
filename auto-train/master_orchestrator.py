"""
Master Orchestrator
===================
The single production entrypoint that:
  1. Analyzes input photo (face quality gate)
  2. Routes to the right engine (LP lip-mode or InstantID img2img)
  3. Validates output identity (ArcFace gate)
  4. Retries with different params if quality is below threshold
  5. Optionally applies background change
  6. Returns final image + confidence + metadata

Usage:
    from master_orchestrator import MasterOrchestrator

    orch = MasterOrchestrator()
    result = orch.process(
        source_path="user_photo.jpg",
        expression="smile",
        background="studio_gray",  # or None to keep original
    )
    # result = {
    #   'image':           PIL Image,
    #   'output_path':     str,
    #   'identity_score':  0.945,
    #   'engine_used':     'lp_lip',
    #   'confidence':      'high',
    #   'category':        'frontal',
    #   'time':            5.2,
    #   'metadata':        {...}
    # }
"""

import os, sys, cv2, time, json, logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [ORCH] %(message)s")

OUTPUT_DIR = ROOT / "production" / "orchestrator_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USAGE_LOG = ROOT / "production" / "orchestrator_usage.jsonl"


# ── Quality thresholds ────────────────────────────────────────────────────────

# Identity score required to PASS each engine's output
IDENTITY_THRESHOLD = {
    "lp_lip":            0.88,   # Lip-mode trades 3-5% identity for real expression
    "instantid_img2img": 0.82,   # InstantID is inherently lower
}

# Max retries before giving up
MAX_RETRIES = 3


class MasterOrchestrator:

    def __init__(self):
        self._gate = None
        self._lp_pipeline = None
        self._iid_engine = None
        self._bg_pipeline = None
        log.info("MasterOrchestrator initialized (lazy loading)")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        source_path: str,
        expression: str = "smile",
        background: Optional[str] = None,
        force_engine: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Main entrypoint. Process a photo end-to-end.

        Args:
            source_path: Input photo path
            expression: "smile" | "open_smile" | "neutral"
            background: Optional background preset (e.g. "studio_gray")
            force_engine: Override routing — "lp_lip" | "instantid_img2img"
            output_path: Optional output path

        Returns:
            Result dict with image, identity_score, engine_used, etc.
        """
        t0 = time.time()
        log.info(f"Processing: {Path(source_path).name} → expression={expression}, bg={background}")

        # ── Step 1: Face Quality Gate ─────────────────────────────────────
        gate = self._get_gate()
        classification = gate.classify(source_path)

        log.info(f"  Gate: category={classification.category}, engine={classification.engine}, "
                 f"yaw={classification.yaw:+.1f}°, conf={classification.confidence:.2f}")

        if classification.engine == "reject":
            return self._build_failure(
                source_path, "rejected_at_gate",
                f"Cannot process: {classification.reject_reason}",
                classification, t0,
            )

        # Engine selection (allow override)
        engine = force_engine or classification.engine
        log.info(f"  Selected engine: {engine}")

        # ── Step 2: Generate (with retries) ───────────────────────────────
        best_result = None
        for attempt in range(MAX_RETRIES):
            try:
                if engine == "lp_lip":
                    out_img, score = self._run_lp_lip(source_path, expression, attempt)
                elif engine == "instantid_img2img":
                    out_img, score = self._run_instantid(source_path, expression, attempt)
                else:
                    raise ValueError(f"Unknown engine: {engine}")

                if out_img is None:
                    log.warning(f"  Attempt {attempt+1}: engine returned None")
                    continue

                threshold = IDENTITY_THRESHOLD[engine]
                log.info(f"  Attempt {attempt+1}: identity={score:.4f} (need ≥{threshold})")

                if best_result is None or score > best_result["identity_score"]:
                    best_result = {
                        "image": out_img,
                        "identity_score": score,
                        "attempt": attempt + 1,
                    }

                if score >= threshold:
                    break  # Good enough, stop retrying

            except Exception as e:
                log.warning(f"  Attempt {attempt+1} failed: {e}")
                continue

        if best_result is None:
            return self._build_failure(
                source_path, "all_attempts_failed",
                f"All {MAX_RETRIES} attempts failed", classification, t0,
            )

        # ── Step 3: Optional background change ────────────────────────────
        final_img = best_result["image"]
        if background:
            try:
                bg_pipe = self._get_bg_pipeline()
                final_img = bg_pipe.replace_background(final_img, background)
                log.info(f"  Background applied: {background}")
            except Exception as e:
                log.warning(f"  Background failed: {e}")

        # ── Step 4: Save & log ────────────────────────────────────────────
        if output_path is None:
            stem = Path(source_path).stem
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(OUTPUT_DIR / f"{stem}_{expression}_{ts}.jpg")

        if isinstance(final_img, np.ndarray):
            cv2.imwrite(output_path, final_img)
            pil_out = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        else:
            pil_out = final_img
            pil_out.save(output_path, quality=95)

        total_time = time.time() - t0

        # Confidence label
        score = best_result["identity_score"]
        if score >= 0.93:
            conf_label = "high"
        elif score >= 0.88:
            conf_label = "medium"
        elif score >= 0.82:
            conf_label = "low"
        else:
            conf_label = "marginal"

        result = {
            "image": pil_out,
            "output_path": output_path,
            "identity_score": score,
            "engine_used": engine,
            "confidence": conf_label,
            "category": classification.category,
            "yaw": classification.yaw,
            "attempts": best_result["attempt"],
            "background_applied": background is not None,
            "time": total_time,
            "status": "success",
        }

        self._log_usage(source_path, result)
        log.info(f"  ✓ Done in {total_time:.1f}s — identity={score:.4f} ({conf_label})")
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # ENGINE WRAPPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _run_lp_lip(self, source_path: str, expression: str, attempt: int):
        """LP in production lip-mode."""
        pipeline = self._get_lp_pipeline()

        # Driver library (verified to produce visible smiles)
        LP_DRIVERS = ROOT.parent / "LivePortrait/assets/examples/driving"
        DRIVER_OPTS = {
            "smile":      [(LP_DRIVERS / "d12.jpg", 1.3),
                          (LP_DRIVERS / "d12.jpg", 1.5),
                          (LP_DRIVERS / "d30.jpg", 1.4)],
            "open_smile": [(LP_DRIVERS / "d12.jpg", 1.5),
                          (LP_DRIVERS / "d12.jpg", 1.7),
                          (LP_DRIVERS / "d12.jpg", 1.3)],
            "neutral":    [(LP_DRIVERS / "d12.jpg", 0.0)],
        }

        opts = DRIVER_OPTS.get(expression, DRIVER_OPTS["smile"])
        driver, mult = opts[min(attempt, len(opts) - 1)]

        if not driver.exists():
            raise FileNotFoundError(f"Driver missing: {driver}")

        out = pipeline._run_lp(
            source_path, str(driver),
            multiplier=mult, region="lip", use_retargeting=False,
        )
        if out is None:
            return None, 0.0

        # Measure identity
        src = cv2.imread(source_path)
        f1 = pipeline.face_analyzer.get(src)
        f2 = pipeline.face_analyzer.get(out)
        if not f1 or not f2:
            return out, 0.0
        score = float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))
        return out, score

    def _run_instantid(self, source_path: str, expression: str, attempt: int):
        """InstantID img2img mode."""
        engine = self._get_iid_engine()

        # Vary strength per attempt
        STRENGTHS = [0.30, 0.35, 0.40]
        strength = STRENGTHS[min(attempt, len(STRENGTHS) - 1)]

        result = engine.generate_img2img(
            source_path,
            expression=expression,
            pose="frontal",
            background="studio_gray",
            strength=strength,
            num_steps=20,
            ip_adapter_scale=0.8,
            controlnet_scale=0.7,
        )
        return result["image_cv"], result["identity_score"]

    # ══════════════════════════════════════════════════════════════════════════
    # LAZY LOADERS
    # ══════════════════════════════════════════════════════════════════════════

    def _get_gate(self):
        if self._gate is None:
            from face_quality_gate import FaceQualityGate
            self._gate = FaceQualityGate()
        return self._gate

    def _get_lp_pipeline(self):
        if self._lp_pipeline is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("npipe", ROOT / "natural_pipeline.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._lp_pipeline = mod.NaturalExpressionPipeline()
        return self._lp_pipeline

    def _get_iid_engine(self):
        if self._iid_engine is None:
            sys.path.insert(0, str(ROOT / "instantid_pipeline" / "repo"))
            from instantid_pipeline.instantid_engine import InstantIDEngine
            self._iid_engine = InstantIDEngine(low_vram=True)
        return self._iid_engine

    def _get_bg_pipeline(self):
        if self._bg_pipeline is None:
            from photoshoot.background.background_pipeline import BackgroundPipeline
            self._bg_pipeline = BackgroundPipeline()
        return self._bg_pipeline

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _build_failure(self, source_path, status, reason, classification, t0):
        return {
            "image": None,
            "output_path": None,
            "identity_score": 0.0,
            "engine_used": None,
            "confidence": "rejected",
            "category": classification.category if classification else "unknown",
            "yaw": classification.yaw if classification else 0,
            "status": status,
            "reason": reason,
            "time": time.time() - t0,
        }

    def _log_usage(self, source_path: str, result: Dict):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "source": Path(source_path).name,
                "engine": result.get("engine_used"),
                "category": result.get("category"),
                "yaw": result.get("yaw"),
                "identity": result.get("identity_score"),
                "confidence": result.get("confidence"),
                "attempts": result.get("attempts"),
                "time": result.get("time"),
                "status": result.get("status"),
            }
            with open(USAGE_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.debug(f"Usage log failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Input image")
    p.add_argument("-e", "--expression", default="smile")
    p.add_argument("-b", "--background", default=None)
    p.add_argument("--force-engine", default=None, choices=["lp_lip", "instantid_img2img"])
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    orch = MasterOrchestrator()
    result = orch.process(
        args.source,
        expression=args.expression,
        background=args.background,
        force_engine=args.force_engine,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"  ORCHESTRATOR RESULT")
    print(f"{'='*60}")
    if result["status"] == "success":
        print(f"  Status:        ✓ {result['status']}")
        print(f"  Engine:        {result['engine_used']}")
        print(f"  Category:      {result['category']} (yaw={result['yaw']:+.1f}°)")
        print(f"  Identity:      {result['identity_score']:.4f}")
        print(f"  Confidence:    {result['confidence']}")
        print(f"  Attempts:      {result['attempts']}")
        print(f"  Time:          {result['time']:.1f}s")
        print(f"  Output:        {result['output_path']}")
    else:
        print(f"  Status:        ✗ {result['status']}")
        print(f"  Reason:        {result.get('reason', 'unknown')}")
        print(f"  Category:      {result.get('category')}")
        print(f"  Time:          {result['time']:.1f}s")
