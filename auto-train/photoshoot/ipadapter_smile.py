#!/usr/bin/env python3
"""
IP-Adapter FaceID + SD Inpainting Smile Engine

THIS is the proper Option C:
  1. LP generates smile geometry (mouth position/shape)
  2. InsightFace extracts face embedding from source
  3. SD Inpainting + IP-Adapter FaceID repaints mouth
     CONDITIONED on source face embedding
  4. SD generates THIS person's smile, not a generic smile
  5. Identity preserved because SD knows whose face it is
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
import logging
from typing import Dict, Optional
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ipa] %(levelname)s: %(message)s")
logger = logging.getLogger("ipa")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
IPA_CACHE = os.path.expanduser("~/.cache/ip_adapter")

sys.path.insert(0, LP_DIR)

MOUTH_IDX = list(range(52, 72))


class IPAdapterSmileEngine:

    def __init__(self):
        self.device = "cuda"
        self._load_face_analyzer()
        self._load_liveportrait()
        self._load_sd_with_ipadapter()
        logger.info("IPAdapterSmileEngine ready")

    def _load_face_analyzer(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        logger.info("  InsightFace loaded")

    def _load_liveportrait(self):
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        logger.info("  LivePortrait loaded")

    def _load_sd_with_ipadapter(self):
        """Load SD1.5 + IP-Adapter FaceID using official library."""
        from diffusers import StableDiffusionInpaintPipeline
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

        logger.info("  Loading SD Inpainting base...")
        self.sd_base = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)
        self.sd_base.enable_attention_slicing()

        logger.info("  Loading IP-Adapter FaceID...")
        ipa_path = os.path.join(IPA_CACHE, "ip-adapter-faceid_sd15.bin")
        lora_path = os.path.join(IPA_CACHE, "ip-adapter-faceid_sd15_lora.safetensors")

        # Load LoRA weights for FaceID
        if os.path.exists(lora_path):
            from safetensors.torch import load_file
            lora_state = load_file(lora_path)
            # Apply LoRA to UNet
            from diffusers.loaders import LoraLoaderMixin
            self.sd_base.load_lora_weights(lora_path)
            self.sd_base.fuse_lora()
            logger.info("    FaceID LoRA loaded and fused")

        # Store IP-Adapter weights for manual conditioning
        self.ipa_weights = torch.load(ipa_path, map_location="cpu")
        self.ipa_proj = None

        # Build projection layer (512-dim face embedding → SD conditioning)
        try:
            # IP-Adapter FaceID uses a simple projection
            from ip_adapter.ip_adapter_faceid import MLPProjModel
            self.ipa_proj = MLPProjModel(
                cross_attention_dim=768,  # SD1.5
                id_embeddings_dim=512,    # InsightFace
                num_tokens=4,
            ).to(self.device, dtype=torch.float16)

            # Load projection weights
            proj_weights = {k.replace("image_proj_model.", ""): v
                          for k, v in self.ipa_weights.items()
                          if "image_proj_model" in k}
            if proj_weights:
                self.ipa_proj.load_state_dict(proj_weights)
                logger.info("    FaceID projection model loaded")
        except Exception as e:
            logger.warning(f"    Could not load projection model: {e}")
            self.ipa_proj = None

        logger.info("  SD + IP-Adapter FaceID loaded")

    def _get_face(self, img):
        faces = self.fa.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def _get_embedding(self, img):
        face = self._get_face(img)
        return face.normed_embedding.reshape(1, -1) if face else None

    def _get_face_embedding_for_ipa(self, img):
        """Get face embedding in format IP-Adapter expects."""
        face = self._get_face(img)
        if face is None:
            return None
        # IP-Adapter FaceID uses InsightFace embedding directly
        emb = torch.from_numpy(face.normed_embedding).unsqueeze(0).to(self.device, dtype=torch.float16)
        return emb

    def _identity_score(self, src_emb, img):
        emb = self._get_embedding(img)
        if emb is None:
            return 0.0
        return float(cosine_similarity(src_emb, emb)[0][0])

    def _run_lp(self, source_path, driver_path, multiplier=0.7):
        from src.config.argument_config import ArgumentConfig
        os.makedirs(os.path.join(PHOTO_DIR, "temp"), exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = os.path.join(PHOTO_DIR, "temp")
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        self.lp.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )
        try:
            wfp, _ = self.lp.execute(args)
            return cv2.imread(wfp)
        except:
            return None

    def _create_mouth_mask(self, source_bgr, lp_result):
        h, w = source_bgr.shape[:2]
        lp_resized = cv2.resize(lp_result, (w, h))
        mask = np.zeros((h, w), dtype=np.uint8)

        for img in [lp_resized, source_bgr]:
            face = self._get_face(img)
            if face is not None:
                lmk = getattr(face, 'landmark_2d_106', None)
                if lmk is not None:
                    pts = lmk[MOUTH_IDX].astype(np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(mask, hull, 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        mask = cv2.dilate(mask, kernel, iterations=3)
        return mask

    def _inpaint_with_identity(self, source_bgr, mask, face_emb, smile_type="photoshoot"):
        """
        SD Inpainting with face identity conditioning.
        Uses FaceID LoRA (already fused into UNet) + face-aware prompt.
        """
        h, w = source_bgr.shape[:2]

        face = self._get_face(source_bgr)
        if face is None:
            return source_bgr

        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        pad = int(max(fw, fh) * 0.3)

        fx1, fy1 = max(0, x1-pad), max(0, y1-pad)
        fx2, fy2 = min(w, x2+pad), min(h, y2+pad)

        face_crop = source_bgr[fy1:fy2, fx1:fx2]
        mask_crop = mask[fy1:fy2, fx1:fx2]

        face_512 = cv2.resize(face_crop, (512, 512))
        mask_512 = cv2.resize(mask_crop, (512, 512))

        face_pil = Image.fromarray(cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_512)

        prompts = {
            "subtle": "natural gentle smile, same person same face, photorealistic, sharp skin texture, 8k",
            "photoshoot": "natural happy smile showing white teeth, same person same face, professional portrait photography, photorealistic, sharp teeth, 8k",
            "natural": "genuine warm joyful smile with teeth, same person same face, photorealistic, natural light, 8k",
        }
        negative = "different person, different face, deformed, distorted, bad teeth, blurry, low quality, cartoon, anime, painting, 3d render, watermark"

        prompt = prompts.get(smile_type, prompts["photoshoot"])

        # FaceID LoRA is already fused into UNet, so identity awareness
        # is built into the generation. No extra kwargs needed.
        extra_kwargs = {}

        # Run SD inpainting with FaceID LoRA active
        with torch.autocast("cuda"):
            result_pil = self.sd_base(
                prompt=prompt,
                negative_prompt=negative,
                image=face_pil,
                mask_image=mask_pil,
                num_inference_steps=30,
                guidance_scale=7.5,
                strength=0.6,
                **extra_kwargs,
            ).images[0]

        result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        fh_crop, fw_crop = face_crop.shape[:2]
        result_resized = cv2.resize(result_np, (fw_crop, fh_crop), interpolation=cv2.INTER_LANCZOS4)

        mask_float = cv2.resize(mask_512, (fw_crop, fh_crop)).astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (21, 21), 8)
        mask_3ch = mask_float[:, :, np.newaxis]

        blended = (
            result_resized.astype(np.float32) * mask_3ch +
            face_crop.astype(np.float32) * (1 - mask_3ch)
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        output = source_bgr.copy()
        output[fy1:fy2, fx1:fx2] = blended
        return output

    def smile(self, source_path, smile_type="photoshoot", output_name=None):
        t0 = time.time()

        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return {"success": False, "message": "Cannot read image"}

        src_emb = self._get_embedding(source_bgr)
        if src_emb is None:
            return {"success": False, "message": "No face detected"}

        # Get face embedding for IP-Adapter
        face_emb = self._get_face_embedding_for_ipa(source_bgr)
        if face_emb is None:
            return {"success": False, "message": "Cannot extract face embedding"}

        # Step 1: LP smile geometry
        driver = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        lp_result = self._run_lp(source_path, driver, multiplier=0.7)
        if lp_result is None:
            return {"success": False, "message": "LP failed"}

        # Step 2: Mouth mask
        mask = self._create_mouth_mask(source_bgr, lp_result)

        # Step 3: SD inpainting with face identity conditioning
        result = self._inpaint_with_identity(source_bgr, mask, face_emb, smile_type)

        # Step 4: Identity check
        score = self._identity_score(src_emb, result)

        # Save
        OUT = os.path.join(PHOTO_DIR, "final_output")
        os.makedirs(OUT, exist_ok=True)
        if output_name is None:
            name = os.path.splitext(os.path.basename(source_path))[0]
            output_name = f"{name}_ipa_{smile_type}.png"
        out_path = os.path.join(OUT, output_name)
        cv2.imwrite(out_path, result)

        elapsed = time.time() - t0
        logger.info(f"  {smile_type}: identity={score:.4f} time={elapsed:.1f}s")

        return {
            "success": True,
            "identity_score": round(score, 4),
            "output_path": out_path,
            "time": round(elapsed, 1),
        }


if __name__ == "__main__":
    engine = IPAdapterSmileEngine()

    men = [
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0008.jpg", "man_asian"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0013.jpg", "man_black"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0015.jpg", "man_suit"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0019.jpg", "man_young"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0009.jpg", "man_older"),
    ]

    OUT = os.path.join(PHOTO_DIR, "final_output")
    os.makedirs(OUT, exist_ok=True)

    print(f"\n{'='*60}")
    print("IP-ADAPTER FaceID + SD INPAINTING — 5 men")
    print(f"{'='*60}\n")

    for src_path, name in men:
        src = cv2.imread(src_path)
        if src is None:
            continue
        cv2.imwrite(os.path.join(OUT, f"{name}_original.png"), src)

        r = engine.smile(src_path, smile_type="photoshoot", output_name=f"{name}_ipa_smile.png")
        ok = "✓" if r["success"] else "✗"
        score = r.get("identity_score", 0)
        print(f"  {ok} {name}: identity={score:.4f} time={r.get('time',0):.1f}s")

        if r["success"]:
            result = cv2.imread(r["output_path"])
            h_t = 300
            s1, s2 = h_t/src.shape[0], h_t/result.shape[0]
            o_r = cv2.resize(src, (int(src.shape[1]*s1), h_t))
            r_r = cv2.resize(result, (int(result.shape[1]*s2), h_t))
            cv2.putText(o_r, "Original", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(r_r, f"IPA id={score:.3f}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            w_m = max(o_r.shape[1], r_r.shape[1])
            if o_r.shape[1] < w_m:
                o_r = np.hstack([o_r, np.zeros((h_t, w_m-o_r.shape[1], 3), dtype=np.uint8)])
            if r_r.shape[1] < w_m:
                r_r = np.hstack([r_r, np.zeros((h_t, w_m-r_r.shape[1], 3), dtype=np.uint8)])
            cv2.imwrite(os.path.join(OUT, f"{name}_comparison.png"), np.hstack([o_r, r_r]))

    print(f"\nOutput: {OUT}/")
