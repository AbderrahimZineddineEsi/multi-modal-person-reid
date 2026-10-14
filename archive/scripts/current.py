%%capture

!pip install gradio ultralytics boxmot opencv-python scipy torch torchvision

!pip install pinecone pymongo matplotlib insightface onnxruntime-gpu imagekitio

print("Dependencies Ready.")
import os, sys, cv2, torch, pickle, time, shutil, json, io, base64

import concurrent.futures

import numpy as np

import urllib.request

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import torch.nn.functional as F

import gradio as gr

from pathlib import Path

from datetime import datetime, timezone

from sklearn.cluster import DBSCAN

from collections import defaultdict

from pinecone import Pinecone

from pymongo import MongoClient

from imagekitio import ImageKit

from tqdm.auto import tqdm

from ultralytics import RTDETR, YOLO



import boxmot, pkgutil, inspect

from boxmot.trackers.botsort.botsort import BotSort

from insightface.app import FaceAnalysis



try:

    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()

    PINECONE_API_KEY     = secrets.get_secret("PINECONE_API_KEY")

    MONGO_URI            = secrets.get_secret("MONGO_URI")

    IMAGEKIT_PRIVATE_KEY = secrets.get_secret("IMAGEKIT_PRIVATE_KEY")

    IMAGEKIT_PUBLIC_KEY  = secrets.get_secret("IMAGEKIT_PUBLIC_KEY")

    IMAGEKIT_URL         = secrets.get_secret("IMAGEKIT_URL_ENDPOINT")

except Exception:

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

    MONGO_URI        = os.environ.get("MONGO_URI", "")

    IMAGEKIT_PRIVATE_KEY = os.environ.get("IMAGEKIT_PRIVATE_KEY", "")

    IMAGEKIT_PUBLIC_KEY  = os.environ.get("IMAGEKIT_PUBLIC_KEY", "")

    IMAGEKIT_URL         = os.environ.get("IMAGEKIT_URL_ENDPOINT", "")



PINECONE_INDEX_NAME      = "supermarket-reid"

PINECONE_FACE_INDEX_NAME = "supermarket-faces"

OUTPUT_DIR   = Path('/kaggle/working/module4_output')

DB_IMAGES_DIR = OUTPUT_DIR / 'saved_identities'

CAMERA_ID = "cam_01"



print("✅ Imports & credentials loaded.")
n_gpus = torch.cuda.device_count()

DET_DEVICE  = 'cuda:0'

POSE_DEVICE = 'cuda:1' if n_gpus > 1 else 'cuda:0'

REID_DEVICE = 'cuda:1' if n_gpus > 1 else 'cuda:0'



print(f"GPUs: {n_gpus} | Det: {DET_DEVICE} | Pose+ReID: {REID_DEVICE}")



print("Loading RT-DETR...")

detector = RTDETR('rtdetr-l.pt').to(DET_DEVICE)



print("Loading BotSort tracker...")

REID_WEIGHTS = "osnet_x0_25_msmt17.pt"



def _make_botsort_tracker():

    import inspect

    from pathlib import Path

    from boxmot.trackers.botsort.botsort import BotSort

    

    # Fully solve the 404 issue without bypassing BotSort's capabilities:

    # Instead of osnet, use the `clip_market1501.pt` model. We KNOW this asset 

    # natively downloads successfully on Kaggle (it's initialized right below

    # by the ReID block). This gives BotSort a highly accurate tracker model 

    # without network errors!

    weights_path = Path("clip_market1501.pt")



    sig = inspect.signature(BotSort.__init__)

    kwargs = {}

    

    if "model_weights" in sig.parameters: kwargs["model_weights"] = weights_path

    elif "reid_weights" in sig.parameters: kwargs["reid_weights"] = str(weights_path)

    elif "weights" in sig.parameters: kwargs["weights"] = weights_path

    elif "reid_model" in sig.parameters: kwargs["reid_model"] = reid_model.model  



    if "device" in sig.parameters: kwargs["device"] = DET_DEVICE

    

    if "fp16" in sig.parameters: kwargs["fp16"] = False

    elif "half" in sig.parameters: kwargs["half"] = False



    tracker = BotSort(**kwargs)

    return tracker



tracker_template = _make_botsort_tracker



print("Loading YOLOv8-Pose...")

pose_model = YOLO('yolo11s-pose.pt').to(POSE_DEVICE)



print("Loading CLIP-ReID...")

ReID = None

for importer, modname, ispkg in pkgutil.walk_packages(boxmot.__path__, boxmot.__name__ + '.'):

    try:

        mod = __import__(modname, fromlist="dummy")

        for name, obj in inspect.getmembers(mod):

            if inspect.isclass(obj) and name in ['ReID', 'ReIDDetectMultiBackend']:

                ReID = obj; break

    except Exception: continue

    if ReID is not None: break

reid_model = ReID(weights='clip_market1501.pt', device=REID_DEVICE, half=False)



print("Loading InsightFace/ArcFace...")

face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

face_app.prepare(ctx_id=0, det_size=(640, 640))



# Cloud services

print("Connecting to cloud services...")

imagekit_client = ImageKit(private_key=IMAGEKIT_PRIVATE_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_body_index = pc.Index(PINECONE_INDEX_NAME)

pinecone_face_index = pc.Index(PINECONE_FACE_INDEX_NAME)

mongo_client = MongoClient(MONGO_URI)

db = mongo_client["supermarket_db"]

persons_col = db["persons"]



print("✅ All models & services ready.")
# ═══════════════════════════════════════════════════════════════

# HELPER FUNCTIONS

# ═══════════════════════════════════════════════════════════════



def extract_reid_features(crops):

    with torch.no_grad():

        embs = reid_model(crops)

        if isinstance(embs, np.ndarray): embs = torch.tensor(embs)

        return F.normalize(embs, p=2, dim=1).to(REID_DEVICE)





def embed_body_image(img_bgr):

    """Embed a single body crop → 1D tensor."""

    with torch.no_grad():

        emb = reid_model([img_bgr])

        if isinstance(emb, np.ndarray): emb = torch.tensor(emb)

        return F.normalize(emb, p=2, dim=1).to(REID_DEVICE)





def embed_face_image(img_bgr):

    """Detect face → return (512-d numpy, det_score) or (None, 0)."""

    faces = face_app.get(img_bgr)

    if not faces: return None, 0.0

    best = max(faces, key=lambda f: f.det_score)

    return best.normed_embedding, float(best.det_score)





def embed_face_crop(img_bgr):

    """For pre-cropped face chips: pad into larger canvas first."""

    h, w = img_bgr.shape[:2]

    pad = max(h, w)

    canvas = np.full((h + 2*pad, w + 2*pad, 3), 128, dtype=np.uint8)

    canvas[pad:pad+h, pad:pad+w] = img_bgr

    return embed_face_image(canvas)





def get_spatially_matched_face(body_box, faces):

    """Match face to body: face center must be in top 40% of body box."""

    bx1, by1, bx2, by2 = body_box

    for f in faces:

        fx1, fy1, fx2, fy2 = f.bbox

        face_cx = (fx1 + fx2) / 2

        face_cy = (fy1 + fy2) / 2

        if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by1 + 0.4 * (by2 - by1):

            return f

    return None





def search_face_pinecone(face_emb, threshold=0.55):

    """Query Pinecone face index → return global_id or None."""

    try:

        resp = pinecone_face_index.query(vector=face_emb.tolist(), top_k=1, include_metadata=True)

        if resp.matches and resp.matches[0].score >= threshold:

            return int(resp.matches[0].metadata["global_id"])

    except Exception:

        pass

    return None





def create_person_document(gid, first_seen_ts, col):

    if not col.find_one({"global_id": gid}):

        col.insert_one({

            "global_id": gid,

            "first_seen_ts": float(first_seen_ts),

            "last_seen_ts": float(first_seen_ts),

            "faces": [],

            "sightings": []

        })





def fetch_image_from_url(url, as_rgb=False):

    try:

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=10) as response:

            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if as_rgb and img is not None:

                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

    except Exception:

        return None





# ═══════════════════════════════════════════════════════════════

# SIGHTING TRACKER

# ═══════════════════════════════════════════════════════════════

class SightingTracker:

    def __init__(self, fps=30, video_start_time=None):

        self.fps = fps

        self.video_start_time = video_start_time or datetime.now(timezone.utc)

        self.active_sightings = {}

        self.completed_sightings = []

        self.active_faces = {}

        self.completed_faces = []



    def _frame_to_ts(self, frame):

        return self.video_start_time.timestamp() + frame / self.fps



    def update_track(self, tid, frame, conf, ratio, img, gid=None, bbox=None):

        if tid not in self.active_sightings:

            self.active_sightings[tid] = {

                "camera_id": CAMERA_ID, "track_id": tid, "global_id": gid,

                "entry_ts": self._frame_to_ts(frame), "exit_ts": self._frame_to_ts(frame),

                "duration_sec": 0.0, "frame_start": frame, "frame_end": frame,

                "n_detections": 1, "sum_det_score": float(conf), "sum_skeleton_ratio": float(ratio),

                "best_crop_quality": float(conf * ratio), "best_crop_img": img,

                "best_crop_bbox": list(bbox) if bbox else None, "best_crop_frame": frame

            }

        else:

            s = self.active_sightings[tid]

            if gid is not None and s["global_id"] is None: s["global_id"] = gid

            s["exit_ts"] = self._frame_to_ts(frame)

            s["frame_end"] = frame

            s["duration_sec"] = s["exit_ts"] - s["entry_ts"]

            s["n_detections"] += 1

            s["sum_det_score"] += float(conf)

            s["sum_skeleton_ratio"] += float(ratio)

            q = float(conf * ratio)

            if q > s["best_crop_quality"]:

                s["best_crop_quality"] = q

                s["best_crop_img"] = img

                s["best_crop_bbox"] = list(bbox) if bbox else None

                s["best_crop_frame"] = frame



    def update_face(self, tid, face_img, face_emb, face_conf):

        """Track face data separately from body sightings."""

        fc = float(face_conf)

        if tid not in self.active_faces:

            self.active_faces[tid] = {

                "best_face_img": face_img,

                "best_face_emb": face_emb,

                "best_face_conf": fc

            }

        elif fc > self.active_faces[tid]["best_face_conf"]:

            self.active_faces[tid]["best_face_img"] = face_img

            self.active_faces[tid]["best_face_emb"] = face_emb

            self.active_faces[tid]["best_face_conf"] = fc



    def assign_global_id(self, tid, gid):

        if tid in self.active_sightings:

            self.active_sightings[tid]["global_id"] = gid



    def close_track(self, tid):

        if tid in self.active_sightings:

            s = self.active_sightings.pop(tid)

            face = self.active_faces.pop(tid, None)

            if s["global_id"] is not None:

                s["avg_det_score"] = float(s["sum_det_score"] / s["n_detections"])

                s["avg_skeleton_ratio"] = float(s["sum_skeleton_ratio"] / s["n_detections"])

                del s["sum_det_score"]; del s["sum_skeleton_ratio"]

                self.completed_sightings.append(s)

                if face is not None:

                    face["global_id"] = s["global_id"]

                    face["track_id"] = tid

                    face["camera_id"] = s["camera_id"]

                    face["timestamp"] = s["exit_ts"]

                    self.completed_faces.append(face)





# ═══════════════════════════════════════════════════════════════

# UNKNOWN DATA MANAGER (DBSCAN)

# ═══════════════════════════════════════════════════════════════

class UnknownDataManager:

    def __init__(self, eps, min_samples):

        self.pool = []

        self.eps = eps

        self.min_samples = min_samples



    def add(self, item):

        self.pool.append(item)



    def run_dbscan(self, gallery):

        if len(self.pool) < self.min_samples: return []

        embs = torch.stack([x['emb'] for x in self.pool])

        sim = torch.mm(embs, embs.t()).cpu().numpy()

        dist = np.clip(1.0 - sim, 0.0, 2.0)

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit(dist).labels_

        actions, new_pool = [], []

        for lbl in set(labels):

            if lbl == -1: continue

            indices = np.where(labels == lbl)[0]

            items = [self.pool[i] for i in indices]

            tids = list(set([x['track_id'] for x in items]))

            forced = [x.get('forced_gid') for x in items if x.get('forced_gid') is not None]

            if forced:

                gid = max(set(forced), key=forced.count)

                if gid not in gallery.classes: gallery.mint_class_at(gid)

                for it in items: gallery.add_observation(gid, it)

                actions.append(("FACE_VIP_BYPASS", gid, len(items), tids)); continue

            cluster_mean = F.normalize(torch.mean(torch.stack([x['emb'] for x in items]), dim=0), p=2, dim=0)

            merged = False

            if gallery.num_classes() > 0:

                centroids, active_gids = gallery.get_weighted_centroids()

                if centroids is not None:

                    sims = torch.mv(centroids, cluster_mean)

                    best_sim, best_idx = torch.max(sims, dim=0)

                    if (1.0 - best_sim.item()) <= self.eps:

                        gid = active_gids[best_idx.item()]

                        for it in items: gallery.add_observation(gid, it)

                        actions.append(("MERGE", gid, len(items), tids)); merged = True

            if not merged:

                gid = gallery.mint_new_class()

                for it in items: gallery.add_observation(gid, it)

                actions.append(("MINT", gid, len(items), tids))

        for i, lbl in enumerate(labels):

            if lbl == -1: new_pool.append(self.pool[i])

        self.pool = new_pool

        return actions





# ═══════════════════════════════════════════════════════════════

# GALLERY MANAGER

# ═══════════════════════════════════════════════════════════════

class GalleryManager:

    def __init__(self, max_images):

        self.classes = {}

        self.gids = []

        self.max_images = max_images

        self.current_max_id = 0



    def num_classes(self): return len(self.gids)



    def mint_new_class(self):

        # Ensure we never mint a GID that already exists in loaded data

        self.current_max_id += 1

        while self.current_max_id in self.classes:

            self.current_max_id += 1

        gid = self.current_max_id

        self._init_class(gid)

        return gid



    def mint_class_at(self, gid):

        if gid in self.classes: return

        self.classes[gid] = {'embs': [], 'ratios': [], 'filepaths': [], 'scores': []}

        self.gids.append(gid)

        self.current_max_id = max(self.current_max_id, gid)

        (DB_IMAGES_DIR / f"global_id_{gid:03d}").mkdir(exist_ok=True)



    def _init_class(self, gid):

        self.classes[gid] = {'embs': [], 'ratios': [], 'filepaths': [], 'scores': []}

        self.gids.append(gid)

        (DB_IMAGES_DIR / f"global_id_{gid:03d}").mkdir(exist_ok=True)



    def get_weighted_centroids(self):

        centroids, active_gids = [], []

        for gid in self.gids:

            data = self.classes[gid]

            if not data['embs']: continue

            embs = torch.stack(data['embs'])

            ratios = torch.tensor(data['ratios'], device=embs.device).unsqueeze(1)

            w = torch.sum(embs * ratios, dim=0)

            centroids.append(F.normalize(w, p=2, dim=0))

            active_gids.append(gid)

        if not centroids: return None, []

        return torch.stack(centroids), active_gids



    def classify_live_track(self, emb, tau, min_cos=0.78, exclude_gids=None):

        if self.num_classes() == 0: return None, 0.0, 0.0

        centroids, active_gids = self.get_weighted_centroids()

        if centroids is None: return None, 0.0, 0.0

        sims = torch.mv(centroids, emb)

        if exclude_gids:

            for i, g in enumerate(active_gids):

                if g in exclude_gids: sims[i] = -float('inf')

        probs = F.softmax(sims * 15.0, dim=0)

        sp, si = torch.sort(probs, descending=True)

        mp = sp[0].item(); bs = sims[si[0]].item(); bg = active_gids[si[0].item()]

        if bs == -float('inf'): return None, 0.0, 0.0

        if bs < min_cos: return bg, 0.0, mp

        ratio = mp / (sp[1].item() + 1e-9) if len(sp) > 1 else float('inf')

        return bg, ratio, mp



    def add_observation(self, gid, item):

        data = self.classes[gid]

        data['embs'].append(item['emb']); data['ratios'].append(item['ratio']); data['scores'].append(item['score'])

        fn = f"frame_{item['frame']:06d}_trk_{item['track_id']:03d}.jpg"

        fp = DB_IMAGES_DIR / f"global_id_{gid:03d}" / fn

        cv2.imwrite(str(fp), item['img']); data['filepaths'].append(fp)

        self._optimize_budget(gid)



    def _optimize_budget(self, gid):

        data = self.classes[gid]

        if len(data['embs']) <= self.max_images: return

        embs = torch.stack(data['embs'])

        sim_m = torch.mm(embs, embs.t()); sim_m.fill_diagonal_(-1.0)

        redundancy, _ = torch.max(sim_m, dim=1)

        centroids, _ = self.get_weighted_centroids()

        if centroids is not None and centroids.shape[0] > 1:

            s2a = torch.mm(embs, centroids.t())

            p = F.softmax(s2a * 15.0, dim=1)

            ent = -torch.sum(p * torch.log(p + 1e-9), dim=1)

            unc = ent / torch.log(torch.tensor(float(centroids.shape[0]), device=embs.device))

        else:

            unc = torch.zeros(embs.shape[0], device=embs.device)

        cost = 0.5 * unc.cpu().numpy() + 0.5 * redundancy.cpu().numpy()

        wi = int(np.argmax(cost))

        ml = min(len(data[k]) for k in ['embs','ratios','scores','filepaths'])

        if wi >= ml: wi = ml - 1

        fp = Path(data['filepaths'][wi])

        if fp.exists() and 'working' in str(fp): fp.unlink()

        for k in ['embs','ratios','scores','filepaths']:

            if wi < len(data[k]): data[k].pop(wi)





# ═══════════════════════════════════════════════════════════════

# IDENTITY MERGE (face-driven retroactive correction)

# ═══════════════════════════════════════════════════════════════

def execute_identity_merge(target_gid, old_gid, gallery_ref, local_map_ref, sighting_ref, persons_ref=None):

    """Merge old_gid INTO target_gid across gallery and tracker (in-memory only)."""

    if old_gid not in gallery_ref.classes or target_gid == old_gid: return



    if target_gid not in gallery_ref.classes:

        gallery_ref.mint_class_at(target_gid)



    o_data = gallery_ref.classes.pop(old_gid)

    gallery_ref.gids.remove(old_gid)

    t_data = gallery_ref.classes[target_gid]

    target_dir = DB_IMAGES_DIR / f"global_id_{target_gid:03d}"



    for j in range(len(o_data['embs'])):

        fp = Path(o_data['filepaths'][j]) if j < len(o_data['filepaths']) else None

        if fp and fp.exists():

            new_fp = target_dir / fp.name

            if 'input' in str(fp): shutil.copy(str(fp), str(new_fp))

            else: shutil.move(str(fp), str(new_fp))

            t_data['embs'].append(o_data['embs'][j]); t_data['ratios'].append(o_data['ratios'][j])

            t_data['scores'].append(o_data['scores'][j]); t_data['filepaths'].append(new_fp)



    # Update local map

    for tid, gid in list(local_map_ref.items()):

        if gid == old_gid: local_map_ref[tid] = target_gid

    # Update sighting tracker

    for tid, s in sighting_ref.active_sightings.items():

        if s["global_id"] == old_gid: s["global_id"] = target_gid





print("✅ Core classes & helpers defined.")
# ═══════════════════════════════════════════════════════════════

# GALLERY PERSISTENCE VIA PINECONE (replaces pkl files)

# ═══════════════════════════════════════════════════════════════

GALLERY_NAMESPACE = "gallery_buffer"



def save_gallery_to_pinecone(gallery_ref):

    """Save all gallery embeddings to Pinecone namespace."""

    try: pinecone_body_index.delete(delete_all=True, namespace=GALLERY_NAMESPACE)

    except: pass

    batch = []

    for gid in gallery_ref.gids:

        d = gallery_ref.classes[gid]

        for j, emb in enumerate(d['embs']):

            batch.append({

                "id": f"gal_g{gid:03d}_{j:03d}",

                "values": emb.cpu().numpy().tolist(),

                "metadata": {

                    "global_id": int(gid), "ratio": float(d['ratios'][j]),

                    "det_score": float(d['scores'][j]),

                    "filepath": str(d['filepaths'][j]) if j < len(d['filepaths']) else ""

                }

            })

            if len(batch) >= 100:

                pinecone_body_index.upsert(vectors=batch, namespace=GALLERY_NAMESPACE)

                batch = []

    if batch:

        pinecone_body_index.upsert(vectors=batch, namespace=GALLERY_NAMESPACE)

    return len(gallery_ref.gids)





def load_gallery_from_pinecone(gallery_ref, max_gid_search=500, max_embs_per_gid=50):

    """Load gallery from Pinecone gallery_buffer namespace into a GalleryManager.



    Uses deterministic ID-based fetching (avoids broken list() API across SDK versions).

    Vector IDs follow the pattern: gal_g{gid:03d}_{j:03d}

    We probe all candidate IDs in batches and keep whatever Pinecone returns.

    """

    # --- First check if namespace has anything at all ---

    try:

        stats = pinecone_body_index.describe_index_stats()

        ns_stats = stats.namespaces if hasattr(stats, 'namespaces') else (stats.get('namespaces', {}) if isinstance(stats, dict) else {})

        ns_info = ns_stats.get(GALLERY_NAMESPACE, None)

        if ns_info is None:

            print("[Gallery Load] Namespace 'gallery_buffer' not found in index stats.")

            return 0

        vector_count = ns_info.vector_count if hasattr(ns_info, 'vector_count') else ns_info.get('vector_count', 0)

        if vector_count == 0:

            print("[Gallery Load] Namespace is empty — no gallery to load.")

            return 0

        print(f"[Gallery Load] Namespace has {vector_count} vectors. Fetching...")

    except Exception as e:

        print(f"[Gallery Load] describe_index_stats failed: {e}. Proceeding anyway...")

        vector_count = max_gid_search * max_embs_per_gid  # fallback: try all



    # --- Build all candidate IDs and fetch in batches ---

    # IDs are: gal_g001_000, gal_g001_001, ..., gal_g500_049

    # We generate all candidates then fetch — Pinecone simply omits missing ones.

    n_gids_to_probe = min(max_gid_search, max(500, vector_count // 10 + 100))

    candidate_ids = [

        f"gal_g{gid:03d}_{j:03d}"

        for gid in range(1, n_gids_to_probe + 1)

        for j in range(max_embs_per_gid)

    ]



    grouped = {}

    total_fetched = 0

    for i in range(0, len(candidate_ids), 100):

        chunk = candidate_ids[i:i+100]

        try:

            fetched = pinecone_body_index.fetch(ids=chunk, namespace=GALLERY_NAMESPACE)

            vecs = fetched.vectors if hasattr(fetched, 'vectors') else (fetched.get('vectors', {}) if isinstance(fetched, dict) else {})

            for vid, vec in vecs.items():

                meta = vec.metadata if hasattr(vec, 'metadata') else (vec.get('metadata', {}) if isinstance(vec, dict) else {})

                vals = vec.values if hasattr(vec, 'values') else (vec.get('values', []) if isinstance(vec, dict) else [])

                if not vals:

                    continue

                gid = int(meta.get("global_id", -1))

                if gid < 1:

                    continue

                if gid not in grouped:

                    grouped[gid] = {"embs": [], "ratios": [], "scores": [], "filepaths": []}

                grouped[gid]["embs"].append(torch.tensor(vals, dtype=torch.float32).to(REID_DEVICE))

                grouped[gid]["ratios"].append(float(meta.get("ratio", 0.5)))

                grouped[gid]["scores"].append(float(meta.get("det_score", 0.5)))

                grouped[gid]["filepaths"].append(meta.get("filepath", ""))

                total_fetched += 1

        except Exception as e:

            print(f"[Gallery Load] fetch error at chunk {i}: {e}")

            continue

        # Early exit if we've found all vectors

        if total_fetched >= vector_count:

            break



    if not grouped:

        print("[Gallery Load] No vectors successfully fetched.")

        return 0



    # --- Populate gallery_ref ---

    max_id = 0

    for gid, data in grouped.items():

        if not data["embs"]:

            continue

        gallery_ref.gids.append(gid)

        gallery_ref.classes[gid] = {

            "embs": data["embs"],

            "ratios": data["ratios"],

            "scores": data["scores"],

            "filepaths": data["filepaths"],

        }

        max_id = max(max_id, gid)

        (DB_IMAGES_DIR / f"global_id_{gid:03d}").mkdir(parents=True, exist_ok=True)



    # CRITICAL: current_max_id must reflect the highest GID so mint_new_class()

    # continues numbering from above existing identities, never from 1.

    gallery_ref.current_max_id = max_id

    print(f"[Gallery Load] ✅ Loaded {len(grouped)} identities ({total_fetched} vectors). Max GID = {max_id}. Next new GID = {max_id + 1}.")

    return len(grouped)





def save_results_to_cloud(gallery_ref, sighting_tracker_ref, is_face_camera_val):

    """Upload sightings + faces + gallery to cloud. Called AFTER pipeline finishes."""

    msgs = []



    # Ensure person documents exist for all GIDs

    all_gids = set()

    for s in sighting_tracker_ref.completed_sightings:

        if s.get('global_id') is not None: all_gids.add(s['global_id'])

    for f in sighting_tracker_ref.completed_faces:

        if f.get('global_id') is not None: all_gids.add(f['global_id'])

    for gid in all_gids:

        first_ts = min((s.get('entry_ts', 0) for s in sighting_tracker_ref.completed_sightings if s.get('global_id') == gid), default=0)

        create_person_document(gid, first_ts, persons_col)

    msgs.append(f"✅ {len(all_gids)} person documents ensured.")



    # Body sightings

    for s in sighting_tracker_ref.completed_sightings:

        gid = s.get('global_id')

        if gid is None: continue

        tid = s['track_id']

        folder = f"/supermarket_reid/global_id_{gid:03d}/"

        img = s.get('best_crop_img')

        if img is not None:

            try:

                _, buf = cv2.imencode('.jpg', img)

                fn = f"body_g{gid:03d}_t{tid}_{int(time.time())}.jpg"

                resp = imagekit_client.files.upload(file=buf.tobytes(), file_name=fn, folder=folder)

                s['best_crop_url'] = resp.url if hasattr(resp, 'url') else ''

            except: s['best_crop_url'] = ""

            del s['best_crop_img']

        gid_val = s.pop('global_id', gid)

        persons_col.update_one({"global_id": gid_val}, {

            "$push": {"sightings": s}, "$set": {"last_seen_ts": float(s.get("exit_ts", 0))}

        })

    msgs.append(f"✅ {len(sighting_tracker_ref.completed_sightings)} sightings uploaded.")



    # Faces

    if is_face_camera_val and sighting_tracker_ref.completed_faces:

        for f_data in sighting_tracker_ref.completed_faces:

            gid = f_data.get('global_id')

            if gid is None: continue

            tid = f_data.get('track_id', 0)

            folder = f"/supermarket_reid/global_id_{gid:03d}/"

            face_img, face_emb = f_data.get('best_face_img'), f_data.get('best_face_emb')

            face_conf = f_data.get('best_face_conf', 0.0)

            if face_img is not None and face_img.size > 0:

                try:

                    _, fb = cv2.imencode('.jpg', face_img)

                    ffn = f"face_g{gid:03d}_t{tid}_{int(time.time())}.jpg"

                    fr = imagekit_client.files.upload(file=fb.tobytes(), file_name=ffn, folder=folder)

                    furl = fr.url if hasattr(fr, 'url') else ''

                    if face_emb is not None:

                        pinecone_face_index.upsert(vectors=[{"id": f"face_g{gid:03d}_{int(time.time())}",

                            "values": face_emb.tolist(),

                            "metadata": {"global_id": gid, "face_url": furl, "camera": CAMERA_ID}}])

                    persons_col.update_one({"global_id": gid}, {"$push": {"faces": {

                        "url": furl, "confidence": face_conf,

                        "timestamp": float(f_data.get("timestamp", 0)),

                        "camera_id": f_data.get("camera_id", CAMERA_ID)}}})

                except Exception as e:

                    msgs.append(f"⚠️ Face error GID {gid}: {e}")

        msgs.append(f"✅ {len(sighting_tracker_ref.completed_faces)} faces uploaded.")



    # Gallery to Pinecone

    n = save_gallery_to_pinecone(gallery_ref)

    msgs.append(f"✅ Gallery saved to Pinecone ({n} identities).")



    # Prototypes

    try:

        pinecone_body_index.delete(delete_all=True, namespace="")

    except: pass

    batch = []

    for gid in gallery_ref.gids:

        d = gallery_ref.classes[gid]

        if not d['embs']: continue

        embs = torch.stack(d['embs'])

        qs = torch.tensor(d['scores'], device=REID_DEVICE) * torch.tensor(d['ratios'], device=REID_DEVICE)

        indices = alpha_fps(embs, qs, K=3, alpha=0.5)

        for pi, idx in enumerate(indices):

            batch.append({"id": f"global_{gid}_proto_{pi+1}",

                          "values": embs[idx].cpu().numpy().tolist(),

                          "metadata": {"global_id": gid, "prototype_idx": pi+1}})

    if batch:

        pinecone_body_index.upsert(vectors=batch, namespace="")

    msgs.append(f"✅ {len(batch)} search prototypes built.")



    return "\n".join(msgs)





def reset_all_databases():

    """Reset MongoDB + Pinecone gallery namespace + local files."""

    msgs = []

    persons_col.delete_many({})

    msgs.append("✅ MongoDB cleared.")

    try:

        pinecone_body_index.delete(delete_all=True, namespace=GALLERY_NAMESPACE)

        msgs.append("✅ Pinecone gallery namespace cleared.")

    except Exception as e:

        msgs.append(f"⚠️ Pinecone gallery clear error: {e}")

    try:

        pinecone_body_index.delete(delete_all=True, namespace="")

        msgs.append("✅ Pinecone prototype namespace cleared.")

    except Exception as e:

        msgs.append(f"⚠️ Pinecone prototype clear error: {e}")

    try:

        pinecone_face_index.delete(delete_all=True, namespace="")

        msgs.append("✅ Pinecone face index cleared.")

    except Exception as e:

        msgs.append(f"⚠️ Pinecone face index clear error: {e}")

    if OUTPUT_DIR.exists():

        shutil.rmtree(OUTPUT_DIR)

        msgs.append("✅ Local output files deleted.")

    DB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    msgs.append("\n⚠️ **Manual action needed:** Delete the ImageKit `/supermarket_reid/` folder if you want a full reset.")

    return "\n".join(msgs)





# ═══════════════════════════════════════════════════════════════

# VIDEO PROCESSING PIPELINE (Module 4 logic, wrapped as a function)

# ═══════════════════════════════════════════════════════════════



def run_video_pipeline(

    video_path, is_face_camera=True,

    use_existing_data=True,

    confidence=0.75, tau=2.0, min_cos_sim=0.78, max_images=50,

    pose_conf=0.3, min_visible_ratio=0.5,

    face_conf=0.7, face_match_th=0.55,

    udm_eps=0.15, udm_min_size=5,

    progress=gr.Progress(track_tqdm=True)

):

    """Run the full Module 4 pipeline on a video."""

    logs = []

    def log(msg):

        logs.append(msg)



    import time as _time

    _wall_start = _time.time()

    VIDEO_START_TIME = datetime.now(timezone.utc)



    DB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)



    # Init core objects

    gallery = GalleryManager(max_images=max_images)

    sighting_tracker = SightingTracker(fps=30, video_start_time=VIDEO_START_TIME)

    udm = UnknownDataManager(eps=udm_eps, min_samples=udm_min_size)

    tracker = tracker_template()

    local_to_global_map = {}



    # Hydrate from Pinecone gallery namespace

    n_loaded = 0

    loaded_gids = set()

    if use_existing_data:

        n_loaded = load_gallery_from_pinecone(gallery)

        loaded_gids = set(gallery.gids)  # capture GIDs before any new minting

        if n_loaded > 0:

            log(f"✅ Hydrated {n_loaded} identities from Pinecone. Max known GID = {gallery.current_max_id}. Next new ID = {gallery.current_max_id + 1}.")

        else:

            log("⚠️ No existing gallery found in Pinecone — starting fresh from GID 1.")



    # Open video

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    sighting_tracker.fps = fps

    log(f"📹 Video: {total_frames} frames @ {fps:.1f} FPS")



    frame_count = 0

    sample_frames = []



    # ── VRAM tracking (mem_get_info = total - free = ALL usage incl. models) ──

    _vram_samples = []

    def _vram_mb():

        if torch.cuda.is_available():

            total_used = 0

            for i in range(torch.cuda.device_count()):

                free, total = torch.cuda.mem_get_info(i)

                total_used += (total - free)

            return total_used / 1024**2

        return 0.0

    vram_idle_mb = _vram_mb()



    for _ in progress.tqdm(range(total_frames), desc="Processing video"):

        ret, frame = cap.read()

        if not ret: break

        frame_count += 1



        # Save a few sample frames for display

        if frame_count % max(1, total_frames // 6) == 0:

            sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



        # 1. Detection

        results = detector(frame, classes=[0], verbose=False)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0: continue



        faces_in_frame = face_app.get(frame) if is_face_camera else []

        dets = torch.cat([boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1)], dim=1).cpu().numpy()

        tracks = tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0: continue



        # 2. Filter

        valid = []

        for t in tracks:

            tid = int(t[4]); conf = float(t[5]) if len(t) > 5 else 1.0

            if conf >= confidence:

                x1, y1, x2, y2 = max(0, int(t[0])), max(0, int(t[1])), int(t[2]), int(t[3])

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0: valid.append({'track_id': tid, 'conf': conf, 'box': [x1,y1,x2,y2], 'crop': crop})



        if not valid: continue



        # 3. Pose gate

        pose_results = pose_model([d['crop'] for d in valid], verbose=False)

        pristine = []

        for i, res in enumerate(pose_results):

            vc = sum(1 for x,y,c in res.keypoints.data[0].cpu().numpy() if c > pose_conf) if res.keypoints is not None and len(res.keypoints.data) > 0 else 0

            ratio = vc / 17.0

            if ratio >= min_visible_ratio:

                pristine.append({**valid[i], 'ratio': ratio, 'score': valid[i]['conf']})



        if not pristine: continue



        # 4. Embed

        embeddings = extract_reid_features([d['crop'] for d in pristine])

        claimed = set()

        unresolved = []



        # 5. Identity resolution

        for i, d in enumerate(pristine):

            item = {'emb': embeddings[i], 'ratio': d['ratio'], 'score': d['score'],

                    'img': d['crop'], 'track_id': d['track_id'], 'frame': frame_count}

            tid = d['track_id']

            gid = local_to_global_map.get(tid)

            sighting_tracker.update_track(tid, frame_count, item['score'], item['ratio'], item['img'], gid, d['box'])



            if is_face_camera:

                mf = get_spatially_matched_face(d['box'], faces_in_frame)

                if mf and mf.det_score > face_conf:

                    fx1,fy1,fx2,fy2 = map(int, mf.bbox)

                    face_img = frame[max(0,fy1):fy2, max(0,fx1):fx2]

                    sighting_tracker.update_face(tid, face_img, mf.normed_embedding, mf.det_score)

                    ftg = search_face_pinecone(mf.normed_embedding, threshold=face_match_th)

                    if ftg:

                        if gid and gid != ftg:

                            execute_identity_merge(ftg, gid, gallery, local_to_global_map, sighting_tracker)

                            gid = ftg

                        elif not gid:

                            item['forced_gid'] = ftg



            gid = local_to_global_map.get(tid)

            if gid is not None:

                if gid not in claimed: claimed.add(gid); gallery.add_observation(gid, item)

                else: del local_to_global_map[tid]; unresolved.append(item)

            else: unresolved.append(item)



        # 6. Unknowns

        for item in unresolved:

            tid = item['track_id']

            forced = item.get('forced_gid')

            if forced is not None:

                if forced not in gallery.classes: gallery.mint_class_at(forced)

                local_to_global_map[tid] = forced; claimed.add(forced)

                gallery.add_observation(forced, item)

                sighting_tracker.assign_global_id(tid, forced)

            elif gallery.num_classes() == 0:

                ng = gallery.mint_new_class()

                local_to_global_map[tid] = ng; claimed.add(ng)

                gallery.add_observation(ng, item)

                sighting_tracker.assign_global_id(tid, ng)

            else:

                bg, sr, mp = gallery.classify_live_track(item['emb'], tau=tau, min_cos=min_cos_sim, exclude_gids=claimed)

                if bg is not None and sr >= tau:

                    local_to_global_map[tid] = bg; claimed.add(bg)

                    gallery.add_observation(bg, item)

                    sighting_tracker.assign_global_id(tid, bg)

                else:

                    udm.add(item)



        if frame_count % 30 == 0:

            _vram_samples.append(_vram_mb())

        if frame_count % 30 == 0 and len(udm.pool) >= udm_min_size:

            actions = udm.run_dbscan(gallery)

            for act, gid, cnt, tids in actions:

                pass  # MongoDB writes deferred to save step

                for tid in tids:

                    sighting_tracker.assign_global_id(tid, gid)

                    local_to_global_map[tid] = gid



    # Close tracks

    for tid in list(sighting_tracker.active_sightings.keys()):

        sighting_tracker.close_track(tid)

    cap.release()



    _wall_end = _time.time()

    wall_time = _wall_end - _wall_start

    processing_fps = frame_count / wall_time if wall_time > 0 else 0



    log(f"✅ Done! {gallery.num_classes()} identities found in {frame_count} frames.")



    vram_min  = min(_vram_samples) if _vram_samples else vram_idle_mb

    vram_max  = max(_vram_samples) if _vram_samples else vram_idle_mb

    vram_avg  = sum(_vram_samples) / len(_vram_samples) if _vram_samples else vram_idle_mb

    log(f"🖥️  VRAM — idle: {vram_idle_mb:.0f} MB | min: {vram_min:.0f} MB | avg: {vram_avg:.0f} MB | peak: {vram_max:.0f} MB")



    video_info = {

        "total_frames": total_frames,

        "video_fps": fps,

        "frames_processed": frame_count,

        "wall_time_sec": wall_time,

        "processing_fps": processing_fps,

        "n_loaded_ids": n_loaded if use_existing_data else 0,

        "loaded_gids": loaded_gids,

        "vram_idle_mb": vram_idle_mb,

        "vram_min_mb": vram_min,

        "vram_max_mb": vram_max,

        "vram_avg_mb": vram_avg,

    }



    summary = "\n".join(logs)

    return summary, gallery, sighting_tracker, is_face_camera, video_info





# ═══════════════════════════════════════════════════════════════

# SEARCH FUNCTIONS (Module 5 logic)

# ═══════════════════════════════════════════════════════════════



def alpha_fps(embeddings, quality_scores, K=3, alpha=0.5):

    N = embeddings.shape[0]

    if N <= K: return list(range(N))

    selected = []

    first = torch.argmax(quality_scores).item()

    selected.append(first)

    distances = 1.0 - torch.mm(embeddings, embeddings[first].unsqueeze(1)).squeeze()

    for _ in range(1, K):

        combined = (alpha * quality_scores) + ((1.0 - alpha) * distances)

        combined[selected] = -1.0

        nxt = torch.argmax(combined).item()

        selected.append(nxt)

        new_dist = 1.0 - torch.mm(embeddings, embeddings[nxt].unsqueeze(1)).squeeze()

        distances = torch.min(distances, new_dist)

    return selected





def search_body(query_img, th_emb=0.75, tau=1.5):

    """Search by body crop → Pinecone supermarket-reid."""

    emb = embed_body_image(query_img)

    vec = emb.cpu().numpy().tolist()[0]

    resp = pinecone_body_index.query(vector=vec, top_k=10, include_metadata=True)

    if not resp.matches or resp.matches[0].score < th_emb:

        return {"status": "UNKNOWN", "reason": f"Below threshold ({th_emb})"}

    ids = {}

    for m in resp.matches:

        g = m.metadata["global_id"]

        if g not in ids: ids[g] = m.score

    sorted_ids = sorted(ids.items(), key=lambda x: x[1], reverse=True)

    bg, bs = sorted_ids[0]

    if len(sorted_ids) > 1:

        p = F.softmax(torch.tensor([sorted_ids[0][1], sorted_ids[1][1]]) * 15.0, dim=0)

        r = p[0].item() / (p[1].item() + 1e-9)

        if r < tau:

            return {"status": "AMBIGUOUS", "reason": f"Confused between GID {bg} and {sorted_ids[1][0]}. Ratio {r:.2f}", "best_guess_gid": bg}

    doc = persons_col.find_one({"global_id": int(bg)}, {"_id": 0})

    return {"status": "MATCH", "global_id": int(bg), "score": bs, "mongodb_data": doc or {}}





def search_face_query(query_img, is_crop=False, th_face=0.55):

    """Search by face → Pinecone supermarket-faces."""

    if is_crop:

        emb, ds = embed_face_crop(query_img)

    else:

        emb, ds = embed_face_image(query_img)

    if emb is None:

        return {"status": "NO_FACE", "reason": "No face detected."}

    resp = pinecone_face_index.query(vector=emb.tolist(), top_k=5, include_metadata=True)

    if not resp.matches or resp.matches[0].score < th_face:

        return {"status": "UNKNOWN", "reason": f"No face match above {th_face}"}

    best = resp.matches[0]

    bg = int(best.metadata["global_id"])

    doc = persons_col.find_one({"global_id": bg}, {"_id": 0})

    return {"status": "MATCH", "global_id": bg, "score": best.score,

            "match_face_url": best.metadata.get("face_url", ""),

            "mongodb_data": doc or {}}





print("✅ Pipeline & search functions defined.")
# ═══════════════════════════════════════════════════════════════

# GRADIO INTERFACE

# ═══════════════════════════════════════════════════════════════



THEME_CSS = """

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');



.gradio-container {

    background: #0a0a0f !important;

    font-family: 'Inter', sans-serif !important;

    color: #e2e8f0 !important;

}



/* ── All text ── */

.gradio-container, .gradio-container p, .gradio-container span,

.gradio-container label, .gradio-container td, .gradio-container th,

.gradio-container li, .gradio-container h1, .gradio-container h2,

.gradio-container h3, .gradio-container h4, .gradio-container .prose,

.gradio-container .prose *, .gradio-container fieldset label span {

    color: #e2e8f0 !important;

}

.gradio-container h1 { color: #ffffff !important; }



/* ── Panels / blocks ── */

.gradio-container .block {

    background: #12121a !important;

    border: 1px solid #1e293b !important;

    border-radius: 12px !important;

    box-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;

}



/* ── Inputs ── */

.gradio-container input[type="text"], .gradio-container textarea,

.gradio-container .scroll-hide, .gradio-container select,

.gradio-container .gr-dropdown, .gradio-container .wrap {

    background: #1a1a2e !important; color: #e2e8f0 !important;

    border: 1px solid #2d3748 !important; border-radius: 8px !important;

}

.gradio-container input[type="text"]::placeholder { color: #64748b !important; }



/* ── Sliders ── */

.gradio-container input[type="range"] { accent-color: #6366f1 !important; }

.gradio-container input[type="checkbox"] { accent-color: #6366f1 !important; }



/* ── Tabs ── */

.gradio-container .tab-nav {

    background: #0a0a0f !important;

    border-bottom: 1px solid #1e293b !important;

}

.gradio-container .tab-nav button {

    font-size: 15px !important; font-weight: 600 !important;

    color: #64748b !important; background: transparent !important;

    border: none !important; border-bottom: 2px solid transparent !important;

    transition: all 0.2s ease !important;

}

.gradio-container .tab-nav button:hover { color: #a5b4fc !important; }

.gradio-container .tab-nav button.selected {

    color: #a5b4fc !important;

    border-bottom: 2px solid #6366f1 !important;

}



/* ── Primary button ── */

.gradio-container button.primary {

    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;

    color: #ffffff !important; border: none !important;

    border-radius: 10px !important; font-weight: 700 !important;

    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;

    transition: all 0.2s ease !important;

}

.gradio-container button.primary:hover {

    box-shadow: 0 0 30px rgba(99, 102, 241, 0.5) !important;

    transform: translateY(-1px) !important;

}



/* ── Secondary button ── */

.gradio-container button.secondary {

    background: #1a1a2e !important; color: #cbd5e1 !important;

    border: 1px solid #334155 !important; border-radius: 10px !important;

    transition: all 0.2s ease !important;

}

.gradio-container button.secondary:hover {

    background: #1e293b !important; border-color: #6366f1 !important;

}



/* ── Accordion ── */

.gradio-container .accordion .label-wrap {

    background: #12121a !important; border-radius: 10px !important;

}

.gradio-container .accordion .label-wrap span { color: #cbd5e1 !important; font-weight: 600 !important; }

.gradio-container .accordion .label-wrap svg path { fill: #94a3b8 !important; }



/* ── Gallery ── */

.gradio-container .gallery, .gradio-container .gallery-item {

    background: #12121a !important; border-color: #1e293b !important;

}



/* ── Tables in markdown ── */

.gradio-container .prose table { border-collapse: collapse !important; width: 100% !important; }

.gradio-container .prose table th {

    background: #1a1a2e !important; color: #a5b4fc !important;

    border: 1px solid #2d3748 !important; padding: 8px 12px !important;

    font-weight: 600 !important;

}

.gradio-container .prose table td {

    background: #12121a !important; color: #cbd5e1 !important;

    border: 1px solid #1e293b !important; padding: 8px 12px !important;

}

.gradio-container .prose table tr:hover td { background: #1a1a2e !important; }



/* ── Code blocks ── */

.gradio-container .prose pre, .gradio-container .prose code {

    background: #1a1a2e !important; color: #a5b4fc !important;

    border: 1px solid #2d3748 !important; border-radius: 8px !important;

}



/* ── Horizontal rules ── */

.gradio-container .prose hr { border-color: #1e293b !important; }



/* ── Strong/bold ── */

.gradio-container .prose strong { color: #ffffff !important; }



/* ── Custom buttons ── */

.danger-btn {

    background: linear-gradient(135deg, #dc2626, #ef4444) !important;

    color: #ffffff !important; border: none !important; border-radius: 10px !important;

    font-weight: 700 !important; box-shadow: 0 0 15px rgba(220,38,38,0.3) !important;

}

.danger-btn:hover { box-shadow: 0 0 25px rgba(220,38,38,0.5) !important; }

.save-btn {

    background: linear-gradient(135deg, #059669, #10b981) !important;

    color: #ffffff !important; border: none !important; border-radius: 10px !important;

    font-weight: 700 !important; box-shadow: 0 0 15px rgba(16,185,129,0.3) !important;

}

.save-btn:hover { box-shadow: 0 0 25px rgba(16,185,129,0.5) !important; }



/* ── File upload ── */

.gradio-container .upload-text { color: #94a3b8 !important; }



/* ── Dropdown ── */

.gradio-container .options { background: #1a1a2e !important; border: 1px solid #2d3748 !important; }

.gradio-container .options .item { color: #e2e8f0 !important; }

.gradio-container .options .item:hover { background: #2d3748 !important; }

.gradio-container .options .item.selected { background: #4f46e5 !important; }



footer { display: none !important; }







/* ── Fix Tiny Scrollable Progress Bars ── */

.gradio-container .progress-text, 

.gradio-container .progress-wrap, 

.gradio-container .progress-level, 

.gradio-container .progress-container,

.gradio-container .progress {

    max-height: none !important;

    min-height: 80px !important;

    overflow: visible !important;

}

.gradio-container .html, .gradio-container .markdown {

    overflow: visible !important;

    max-height: none !important;

}





/* ── Remove outline boxes inside accordions ── */

.gradio-container .gradio-accordion .block, 

.gradio-container .gradio-accordion .form {

    border: none !important; 

    box-shadow: none !important; 

    background: transparent !important;

}



/* ── Progress bar height fix ── */

.gradio-container .min-height-block {

    min-height: 250px !important;

}



/* ── Clean UI for sliders and parameters ── */

.gradio-container .gradio-accordion .block,

.gradio-container .gradio-accordion .form,

.gradio-container .gradio-accordion fieldset,

.gradio-container .gradio-accordion .gr-form,

.gradio-container .gradio-accordion .gr-box {

    border: none !important; 

    box-shadow: none !important; 

    background: transparent !important;

}

.gradio-container .form, .gradio-container .block.padded:not(.gradio-accordion) {

    border: none !important; box-shadow: none !important;

    background: transparent !important;

}

.gradio-container .gradio-slider {

    background: transparent !important; border: none !important;

}

.gradio-container .gradio-slider > div {

    border: none !important; background: transparent !important;

}

.gradio-container .gradio-accordion {

    background: #12121a !important; border: 1px solid #1e293b !important;

    border-radius: 12px !important;

}

.gradio-container .gradio-accordion .form {

    background: transparent !important; border: none !important; box-shadow: none !important;

}

.gradio-container .gradio-slider input[type="number"] {

    background: #1a1a2e !important; color: #e2e8f0 !important;

    border: 1px solid #2d3748 !important; border-radius: 6px !important;

}

.gradio-container .gradio-slider .wrap { background: transparent !important; border: none !important; }



/* ── Gradio Group/Row borders ── */

.gradio-container .gr-group, .gradio-container .gr-panel {

    border: none !important; box-shadow: none !important;

}



/* ── File upload area ── */

.gradio-container .file-preview { background: #12121a !important; }



/* ── Markdown section headers inside accordion ── */

.gradio-container .accordion .prose p strong {

    color: #a5b4fc !important; font-size: 14px !important;

}

"""





# ═══════════════════════════════════════════════════════════════

# HELPER FUNCTIONS

# ═══════════════════════════════════════════════════════════════



def get_person_details(gid):

    doc = persons_col.find_one({"global_id": int(gid)}, {"_id": 0})

    if not doc:

        return f"<p style='color:#ef4444;padding:16px;'>No record for GID {gid}.</p>", [], []



    sightings = doc.get("sightings", [])

    faces     = doc.get("faces", [])

    first     = doc.get("first_seen_ts", 0)

    last      = doc.get("last_seen_ts", 0)

    fs = datetime.fromtimestamp(first).strftime('%Y-%m-%d %H:%M:%S') if first else "N/A"

    ls = datetime.fromtimestamp(last).strftime('%Y-%m-%d %H:%M:%S') if last else "N/A"

    lbl = "color:#64748b;"

    val_s = "color:#e2e8f0;font-weight:600;"

    rows = ""

    for i, s in enumerate(sightings):

        entry = datetime.fromtimestamp(s.get("entry_ts", 0)).strftime('%H:%M:%S') if s.get("entry_ts") else "?"

        dur  = f"{s.get('duration_sec', 0):.1f}s"

        nd   = s.get("n_detections", 0)

        ac   = f"{s.get('avg_det_score', 0):.2f}"

        cam  = s.get("camera_id", "?")

        bg   = "#12121a" if i % 2 == 0 else "#0f0f18"

        rows += (f'<tr style="background:{bg};">'

                 f'<td style="padding:7px 12px;color:#a5b4fc;">{i+1}</td>'

                 f'<td style="padding:7px 12px;color:#94a3b8;">{cam}</td>'

                 f'<td style="padding:7px 12px;color:#cbd5e1;">{entry}</td>'

                 f'<td style="padding:7px 12px;color:#6ee7b7;">{dur}</td>'

                 f'<td style="padding:7px 12px;color:#cbd5e1;">{nd}</td>'

                 f'<td style="padding:7px 12px;color:#fbbf24;">{ac}</td></tr>')



    info = f"""<div style="background:#12121a;border:1px solid #1e293b;border-radius:12px;padding:20px;margin-bottom:16px;">

<h2 style="color:#a5b4fc;margin:0 0 14px 0;font-size:20px;">Person: Global ID {int(gid):03d}</h2>

<div style="display:flex;gap:28px;flex-wrap:wrap;margin-bottom:16px;font-size:13px;">

  <span style="{lbl}">First seen: <b style="{val_s}">{fs}</b></span>

  <span style="{lbl}">Last seen: <b style="{val_s}">{ls}</b></span>

  <span style="{lbl}">Total visits: <b style="{val_s}">{len(sightings)}</b></span>

  <span style="{lbl}">Faces captured: <b style="color:#10b981;font-weight:600;">{len(faces)}</b></span>

</div>

<table style="width:100%;border-collapse:collapse;font-size:13px;">

<tr style="background:#1a1a2e;">

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">#</th>

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">Camera</th>

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">Entry</th>

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">Duration</th>

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">Detections</th>

  <th style="padding:8px 12px;color:#a5b4fc;text-align:left;">Avg Conf</th>

</tr>{rows}

</table></div>"""



    body_imgs = []

    for s in sightings:

        url = s.get("best_crop_url", "")

        if url:

            img = fetch_image_from_url(url, as_rgb=True)

            if img is not None: body_imgs.append(img)



    face_imgs = []

    for f in faces:

        url = f.get("url", "")

        if url:

            img = fetch_image_from_url(url, as_rgb=True)

            if img is not None: face_imgs.append(img)



    return info, body_imgs[:12], face_imgs[:12]





def run_search(image, image_url, search_mode, is_face_crop, body_threshold, face_threshold, tau):

    if image is None and not image_url:

        return "Please upload an image or provide a URL.", None, ""

    

    if image is None and image_url:

        try:

            import urllib.request

            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})

            resp = urllib.request.urlopen(req)

            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)

            image = cv2.imdecode(arr, -1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:

            return f"Failed to load image from URL: {e}", None, ""



    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    body_result, face_result = None, None

    body_text, face_text = "", ""

    display_img = None



    if search_mode in ["Body", "Both"]:

        body_result = search_body(img_bgr, th_emb=body_threshold, tau=tau)

        if body_result["status"] == "MATCH":

            body_text = f"**Body Match: GID {body_result['global_id']}** (Score: {body_result['score']:.3f})"

            doc = body_result.get("mongodb_data", {})

            for s in doc.get("sightings", []):

                if s.get("best_crop_url"):

                    display_img = fetch_image_from_url(s["best_crop_url"], as_rgb=True); break

        else:

            body_text = f"Body: {body_result['status']} — {body_result.get('reason', '')}"



    if search_mode in ["Face", "Both"]:

        face_result = search_face_query(img_bgr, is_crop=is_face_crop, th_face=face_threshold)

        if face_result["status"] == "MATCH":

            face_text = f"**Face Match: GID {face_result['global_id']}** (Score: {face_result['score']:.3f})"

            if display_img is None:

                furl = face_result.get("match_face_url", "")

                if furl: display_img = fetch_image_from_url(furl, as_rgb=True)

        else:

            face_text = f"Face: {face_result['status']} — {face_result.get('reason', '')}"



    lines = ["## Search Results\n"]

    if body_text: lines.append(body_text)

    if face_text: lines.append(face_text)



    details = ""

    best_gid = None

    if body_result and body_result["status"] == "MATCH": best_gid = body_result["global_id"]

    if face_result and face_result["status"] == "MATCH": best_gid = face_result["global_id"]

    if best_gid:

        info, _, _ = get_person_details(best_gid)

        details = info



    return "\n\n".join(lines), display_img, details





# ═══════════════════════════════════════════════════════════════

# PIPELINE HANDLER + RESULTS RENDERING

# ═══════════════════════════════════════════════════════════════

_last_run = {"gallery": None, "sighting_tracker": None, "is_face": True, "camera_id": CAMERA_ID}





def _img_to_b64(img_bgr):

    """Convert a BGR numpy image to a base64 data URI."""

    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return f"data:image/jpeg;base64,{b64}"





def build_results_html(gallery_out, st_out, video_info, summary_text):

    """Build card-based HTML results grouped by identity, with face cards."""



    n_ids_total = gallery_out.num_classes()

    n_loaded = video_info.get('n_loaded_ids', 0)

    n_new = n_ids_total - n_loaded

    n_sightings = len(st_out.completed_sightings)

    n_faces = len(st_out.completed_faces)



    all_confs = [s.get('avg_det_score', 0) for s in st_out.completed_sightings]

    all_skels = [s.get('avg_skeleton_ratio', 0) for s in st_out.completed_sightings]

    all_durs = [s.get('duration_sec', 0) for s in st_out.completed_sightings]

    total_dets = sum(s.get('n_detections', 0) for s in st_out.completed_sightings)



    avg_conf = sum(all_confs)/len(all_confs) if all_confs else 0

    min_conf = min(all_confs) if all_confs else 0

    max_conf = max(all_confs) if all_confs else 0

    avg_skel = sum(all_skels)/len(all_skels) if all_skels else 0

    avg_dur = sum(all_durs)/len(all_durs) if all_durs else 0

    max_dur = max(all_durs) if all_durs else 0



    vfps = video_info.get('video_fps', 30)

    total_f = video_info.get('total_frames', 0)

    wall = video_info.get('wall_time_sec', 0)

    pfps = video_info.get('processing_fps', 0)



    # ── Metric card style ──

    mc  = "display:inline-block;background:#1a1a2e;border:1px solid #2d3748;border-radius:12px;padding:16px 24px;margin:6px;text-align:center;min-width:140px;"

    mv  = "font-size:28px;font-weight:700;color:#a5b4fc;margin:0;"

    ml  = "font-size:12px;color:#94a3b8;margin:4px 0 0 0;text-transform:uppercase;letter-spacing:1px;"

    mc2 = "display:inline-block;background:#1a1a2e;border:1px solid #2d3748;border-radius:12px;padding:16px 24px;margin:6px;text-align:center;min-width:140px;"



    html = '<div style="padding:8px 0;">'



    # ── Top metric cards ──

    html += '<div style="text-align:center;margin-bottom:20px;">'

    metrics = [

        (f"{n_ids_total}", "#a5b4fc", "Total Identities"),

        (f"{n_ids_total - n_loaded}", "#6ee7b7", "New This Video"),

        (f"{n_loaded}", "#94a3b8", "Pre-Loaded"),

        (f"{n_sightings}", "#a5b4fc", "Sightings"),

        (f"{total_dets}", "#a5b4fc", "Detections"),

        (f"{n_faces}", "#10b981", "Faces"),

        (f"{wall:.1f}s", "#a5b4fc", "Processing Time"),

        (f"{pfps:.1f}", "#a5b4fc", "FPS (Pipeline)"),

    ]

    for val, color, label in metrics:

        html += f'<div style="{mc}"><p style="font-size:28px;font-weight:700;color:{color};margin:0;">{val}</p><p style="{ml}">{label}</p></div>'

    html += '</div>'



    # ── Video info bar ──

    html += '<div style="background:#12121a;border:1px solid #1e293b;border-radius:10px;padding:12px 20px;margin-bottom:20px;display:flex;gap:24px;flex-wrap:wrap;justify-content:center;">'

    v_idle = video_info.get('vram_idle_mb', 0)

    v_min  = video_info.get('vram_min_mb', 0)

    v_max  = video_info.get('vram_max_mb', 0)

    v_avg  = video_info.get('vram_avg_mb', 0)

    cuda_available = v_max > 0



    for lab, val in [("Frames", total_f), ("Video FPS", f"{vfps:.0f}"), ("Duration", f"{total_f/vfps:.1f}s"),

                     ("Avg Conf", f"{avg_conf:.3f}"), ("Min/Max Conf", f"{min_conf:.3f} / {max_conf:.3f}"),

                     ("Avg Skeleton", f"{avg_skel:.3f}"), ("Avg Duration", f"{avg_dur:.1f}s"), ("Max Duration", f"{max_dur:.1f}s")]:

        html += f'<span style="color:#94a3b8;font-size:13px;"><b style="color:#e2e8f0;">{val}</b> {lab}</span>'



    if cuda_available:

        html += f'<span style="color:#94a3b8;font-size:13px;margin-top:6px;width:100%;text-align:center;">'

        html += f'🖥️ &nbsp;<b style="color:#e2e8f0;">VRAM</b> &mdash; '

        html += f'Idle: <b style="color:#6ee7b7;">{v_idle:.0f} MB</b> &nbsp;|&nbsp; '

        html += f'Min: <b style="color:#6ee7b7;">{v_min:.0f} MB</b> &nbsp;|&nbsp; '

        html += f'Avg: <b style="color:#fbbf24;">{v_avg:.0f} MB</b> &nbsp;|&nbsp; '

        html += f'Peak: <b style="color:#f87171;">{v_max:.0f} MB</b>'

        html += f'</span>'

    html += '</div>'



    # ── Group sightings by GID ──

    by_gid = {}

    for s in st_out.completed_sightings:

        gid = s.get('global_id', -1)

        if gid not in by_gid: by_gid[gid] = []

        by_gid[gid].append(s)



    # ── Group faces by GID ──

    faces_by_gid = {}

    for f in st_out.completed_faces:

        gid = f.get('global_id', -1)

        if gid not in faces_by_gid: faces_by_gid[gid] = []

        faces_by_gid[gid].append(f)



    # ── Card styles ──

    card_bg      = "background:#12121a;border:1px solid #2d3748;border-radius:12px;overflow:hidden;min-width:180px;max-width:220px;flex-shrink:0;"

    face_card_bg = "background:#0d1f17;border:1px solid #10b981;border-radius:12px;overflow:hidden;min-width:180px;max-width:220px;flex-shrink:0;"

    info_style   = "padding:12px;font-size:12px;line-height:1.7;color:#cbd5e1;"

    lbl          = "color:#64748b;"

    val_s        = "color:#e2e8f0;font-weight:600;"



    for gid in sorted(by_gid.keys()):

        sightings = by_gid[gid]

        faces     = faces_by_gid.get(gid, [])

        face_tag  = f' &nbsp;<span style="color:#10b981;font-size:14px;">&#128065; {len(faces)} face(s)</span>' if faces else ''

        # Mark as existing or new based on whether GID was loaded from Pinecone

        _loaded_gids = video_info.get('loaded_gids', set())

        if gid in _loaded_gids:

            ik_tag = ' &nbsp;<span style="background:#1e3a5f;color:#60a5fa;font-size:12px;padding:2px 8px;border-radius:6px;font-weight:600;">✓ Existing</span>'

        else:

            ik_tag = ' &nbsp;<span style="background:#1a2e1a;color:#4ade80;font-size:12px;padding:2px 8px;border-radius:6px;font-weight:600;">★ New</span>'

        html += f'<div style="margin-bottom:24px;">'

        html += (f'<h3 style="color:#a5b4fc;margin:0 0 12px 0;font-size:18px;'

                 f'border-bottom:1px solid #1e293b;padding-bottom:8px;">'

                 f'Identity GID {gid:03d} &mdash; {len(sightings)} sighting(s){face_tag}{ik_tag}</h3>')

        html += '<div style="display:flex;gap:14px;overflow-x:auto;padding-bottom:8px;">'



        # ── Face cards first (green border) ──

        for f_data in faces:

            f_tid  = f_data.get('track_id', '?')

            f_conf = f"{f_data.get('best_face_conf', 0):.3f}"

            f_img  = f_data.get('best_face_img')

            if f_img is not None:

                f_uri = _img_to_b64(f_img)

                f_img_html = f'<img src="{f_uri}" style="width:100%;height:180px;object-fit:contain;background:#0a1a10;display:block;"/>'

            else:

                f_img_html = '<div style="width:100%;height:180px;background:#0a2e1c;display:flex;align-items:center;justify-content:center;color:#10b981;">No Face</div>'

            html += f'<div style="{face_card_bg}">'

            html += f_img_html

            html += f'<div style="{info_style}">'

            html += f'<span style="color:#10b981;font-weight:700;font-size:13px;">&#128065; Face · Track {f_tid}</span><br>'

            html += f'<span style="{lbl}">Conf:</span> <span style="{val_s}">{f_conf}</span>'

            html += '</div></div>'



        # ── Body sighting cards ──

        for s in sightings:

            tid   = s.get('track_id', '?')

            entry = datetime.fromtimestamp(s.get("entry_ts", 0)).strftime('%H:%M:%S') if s.get("entry_ts") else "?"

            exit_t= datetime.fromtimestamp(s.get("exit_ts", 0)).strftime('%H:%M:%S') if s.get("exit_ts") else "?"

            dur   = f"{s.get('duration_sec', 0):.1f}s"

            nd    = s.get('n_detections', 0)

            ac    = f"{s.get('avg_det_score', 0):.3f}"

            ar    = f"{s.get('avg_skeleton_ratio', 0):.3f}"

            bq    = f"{s.get('best_crop_quality', 0):.3f}"

            frame = s.get('best_crop_frame', '?')

            bbox  = s.get('best_crop_bbox', None)



            img = s.get('best_crop_img')

            if img is not None:

                img_uri  = _img_to_b64(img)

                img_html = f'<img src="{img_uri}" style="width:100%;height:220px;object-fit:contain;background:#0a0a14;display:block;"/>'

            else:

                img_html = '<div style="width:100%;height:220px;background:#1a1a2e;display:flex;align-items:center;justify-content:center;color:#475569;">No Image</div>'



            html += f'<div style="{card_bg}">'

            html += img_html

            html += f'<div style="{info_style}">'

            html += f'<span style="color:#a5b4fc;font-weight:700;font-size:14px;">Track {tid}</span><br>'

            html += f'<span style="{lbl}">Entry:</span> <span style="{val_s}">{entry}</span> &rarr; <span style="{val_s}">{exit_t}</span><br>'

            html += f'<span style="{lbl}">Duration:</span> <span style="{val_s}">{dur}</span><br>'

            html += f'<span style="{lbl}">Detections:</span> <span style="{val_s}">{nd}</span><br>'

            html += f'<span style="{lbl}">Avg Conf:</span> <span style="{val_s}">{ac}</span><br>'

            html += f'<span style="{lbl}">Skeleton:</span> <span style="{val_s}">{ar}</span><br>'

            html += f'<span style="{lbl}">Crop Quality:</span> <span style="{val_s}">{bq}</span><br>'

            html += f'<span style="{lbl}">Frame:</span> <span style="{val_s}">{frame}</span>'

            if bbox:

                html += f'<br><span style="{lbl}">BBox:</span> <span style="color:#94a3b8;font-size:11px;">{bbox}</span>'

            html += '</div></div>'



        html += '</div></div>'



    # ── Pipeline log ──

    html += f'<details style="margin-top:20px;"><summary style="color:#94a3b8;cursor:pointer;font-size:13px;">Pipeline Log</summary>'

    html += f'<pre style="background:#1a1a2e;color:#94a3b8;padding:12px;border-radius:8px;font-size:12px;margin-top:8px;white-space:pre-wrap;">{summary_text}</pre>'

    html += '</details>'



    html += '</div>'

    return html





def build_sighting_gallery(st_out):

    """Not needed anymore — images are embedded in HTML cards."""

    return []





def run_video_handler(video_file, video_path_text, camera_id, is_face, use_existing,

                      conf, tau, min_cos, max_img, pose_c, min_vis,

                      face_c, face_m, udm_e, udm_s, progress=gr.Progress(track_tqdm=True)):

    global CAMERA_ID

    cam_val = (camera_id or "").strip()

    if cam_val:

        CAMERA_ID = cam_val



    path = video_path_text.strip() if video_path_text.strip() else None

    if video_file is not None:

        path = video_file

    if not path or not Path(str(path)).exists():

        return "<div style='background:#7f1d1d;color:#fca5a5;padding:12px 20px;border-radius:10px;'>No valid video path provided.</div>", gr.update(visible=False)



    summary, gallery_out, st_out, face_flag, video_info = run_video_pipeline(

        str(path), is_face_camera=is_face,

        use_existing_data=use_existing,

        confidence=conf, tau=tau, min_cos_sim=min_cos, max_images=int(max_img),

        pose_conf=pose_c, min_visible_ratio=min_vis,

        face_conf=face_c, face_match_th=face_m,

        udm_eps=udm_e, udm_min_size=int(udm_s),

        progress=progress

    )



    _last_run["gallery"] = gallery_out

    _last_run["sighting_tracker"] = st_out

    _last_run["is_face"] = face_flag

    _last_run["camera_id"] = CAMERA_ID



    report = build_results_html(gallery_out, st_out, video_info, summary)



    return report, gr.update(visible=True)





def do_save_results():

    g = _last_run.get("gallery")

    st = _last_run.get("sighting_tracker")

    if g is None or st is None:

        return '<div style="background:#7f1d1d;color:#fca5a5;padding:12px 20px;border-radius:10px;margin-top:8px;">No pipeline results to save. Run the pipeline first.</div>'

    global CAMERA_ID

    CAMERA_ID = _last_run.get("camera_id", CAMERA_ID)

    result = save_results_to_cloud(g, st, _last_run["is_face"])

    lines_html = result.replace("\n", "<br>")

    return f'<div style="background:#064e3b;color:#6ee7b7;padding:12px 20px;border-radius:10px;margin-top:8px;line-height:1.8;">{lines_html}</div>' 





def do_reset_database():

    return reset_all_databases()





def rebuild_prototypes():

    temp_gallery = GalleryManager(max_images=100)

    n = load_gallery_from_pinecone(temp_gallery)

    if n == 0:

        return "No gallery data found in Pinecone."

    try: pinecone_body_index.delete(delete_all=True, namespace="")

    except: pass

    batch = []

    for gid in temp_gallery.gids:

        d = temp_gallery.classes[gid]

        if not d['embs']: continue

        embs = torch.stack(d['embs'])

        qs = torch.tensor(d['scores'], device=REID_DEVICE) * torch.tensor(d['ratios'], device=REID_DEVICE)

        indices = alpha_fps(embs, qs, K=3, alpha=0.5)

        for pi, idx in enumerate(indices):

            batch.append({"id": f"global_{gid}_proto_{pi+1}", "values": embs[idx].cpu().numpy().tolist(),

                          "metadata": {"global_id": gid, "prototype_idx": pi+1}})

    if batch:

        pinecone_body_index.upsert(vectors=batch, namespace="")

    return f"Built {len(batch)} prototypes for {n} persons."





def show_reset_confirm():

    return gr.update(visible=True), "**⚠️ Are you sure?** This will delete all data in MongoDB, Pinecone, and local files."





def browse_all_html():

    """Build a card-based bulk view of all persons from MongoDB."""

    all_persons = list(persons_col.find({}, {"_id": 0}).sort("global_id", 1))

    if not all_persons:

        return "<p style='color:#94a3b8;padding:20px;'>No persons in database yet.</p>"



    lbl    = "color:#64748b;"

    val_s  = "color:#e2e8f0;font-weight:600;"

    face_card_bg = "background:#0d1f17;border:2px solid #10b981;border-radius:10px;overflow:hidden;min-width:140px;max-width:160px;flex-shrink:0;text-align:center;"

    body_card_bg = "background:#12121a;border:1px solid #2d3748;border-radius:10px;overflow:hidden;min-width:140px;max-width:160px;flex-shrink:0;text-align:center;"

    img_style = "width:100%;height:160px;object-fit:contain;background:#0a0a14;display:block;"



    html = f'<div style="padding:8px 0;"><h2 style="color:#a5b4fc;margin-bottom:16px;">All Identities ({len(all_persons)} total)</h2>'



    for doc in all_persons:

        gid      = doc.get("global_id", "?")

        sightings = doc.get("sightings", [])

        faces     = doc.get("faces", [])

        n_visits  = len(sightings)

        n_faces   = len(faces)

        ik_link  = ""  # ImageKit folder deep-links require auth, omitted



        html += f'<div style="background:#12121a;border:1px solid #1e293b;border-radius:14px;padding:14px;margin-bottom:20px;">'

        html += (f'<h3 style="color:#a5b4fc;margin:0 0 10px 0;font-size:16px;">'

                 f'GID {gid:03d} &mdash; '

                 f'<span style="color:#10b981;">&#128065; {n_faces} face(s)</span> &nbsp; '

                 f'<span style="color:#94a3b8;">{n_visits} visit(s)</span></h3>')

        html += '<div style="display:flex;gap:10px;overflow-x:auto;padding-bottom:6px;">'



        # Face cards first

        for f in faces[:8]:  # cap at 8 per row for perf

            furl = f.get("url", "") or f.get("best_face_url", "")

            if furl:

                html += f'<div style="{face_card_bg}"><img src="{furl}" style="{img_style}"/><div style="padding:4px;font-size:11px;color:#10b981;">Face</div></div>'



        # Body cards

        for s in sightings[:8]:

            burl = s.get("best_crop_url", "")

            if burl:

                html += f'<div style="{body_card_bg}"><img src="{burl}" style="{img_style}"/><div style="padding:4px;font-size:11px;color:#94a3b8;">Track {s.get("track_id","?")}</div></div>'



        if not any(f.get("url") or f.get("best_face_url") for f in faces) and not any(s.get("best_crop_url") for s in sightings):

            html += '<span style="color:#475569;font-size:13px;padding:8px;">No saved images yet</span>'



        html += '</div></div>'



    html += '</div>'

    return html





def browse_database():

    all_persons = list(persons_col.find({}, {"_id": 0, "global_id": 1,

                                              "first_seen_ts": 1, "last_seen_ts": 1,

                                              "sightings": 1, "faces": 1}).sort("global_id", 1))

    if not all_persons:

        return "<p style='color:#94a3b8;'>No persons in database.</p>", gr.update(choices=[], value=None)

    choices = []

    rows = ""

    for i, p in enumerate(all_persons):

        gid = p["global_id"]

        nv  = len(p.get("sightings", []))

        nf  = len(p.get("faces", []))

        fs  = datetime.fromtimestamp(p.get("first_seen_ts", 0)).strftime('%H:%M:%S') if p.get("first_seen_ts") else "?"

        ls  = datetime.fromtimestamp(p.get("last_seen_ts", 0)).strftime('%H:%M:%S') if p.get("last_seen_ts") else "?"

        bg  = "#12121a" if i % 2 == 0 else "#0f0f18"

        rows += (f'<tr style="background:{bg};">'

                 f'<td style="padding:7px 14px;color:#a5b4fc;font-weight:600;">{gid:03d}</td>'

                 f'<td style="padding:7px 14px;color:#cbd5e1;">{nv}</td>'

                 f'<td style="padding:7px 14px;color:#10b981;">{nf}</td>'

                 f'<td style="padding:7px 14px;color:#94a3b8;">{fs}</td>'

                 f'<td style="padding:7px 14px;color:#94a3b8;">{ls}</td></tr>')

        choices.append(str(gid))

    html = (

        f'<details open style="background:#12121a;border:1px solid #1e293b;border-radius:12px;padding:14px;margin-bottom:16px;">'

        f'<summary style="cursor:pointer;color:#a5b4fc;font-size:16px;font-weight:700;list-style:none;padding:2px 0;">'

        f'▸ Database Overview — {len(all_persons)} persons (click to toggle)</summary>'

        f'<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:12px;">'

        f'<tr style="background:#1a1a2e;">'

        f'<th style="padding:8px 14px;color:#a5b4fc;text-align:left;">GID</th>'

        f'<th style="padding:8px 14px;color:#a5b4fc;text-align:left;">Visits</th>'

        f'<th style="padding:8px 14px;color:#a5b4fc;text-align:left;">Faces</th>'

        f'<th style="padding:8px 14px;color:#a5b4fc;text-align:left;">First Seen</th>'

        f'<th style="padding:8px 14px;color:#a5b4fc;text-align:left;">Last Seen</th></tr>'

        f'{rows}</table></details>'

    )

    return html, gr.update(choices=choices, value=choices[0] if choices else None)





def browse_person(gid_str):

    if not gid_str: return "Select a person.", [], []

    return get_person_details(int(gid_str))





# ═══════════════════════════════════════════════════════════════

# BUILD THE DASHBOARD

# ═══════════════════════════════════════════════════════════════



with gr.Blocks(css=THEME_CSS, title="ReID Dashboard", theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate")) as demo:

    gr.HTML("""

    <div style="text-align:center; padding: 20px 0 10px 0;">

        <h1 style="color:#ffffff !important; font-size:36px; font-weight:800; margin:0; letter-spacing:-1px;">

            <span style="color:#a5b4fc;">◉</span> Moraqib AI

        </h1>

        <p style="color:#94a3b8 !important; font-size:15px; margin:6px 0 0 0; font-weight:400;">

            Multi-Modal Person Re-Identification &amp; Face Association — Intelligent Retail Surveillance

        </p>

    </div>

    """)



    with gr.Tabs():

        # ── TAB 1: VIDEO PROCESSING ──

        with gr.TabItem("Video Processing"):

            gr.Markdown("### Input")

            with gr.Row():

                with gr.Column(scale=1):

                    vid_file = gr.File(label="Upload Video", file_types=["video"])

                    vid_path = gr.Textbox(label="Or enter dataset path",

                                          placeholder="/kaggle/input/.../video.mp4", value="")

                with gr.Column(scale=1):

                    cam_id = gr.Textbox(label="Camera ID", value=CAMERA_ID, placeholder="cam_01")

                    is_face = gr.Checkbox(label="Face Camera", value=True)

                    use_existing = gr.Checkbox(label="Load existing gallery from Pinecone", value=True)



            with gr.Accordion("Parameters", open=False):

                gr.Markdown("**Detection & Quality**")

                with gr.Row():

                    conf_sl = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Detection Confidence")

                    pose_sl = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Pose Keypoint Conf")

                    vis_sl = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Min Visible Ratio")

                gr.Markdown("**Identity Resolution**")

                with gr.Row():

                    tau_sl = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Softmax Tau")

                    cos_sl = gr.Slider(0.1, 0.95, value=0.78, step=0.01, label="Min Cosine Similarity")

                    budget_sl = gr.Slider(10, 100, value=50, step=5, label="Gallery Budget")

                gr.Markdown("**Face Recognition**")

                with gr.Row():

                    fc_sl = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Face Detection Conf")

                    fm_sl = gr.Slider(0.1, 1.0, value=0.55, step=0.05, label="Face Match Threshold")

                gr.Markdown("**Unknown Data Manager (DBSCAN)**")

                with gr.Row():

                    ue_sl = gr.Slider(0.05, 0.5, value=0.15, step=0.05, label="Epsilon")

                    us_sl = gr.Slider(2, 10, value=5, step=1, label="Min Cluster Size")



            run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

            processing_note = gr.HTML(value="", visible=True, elem_classes=["min-height-block"])

            vid_output = gr.HTML(label="Results", elem_classes=["min-height-block"])



            with gr.Row(visible=False) as save_row:

                save_btn = gr.Button("Save Results to Cloud", elem_classes=["save-btn"], size="lg")

            save_status = gr.HTML(value="", visible=True, elem_classes=["min-height-block"])



            run_btn.click(

                fn=run_video_handler,

                inputs=[vid_file, vid_path, cam_id, is_face, use_existing,

                        conf_sl, tau_sl, cos_sl, budget_sl, pose_sl, vis_sl, fc_sl, fm_sl, ue_sl, us_sl],

                outputs=[vid_output, save_row]

            )

            save_btn.click(fn=do_save_results, outputs=[save_status])



        # ── TAB 2: PERSON SEARCH ──

        with gr.TabItem("Person Search"):

            with gr.Row():

                with gr.Column(scale=1):

                    search_img = gr.Image(label="Upload Query Image", type="numpy")

                    search_url = gr.Textbox(label="Or Image URL", placeholder="https://example.com/image.jpg")

                    search_mode = gr.Radio(["Body", "Face", "Both"], value="Both", label="Search Mode")

                    is_crop = gr.Checkbox(label="Image is a pre-cropped face", value=False)

                with gr.Column(scale=1):

                    with gr.Accordion("Search Settings", open=False):

                        body_th = gr.Slider(0.3, 1.0, value=0.75, step=0.05, label="Body Threshold")

                        face_th = gr.Slider(0.2, 1.0, value=0.55, step=0.05, label="Face Threshold")

                        search_tau = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Tau")

                    search_btn = gr.Button("Search", variant="primary", size="lg")

                    proto_btn = gr.Button("Rebuild Body Prototypes", variant="secondary")

                    proto_out = gr.Markdown()



            search_summary = gr.Markdown(label="Search Results")

            match_img = gr.Image(label="Best Match", height=300)

            person_details = gr.HTML()



            search_btn.click(fn=run_search,

                inputs=[search_img, search_url, search_mode, is_crop, body_th, face_th, search_tau],

                outputs=[search_summary, match_img, person_details])

            proto_btn.click(fn=rebuild_prototypes, outputs=[proto_out])



        # ── TAB 3: DATABASE BROWSER ──

        with gr.TabItem("Database Browser"):

            with gr.Row():

                refresh_btn = gr.Button("🔄 Refresh Database", variant="primary")

                reset_btn = gr.Button("🗑️ Reset All Databases", elem_classes=["danger-btn"])



            reset_output = gr.Markdown()

            reset_confirm = gr.Button("Yes, DELETE EVERYTHING", visible=False, elem_classes=["danger-btn"])



            reset_btn.click(fn=show_reset_confirm, outputs=[reset_confirm, reset_output])

            reset_confirm.click(fn=do_reset_database, outputs=[reset_output])



            with gr.Tabs():

                # ── Sub-tab A: Browse All ──

                with gr.TabItem("👥 Browse All"):

                    bulk_html = gr.HTML(value="<p style='color:#94a3b8;'>Click Refresh to load all identities.</p>",

                                        elem_classes=["min-height-block"])



                    refresh_btn.click(fn=browse_all_html, outputs=[bulk_html])



                # ── Sub-tab B: Person Detail ──

                with gr.TabItem("🔍 Person Detail"):

                    db_overview = gr.HTML()

                    person_select = gr.Dropdown(label="Select Person (GID)", choices=[], interactive=True)



                    refresh_btn.click(fn=browse_database, outputs=[db_overview, person_select])



                    person_info = gr.HTML()

                    with gr.Row():

                        body_gallery = gr.Gallery(label="Body Crops", columns=3, object_fit="contain")

                        face_gallery = gr.Gallery(label="Face Crops", columns=3, object_fit="contain")



                    person_select.change(fn=browse_person, inputs=[person_select],

                                         outputs=[person_info, body_gallery, face_gallery])



print("Dashboard built. Run the next cell to launch.")

import gradio as gr



# Clean up any stuck background Gradio servers

try:

    gr.close_all()

except Exception:

    pass



print("Starting Gradio...")

print("⏳ Please wait for the PUBLIC URL (it will end in .gradio.live)")

print("⚠️ Note: The cell must stay running to keep the server alive.")



# Use the standard blocking share method. 

# Bypassing the proxy/threading hacks which often fail in Kaggle.

demo.queue().launch(

    share=True,

    server_name="127.0.0.1",

    inline=False,

    debug=True

)
