"""
pipeline.py - Moteur d'analyse sprint : pose -> stabilisation caméra -> mètres -> appuis -> métriques.

Chaîne :
 1. Lecture vidéo image par image (réduite à 960 px de large max).
 2. Pose (RTMPose via rtmlib) sur l'athlète.
 3. Stabilisation : le mouvement de la caméra est estimé sur l'ARRIÈRE-PLAN (flux optique
    + transformation de similarité robuste RANSAC), athlète masqué. Tout est ramené
    au repère de la 1re image (translation, rotation ET zoom compensés).
 4. Échelle : longueur des jambes mesurée / (0,491 x taille de l'athlète).
 5. Appuis au sol dans le repère monde + métriques avec incertitudes.
"""
import shutil
import subprocess
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

MAX_WIDTH = 960
LEG_RATIO = 0.491          # (cuisse + tibia) / taille, Dempster
SCALE_UNCERTAINTY = 0.06   # incertitude relative sur l'échelle (taille -> longueur des jambes)
CONTACT_MODEL_BAND_S = 0.025


# =====================================================================
# 1. Signal : trous, lissage
# =====================================================================

def fill_gaps(sig, valid, max_gap=6):
    sig = np.asarray(sig, float).copy()
    valid = np.asarray(valid, bool)
    sig[~valid] = np.nan
    n = len(sig)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:
            sig[i:j] = np.interp(np.arange(i, j), [i - 1, j], [sig[i - 1], sig[j]])
        i = j
    return sig


def butter_smooth(sig, fps, cutoff=15.0, order=4):
    sig = np.asarray(sig, float)
    out = np.full_like(sig, np.nan)
    cutoff = min(cutoff, fps / 2.0 - 1.0)
    b, a = butter(order, cutoff / (fps / 2.0))
    ok = ~np.isnan(sig)
    i, n = 0, len(sig)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        seg = sig[i:j]
        out[i:j] = filtfilt(b, a, seg) if len(seg) > 3 * max(len(a), len(b)) else seg
        i = j
    return out


def derivative(sig, fps):
    return np.gradient(sig, 1.0 / fps)


# =====================================================================
# 2. Appuis au sol
# =====================================================================

@dataclass
class Contact:
    foot: str
    t_on: float
    t_off: float
    x: float

    @property
    def duration(self):
        return self.t_off - self.t_on


def _cross(t0, t1, v0, v1, thr):
    if v1 == v0:
        return t0
    return t0 + (thr - v0) / (v1 - v0) * (t1 - t0)


def detect_contacts(x, h, fps, foot="L", alpha=0.1, k_noise=3.0, h_tol=0.06,
                    min_dur=0.04, max_dur=0.35):
    """Contact = pied lent ET bas (repère sol, en mètres). Seuil de vitesse adaptatif."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    n = len(x)
    t = np.arange(n) / fps
    speed = np.hypot(derivative(x, fps), derivative(h, fps))
    ok = ~np.isnan(speed)
    if ok.sum() < 10:
        return []
    vp = np.nanpercentile(speed, 90)
    low = np.sort(speed[ok])[:max(5, ok.sum() // 5)]
    thr = max(alpha * vp, k_noise * np.median(low), 1e-6)
    h_base = np.nanpercentile(h, 5)
    score = np.maximum(speed / thr, (h - h_base) / h_tol)
    mask = (score < 1.0) & ok
    res, i = [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if i > 0 and j < n:
            t_on = _cross(t[i - 1], t[i], score[i - 1], score[i], 1.0)
            t_off = _cross(t[j - 1], t[j], score[j - 1], score[j], 1.0)
            if min_dur <= (t_off - t_on) <= max_dur:
                res.append(Contact(foot, float(t_on), float(t_off), float(np.nanmedian(x[i:j]))))
        i = j
    return res


def sprint_metrics(contacts, fps, scale_unc=SCALE_UNCERTAINTY):
    cs = sorted(contacts, key=lambda c: c.t_on)
    if len(cs) < 3:
        return {"erreur": "Pas assez d'appuis détectés (il en faut au moins 3)."}
    ct_unc = float(np.hypot(1.0 / fps, CONTACT_MODEL_BAND_S))
    td = np.array([c.t_on for c in cs])
    to = np.array([c.t_off for c in cs])
    xs = np.array([c.x for c in cs])
    step_t = np.diff(td)
    keep = np.abs(step_t - np.median(step_t)) < 0.3 * np.median(step_t)
    if keep.sum() < 2:
        return {"erreur": "Foulées trop irrégulières pour calculer des métriques."}
    st = step_t[keep]
    flight = (td[1:] - to[:-1])[keep]
    sl = np.diff(xs)[keep]
    ct = np.array([c.duration for c in cs])
    v = float(sl.sum() / st.sum())
    return {
        "appuis_detectes": len(cs),
        "foulees_rejetees": int((~keep).sum()),
        "temps_contact_ms": {"valeur": round(float(1000 * ct.mean()), 1), "incertitude": round(1000 * ct_unc, 1)},
        "temps_vol_ms": {"valeur": round(float(1000 * flight.mean()), 1), "incertitude": round(1000 * ct_unc, 1)},
        "frequence_foulee_hz": {"valeur": round(float((1 / st).mean()), 2), "incertitude": round(float((1 / st).std()), 2)},
        "longueur_foulee_m": {"valeur": round(float(sl.mean()), 2),
                              "incertitude": round(float(np.hypot(sl.std(), sl.mean() * scale_unc)), 2)},
        "vitesse_moyenne_ms": {"valeur": round(v, 2), "incertitude": round(v * scale_unc, 2)},
    }


# =====================================================================
# 3. Stabilisation caméra (arrière-plan)
# =====================================================================

def _background_mask(shape, box):
    m = np.full(shape, 255, np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        pad = 0.2 * max(x1 - x0, y1 - y0)
        m[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)] = 0
    return m


def camera_step(prev_gray, gray, box):
    """Transformation de similarité image courante -> image précédente (fond uniquement)."""
    mask = _background_mask(prev_gray.shape, box)
    p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 12:
        return None, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(31, 31), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 12:
        return None, 0.0
    M, inl = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None:
        return None, 0.0
    return M, float(inl.mean())


# =====================================================================
# 4. Pose (RTMPose via rtmlib)
# =====================================================================

_POSE = {"tracker": None, "lock": threading.Lock()}


def _get_tracker():
    with _POSE["lock"]:
        if _POSE["tracker"] is None:
            try:
                from rtmlib import PoseTracker, BodyWithFeet as Solution
            except ImportError:
                from rtmlib import PoseTracker, Wholebody as Solution
            _POSE["tracker"] = PoseTracker(Solution, det_frequency=5, mode="lightweight",
                                           backend="onnxruntime", device="cpu", tracking=False)
        return _POSE["tracker"]


def rtmlib_pose(frame_bgr, prev_center):
    """Retourne (keypoints (K,2), scores (K,), bbox) de l'athlète, ou (None, None, None)."""
    kpts, scores = _get_tracker()(frame_bgr)
    kpts, scores = np.asarray(kpts), np.asarray(scores)
    if kpts.ndim != 3 or len(kpts) == 0:
        return None, None, None
    best, best_val = 0, -1e18
    for i in range(len(kpts)):
        good = scores[i] > 0.3
        if good.sum() < 5:
            continue
        pts = kpts[i][good]
        h = pts[:, 1].max() - pts[:, 1].min()
        c = pts.mean(axis=0)
        val = h if prev_center is None else -np.hypot(*(c - prev_center))
        if val > best_val:
            best, best_val = i, val
    good = scores[best] > 0.3
    if good.sum() < 5:
        return None, None, None
    pts = kpts[best][good]
    box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    return kpts[best], scores[best], box


def keypoint_map(K):
    """Indices des points utiles selon le modèle (Halpe-26 ou Wholebody-133)."""
    if K == 26:
        return dict(hip=(11, 12), knee=(13, 14), ankle=(15, 16), toe=(20, 21), heel=(24, 25))
    if K == 133:
        return dict(hip=(11, 12), knee=(13, 14), ankle=(15, 16), toe=(17, 20), heel=(19, 22))
    raise ValueError(f"Modèle de pose non pris en charge ({K} points)")


# =====================================================================
# 5. Pipeline complet
# =====================================================================

def run_pipeline(video_path, height_m=1.75, fps_override=None, max_seconds=8.0,
                 progress=lambda frac: None, pose_fn=None, out_video=None, playback="slow"):
    """pose_fn(frame_index, frame_bgr, prev_center) -> (kpts, scores, box) ; par défaut RTMPose.
    progress(frac) avec frac entre 0 et 1. Si out_video est donné, écrit la vidéo annotée."""
    pose_fn = pose_fn or (lambda i, f, c: rtmlib_pose(f, c))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Vidéo illisible.")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps_override) if fps_override else float(file_fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n_max = int(max_seconds * fps)
    n_target = min(total, n_max) if total else n_max

    kp_list, sc_list, T_list, inl_list = [], [], [], []
    T = np.eye(3)
    prev_gray, prev_box, prev_center = None, None, None
    K = None
    i = 0
    while i < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        if w0 > MAX_WIDTH:
            s = MAX_WIDTH / w0
            frame = cv2.resize(frame, (MAX_WIDTH, int(h0 * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -- caméra
        if prev_gray is not None:
            M, inl = camera_step(prev_gray, gray, prev_box)
            if M is not None:
                T = T @ np.vstack([M, [0, 0, 1]])
            inl_list.append(inl)
        T_list.append(T.copy())

        # -- pose
        kp, sc, box = pose_fn(i, frame, prev_center)
        if kp is None:
            kp_list.append(None)
            sc_list.append(None)
        else:
            K = len(kp)
            kp_list.append(np.asarray(kp, float))
            sc_list.append(np.asarray(sc, float))
            prev_box = box
            prev_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        prev_gray = gray
        i += 1
        progress(0.7 * i / max(n_target, 1) if out_video else i / max(n_target, 1))
    cap.release()

    n = len(T_list)
    if n < 30 or K is None:
        raise ValueError("Trop peu d'images exploitables (athlète non détecté ?).")
    result, ctx = analyse(kp_list, sc_list, T_list, inl_list, fps, file_fps, height_m, K)
    result["video_annotee"] = False
    if out_video:
        try:
            render_video(video_path, ctx, out_video, playback, lambda f: progress(0.7 + 0.3 * f))
            result["video_annotee"] = True
        except Exception as e:  # noqa
            result["avertissements"].append(f"Vidéo annotée non générée : {e}")
    return result


def analyse(kp_list, sc_list, T_list, inl_list, fps, file_fps, height_m, K):
    n = len(T_list)
    km = keypoint_map(K)
    kp = np.full((n, K, 2), np.nan)
    sc = np.zeros((n, K))
    for i in range(n):
        if kp_list[i] is None:
            continue
        pts = np.c_[kp_list[i], np.ones(K)] @ T_list[i][:2].T   # -> repère de la 1re image
        kp[i], sc[i] = pts, sc_list[i]
    THR = 0.3
    kp[sc < THR] = np.nan

    def series(idx, axis):
        s = kp[:, idx, axis]
        return fill_gaps(s, ~np.isnan(s))

    # -- échelle : longueur des jambes
    legs = []
    for side in (0, 1):
        hp = np.stack([kp[:, km["hip"][side], 0], kp[:, km["hip"][side], 1]], 1)
        kn = np.stack([kp[:, km["knee"][side], 0], kp[:, km["knee"][side], 1]], 1)
        an = np.stack([kp[:, km["ankle"][side], 0], kp[:, km["ankle"][side], 1]], 1)
        legs.append(np.hypot(*(hp - kn).T) + np.hypot(*(kn - an).T))
    legs = np.concatenate(legs)
    legs = legs[~np.isnan(legs)]
    if len(legs) < 20:
        raise ValueError("Jambes trop rarement visibles pour estimer l'échelle.")
    ppm = float(np.percentile(legs, 90) / (LEG_RATIO * height_m))   # pixels (repère 1re image) par mètre

    # -- sens de la course
    hip_x = np.nanmean([kp[:, km["hip"][0], 0], kp[:, km["hip"][1], 0]], axis=0)
    valid_hip = np.where(~np.isnan(hip_x))[0]
    sign = 1.0 if hip_x[valid_hip[-1]] >= hip_x[valid_hip[0]] else -1.0

    # -- pieds en mètres (repère sol)
    foot_y_all = []
    feet = {}
    for side, name in ((0, "G"), (1, "D")):
        yh, yt = series(km["heel"][side], 1), series(km["toe"][side], 1)
        xh, xt = series(km["heel"][side], 0), series(km["toe"][side], 0)
        y = np.fmax(yh, yt)                    # point le plus bas à l'écran
        x = (xh + xt) / 2
        feet[name] = (x, y)
        foot_y_all.append(y[~np.isnan(y)])
    y_ground = float(np.percentile(np.concatenate(foot_y_all), 95))

    contacts = []
    cutoff = min(25.0, fps / 2 - 5)
    for name, (x, y) in feet.items():
        xm = sign * x / ppm
        hm = (y_ground - y) / ppm
        contacts += detect_contacts(butter_smooth(xm, fps, cutoff), butter_smooth(hm, fps, cutoff), fps, name)

    metrics = sprint_metrics(contacts, fps)
    contacts = sorted(contacts, key=lambda c: c.t_on)

    # -- squelette lissé (repère 1re image) + angles
    need = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(km["heel"]) + list(km["toe"])
    kps = np.full((n, K, 2), np.nan)
    for j in need:
        for ax in (0, 1):
            col = kp[:, j, ax]
            kps[:, j, ax] = butter_smooth(fill_gaps(col, ~np.isnan(col), max_gap=10), fps, 15.0)

    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cos = (v1 * v2).sum(1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    knee = np.stack([ang(kps[:, km["hip"][s_]], kps[:, km["knee"][s_]], kps[:, km["ankle"][s_]]) for s_ in (0, 1)], 1)
    sh = (kps[:, 5] + kps[:, 6]) / 2
    hp = (kps[:, 11] + kps[:, 12]) / 2
    d = sh - hp
    trunk = np.degrees(np.arctan2(d[:, 0] * sign, -d[:, 1]))       # + = penché vers l'avant

    ang_summary = {}
    if np.isfinite(trunk).any():
        ang_summary["inclinaison_tronc_deg"] = {"valeur": round(float(np.nanmean(trunk)), 1),
                                                "incertitude": round(float(np.nanstd(trunk)), 1)}
    td_knee = []
    for c in contacts:
        fi = int(round(c.t_on * fps))
        si = 0 if c.foot == "G" else 1
        if 0 <= fi < n and not np.isnan(knee[fi, si]):
            td_knee.append(knee[fi, si])
    if td_knee:
        ang_summary["angle_genou_a_l_attaque_deg"] = {"valeur": round(float(np.mean(td_knee)), 1),
                                                      "incertitude": round(float(np.std(td_knee)), 1)}
    if isinstance(metrics, dict) and "erreur" not in metrics:
        metrics.update(ang_summary)

    # -- vitesse du centre de masse (hanches), lissée
    xm = sign * butter_smooth(fill_gaps(hip_x, ~np.isnan(hip_x)), fps, 6.0) / ppm
    speed = np.abs(derivative(xm, fps))
    speed_pts = [round(float(v), 2) if not np.isnan(v) else None for v in speed[::max(1, n // 200)]]
    vmax = float(np.nanmax(speed)) if np.isfinite(speed).any() else None

    # -- qualité & avertissements
    pose_ok = float(np.mean([k is not None for k in kp_list]))
    foot_ok = float(np.mean(~np.isnan(kp[:, km["heel"][0], 0]) | ~np.isnan(kp[:, km["heel"][1], 0])))
    inl = float(np.median(inl_list)) if inl_list else 0.0
    warns = []
    if abs(fps - file_fps) > 1:
        warns.append(f"Cadence forcée à {fps:.0f} images/s (le fichier indique {file_fps:.0f}).")
    if fps < 100:
        warns.append("Cadence < 100 images/s : les temps de contact et de vol sont très peu fiables. Filme en 120-240 images/s.")
    if pose_ok < 0.9:
        warns.append(f"Athlète détecté sur seulement {100 * pose_ok:.0f} % des images.")
    if inl < 0.5:
        warns.append("Stabilisation peu fiable (peu de fond net exploitable) : les distances et vitesses peuvent être fausses.")
    warns.append("Les distances et vitesses dépendent de la taille saisie (±6 % environ). Les temps et la fréquence n'en dépendent pas.")
    warns.append("Hypothèse : caméra de profil, athlète dans un plan à peu près perpendiculaire à l'axe de la caméra.")

    ctx = dict(n=n, fps=fps, K=K, km=km, kps=kps, T=T_list, sign=sign, ppm=ppm, y_ground=y_ground,
               contacts=contacts, knee=knee, trunk=trunk, speed=speed, xm=xm)
    result = {
        "fps_utilise": fps,
        "images_analysees": n,
        "echelle_px_par_m": round(ppm, 1),
        "sens_course": "gauche -> droite" if sign > 0 else "droite -> gauche",
        "vitesse_max_hanches_ms": round(vmax, 2) if vmax else None,
        "metriques": metrics,
        "appuis": [{"pied": c.foot, "debut_s": round(c.t_on, 3), "fin_s": round(c.t_off, 3),
                    "duree_ms": round(1000 * c.duration, 1), "x_m": round(c.x, 2)}
                   for c in sorted(contacts, key=lambda c: c.t_on)],
        "courbe_vitesse_ms": speed_pts,
        "qualite": {"athlete_detecte_pct": round(100 * pose_ok), "pieds_visibles_pct": round(100 * foot_ok),
                    "stabilisation_fiabilite_pct": round(100 * inl)},
        "avertissements": warns,
    }
    return result, ctx


# =====================================================================
# 6. Vidéo annotée (montage biomécanique)
# =====================================================================

_GREEN, _ORANGE, _CYAN, _WHITE, _RED = (80, 220, 90), (40, 150, 255), (255, 200, 60), (255, 255, 255), (70, 70, 240)


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg introuvable : impossible d'encoder la vidéo annotée.")


def _text(img, txt, org, scale, color, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _to_frame(Tinv, pts):
    """Points du repère de référence -> image courante. pts (...,2)."""
    p = np.asarray(pts, float)
    return p @ Tinv[:2, :2].T + Tinv[:2, 2]


def render_video(video_path, ctx, out_path, playback="slow", progress=lambda f: None):
    """Relit la vidéo et dessine : squelette, angles, appuis, empreintes au sol, bandeau de métriques.
    playback='slow' : toutes les images à 30 i/s (ralenti) ; 'real' : vitesse réelle."""
    n, fps, km = ctx["n"], ctx["fps"], ctx["km"]
    kps, T, sign, ppm, yg = ctx["kps"], ctx["T"], ctx["sign"], ctx["ppm"], ctx["y_ground"]
    contacts = ctx["contacts"]
    step = 1 if playback == "slow" else max(1, int(round(fps / 30.0)))
    out_fps = 30.0 if playback == "slow" else fps / step

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Relecture de la vidéo impossible.")
    h0, w0 = frame.shape[:2]
    if w0 > MAX_WIDTH:
        w, h = MAX_WIDTH, int(h0 * MAX_WIDTH / w0)
    else:
        w, h = w0, h0
    w, h = w - w % 2, h - h % 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sc = w / 960.0
    th = max(1, int(round(2 * sc)))

    proc = subprocess.Popen(
        [_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{out_fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    # squelette : paires de points
    L, R = 0, 1
    edges = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10)]
    for s_ in (L, R):
        edges += [(km["hip"][s_], km["knee"][s_]), (km["knee"][s_], km["ankle"][s_]),
                  (km["ankle"][s_], km["heel"][s_]), (km["ankle"][s_], km["toe"][s_]),
                  (km["heel"][s_], km["toe"][s_])]
    left_set = {5, 7, 9, 11, 13, 15, km["heel"][L], km["toe"][L]}
    last_ct = {"G": None, "D": None}
    names = ("G", "D")

    try:
        i = 0
        while i < n:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                t = i / fps
                Tinv = np.linalg.inv(T[i])
                P = _to_frame(Tinv, kps[i])                      # (K,2) en pixels image
                ok_pt = ~np.isnan(P).any(axis=1)

                # sol + empreintes collées au sol (compensent le mouvement de caméra)
                done = [c for c in contacts if c.t_on <= t]
                prev_pt, prev_c = None, None
                for c in done:
                    pt = _to_frame(Tinv, [c.x * sign * ppm, yg])
                    if 0 <= pt[0] < w and 0 <= pt[1] < h:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), int(7 * sc), _ORANGE if c.foot == "G" else _CYAN, -1, cv2.LINE_AA)
                    if prev_pt is not None:
                        a, b = prev_pt, pt
                        if 0 <= a[0] < w and 0 <= b[0] < w:
                            y2 = int(min(a[1], b[1]) + 14 * sc)
                            cv2.line(frame, (int(a[0]), y2), (int(b[0]), y2), _WHITE, 1, cv2.LINE_AA)
                            _text(frame, f"{abs(c.x - prev_c.x):.2f} m",
                                  (int((a[0] + b[0]) / 2 - 22 * sc), y2 + int(16 * sc)), 0.45 * sc, _WHITE)
                    prev_pt, prev_c = pt, c

                # squelette
                for a, b in edges:
                    if ok_pt[a] and ok_pt[b]:
                        col = _ORANGE if (a in left_set and b in left_set) else (_CYAN if a not in (5, 6, 11, 12) or b not in (5, 6, 11, 12) else _WHITE)
                        cv2.line(frame, tuple(P[a].astype(int)), tuple(P[b].astype(int)), col, th + 1, cv2.LINE_AA)
                for j in range(len(P)):
                    if ok_pt[j] and kps[i, j, 0] == kps[i, j, 0] and j in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, *km["heel"], *km["toe"]):
                        cv2.circle(frame, tuple(P[j].astype(int)), int(3 * sc) + 1, _WHITE, -1, cv2.LINE_AA)

                # appui en cours
                phase, in_c = "VOL", None
                for c in contacts:
                    if c.t_on <= t < c.t_off:
                        in_c = c
                        break
                    if c.t_off <= t:
                        last_ct[c.foot] = c
                if in_c is not None:
                    phase = "APPUI " + in_c.foot
                    s_ = names.index(in_c.foot)
                    pf = (P[km["heel"][s_]] + P[km["toe"][s_]]) / 2
                    if not np.isnan(pf).any():
                        cv2.circle(frame, tuple(pf.astype(int)), int(16 * sc), _GREEN, th + 1, cv2.LINE_AA)

                # angles de genou à côté des genoux
                for s_ in (L, R):
                    k_ = km["knee"][s_]
                    a_ = ctx["knee"][i, s_]
                    if ok_pt[k_] and not np.isnan(a_):
                        _text(frame, f"{a_:.0f}", (int(P[k_][0] + 8 * sc), int(P[k_][1])), 0.5 * sc, _ORANGE if s_ == L else _CYAN, th)

                # bandeau d'infos
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, int(58 * sc)), (20, 20, 20), -1)
                cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
                v = ctx["speed"][i]
                tr = ctx["trunk"][i]
                _text(frame, f"t {t:5.2f}s", (int(8 * sc), int(20 * sc)), 0.55 * sc, _WHITE)
                _text(frame, f"V {v:4.1f} m/s" if v == v else "V --", (int(120 * sc), int(20 * sc)), 0.55 * sc, _GREEN)
                _text(frame, f"Tronc {tr:3.0f} deg" if tr == tr else "Tronc --", (int(270 * sc), int(20 * sc)), 0.55 * sc, _WHITE)
                gt = f"{1000 * last_ct['G'].duration:.0f}ms" if last_ct["G"] else "--"
                dt = f"{1000 * last_ct['D'].duration:.0f}ms" if last_ct["D"] else "--"
                _text(frame, f"Contact G {gt}", (int(8 * sc), int(46 * sc)), 0.55 * sc, _ORANGE)
                _text(frame, f"Contact D {dt}", (int(190 * sc), int(46 * sc)), 0.55 * sc, _CYAN)
                seen = [c for c in contacts if c.t_on <= t]
                if len(seen) >= 3:
                    dts = np.diff([c.t_on for c in seen])
                    _text(frame, f"Foulee {1 / np.median(dts):.2f} Hz", (int(380 * sc), int(46 * sc)), 0.55 * sc, _WHITE)
                col = _GREEN if phase.startswith("APPUI") else _RED
                _text(frame, phase, (int(w - 120 * sc), int(38 * sc)), 0.8 * sc, col, th)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            i += 1
            if i % 10 == 0:
                progress(i / n)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Échec de l'encodage de la vidéo annotée.")
    progress(1.0)


def parse_range(header, size):
    """En-tête HTTP Range -> (start, end) inclusifs, ou None si absent/invalide. Nécessaire pour lire la vidéo sur iPhone."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    a, _, b = header[6:].partition("-")
    try:
        if a == "":
            start, end = max(0, size - int(b)), size - 1
        else:
            start, end = int(a), (int(b) if b else size - 1)
    except ValueError:
        return None
    end = min(end, size - 1)
    return (start, end) if 0 <= start <= end else "invalid"
