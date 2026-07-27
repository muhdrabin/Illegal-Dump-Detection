import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# Tunable constants

PROXIMITY_PX      = 150
ASSOC_MIN_FRAMES  = 8
DEPART_PX         = 180
STATIONARY_VEL    = 5.0
STATIONARY_FRAMES = 10
BASELINE_FRAMES   = 60
BUFFER_SECONDS    = 10
POST_SECONDS      = 3
TRASH_ID_OFFSET   = 10_000
CROP_PAD          = 20

# Dump state machine
class DumpState(Enum):
    NEW        = auto()
    ASSOCIATED = auto()
    DUMPED     = auto()
    BASELINE   = auto()
    DISMISSED  = auto()


@dataclass
class TrashDumpTrack:
    tid:               int
    state:             DumpState = DumpState.NEW
    centre:            tuple     = (0.0, 0.0)
    prev_centre:       tuple     = (0.0, 0.0)
    velocity:          float     = 0.0
    assoc_actor_id:    Optional[int] = None
    assoc_actor_label: str       = ''
    assoc_frames:      int       = 0
    stationary_frames: int       = 0
    actor_last_centre: tuple     = (0.0, 0.0)
    best_actor_bbox:   Optional[tuple]      = None
    best_actor_frame:  Optional[np.ndarray] = None
    best_actor_conf:   float      = 0.0

# Kalman filter — one per trash track

class TrashKalmanTrack:
    _next_stable_id = 0

    def __init__(self, bbox, bytetrack_id: int, max_missed: int = 10):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  =  bbox[2] - bbox[0]
        h  =  bbox[3] - bbox[1]

        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 25.0
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

        self.stable_id     = TrashKalmanTrack._next_stable_id
        TrashKalmanTrack._next_stable_id += 1
        self.bytetrack_ids = {bytetrack_id}
        self.missed_frames = 0
        self.max_missed    = max_missed
        self.last_bbox     = bbox
        self.active        = True

    def predict(self):
        self.kf.predict()

    def update(self, bbox, bytetrack_id: int):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  =  bbox[2] - bbox[0]
        h  =  bbox[3] - bbox[1]
        self.kf.correct(np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1))
        self.bytetrack_ids.add(bytetrack_id)
        self.missed_frames = 0
        self.last_bbox     = bbox
        self.active        = True

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > self.max_missed:
            self.active = False

    def predicted_bbox(self):
        s = self.kf.statePost.flatten()
        cx, cy, w, h = s[0], s[1], max(1, s[2]), max(1, s[3])
        return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))

    def iou(self, bbox) -> float:
        ax1, ay1, ax2, ay2 = self.predicted_bbox()
        bx1, by1, bx2, by2 = bbox
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        iw, ih   = max(0,ix2-ix1), max(0,iy2-iy1)
        inter    = iw * ih
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

# Kalman manager

class TrashKalmanManager:
    def __init__(self, max_missed: int = 10, iou_thresh: float = 0.08):
        self.tracks:    dict = {}
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh

    def update(self, detections: list) -> list:
        for tr in self.tracks.values():
            tr.predict()

        matched_stable = set()
        matched_det    = set()

        bt_to_stable = {}
        for tr in self.tracks.values():
            for bt in tr.bytetrack_ids:
                bt_to_stable[bt] = tr.stable_id

        for i, det in enumerate(detections):
            bt = det['track_id']
            if bt in bt_to_stable:
                sid = bt_to_stable[bt]
                if sid in self.tracks:
                    self.tracks[sid].update(det['bbox'], bt)
                    matched_stable.add(sid)
                    matched_det.add(i)

        unmatched_dets   = [i for i in range(len(detections)) if i not in matched_det]
        unmatched_tracks = [tr for tr in self.tracks.values()
                            if tr.stable_id not in matched_stable and tr.active]

        for i in unmatched_dets[:]:
            best_iou, best_tr = 0.0, None
            for tr in unmatched_tracks:
                v = tr.iou(detections[i]['bbox'])
                if v > best_iou:
                    best_iou, best_tr = v, tr
            if best_tr is not None and best_iou >= self.iou_thresh:
                best_tr.update(detections[i]['bbox'], detections[i]['track_id'])
                matched_stable.add(best_tr.stable_id)
                matched_det.add(i)
                unmatched_tracks.remove(best_tr)

        for i in range(len(detections)):
            if i not in matched_det:
                det = detections[i]
                nt  = TrashKalmanTrack(det['bbox'], det['track_id'], self.max_missed)
                self.tracks[nt.stable_id] = nt
                matched_stable.add(nt.stable_id)

        for tr in self.tracks.values():
            if tr.stable_id not in matched_stable:
                tr.mark_missed()

        output = []
        for i, det in enumerate(detections):
            if i in matched_det:
                bt  = det['track_id']
                sid = bt_to_stable.get(bt)
                if sid is None:
                    for tr in self.tracks.values():
                        if bt in tr.bytetrack_ids:
                            sid = tr.stable_id
                            break
                if sid is not None:
                    out = dict(det)
                    out['track_id'] = sid + TRASH_ID_OFFSET
                    output.append(out)

        for tr in self.tracks.values():
            if tr.active and tr.stable_id not in matched_stable:
                if 0 < tr.missed_frames <= 8:
                    output.append({
                        'bbox':       tr.predicted_bbox(),
                        'label':      'trash',
                        'confidence': 0.0,
                        'track_id':   tr.stable_id + TRASH_ID_OFFSET,
                    })

        self.tracks = {sid: tr for sid, tr in self.tracks.items() if tr.active}
        return output

    def reset(self):
        self.tracks.clear()
        TrashKalmanTrack._next_stable_id = 0

# RAM ring buffer

class RingBuffer:
    def __init__(self, fps: int, seconds: int):
        self._maxlen = max(1, fps * seconds)
        self._buf: deque = deque(maxlen=self._maxlen)

    def push(self, frame: np.ndarray):
        self._buf.append(frame.copy())

    def snapshot(self) -> list:
        return list(self._buf)

# Main tracker + dump detector

class PersonVehicleTrashTracker:
    TARGET_COCO_IDS = [0, 1, 2, 3, 5, 7]
    COCO_TO_LABEL   = {0:'person', 1:'vehicle', 2:'vehicle',
                       3:'vehicle', 5:'vehicle', 7:'vehicle'}
    LABEL_COLOR = {
        'person':  (0,   200, 255),
        'vehicle': (0,   255, 100),
        'trash':   (0,    80, 255),
    }
    STATE_BORDER = {
        DumpState.NEW:        (200, 200,   0),
        DumpState.ASSOCIATED: (0,   165, 255),
        DumpState.DUMPED:     (0,     0, 255),
        DumpState.BASELINE:   (120, 120, 120),
    }

    def __init__(self,
                 coco_model_path:    str   = 'yolov8n.pt',
                 trash_model_path:   str   = 'best.pt',
                 conf_threshold:     float = 0.5,
                 trash_conf:         float = 0.25,
                 tracker:            str   = 'bytetrack.yaml',
                 kalman_max_missed:  int   = 10,
                 kalman_iou_thresh:  float = 0.08,
                 output_dir:         str   = 'dump_events',
                 proximity_px:       float = PROXIMITY_PX,
                 assoc_min_frames:   int   = ASSOC_MIN_FRAMES,
                 depart_px:          float = DEPART_PX,
                 stationary_vel:     float = STATIONARY_VEL,
                 stationary_frames:  int   = STATIONARY_FRAMES,
                 baseline_frames:    int   = BASELINE_FRAMES,
                 buffer_seconds:     int   = BUFFER_SECONDS,
                 post_seconds:       int   = POST_SECONDS):

        print(f"[init] COCO model  : {coco_model_path}")
        self.coco_model  = YOLO(coco_model_path)
        print(f"[init] Trash model : {trash_model_path}")
        self.trash_model = YOLO(trash_model_path)

        self.conf_threshold   = conf_threshold
        self.trash_conf       = trash_conf
        self.tracker_cfg      = tracker
        self.output_dir       = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.proximity_px      = proximity_px
        self.assoc_min_frames  = assoc_min_frames
        self.depart_px         = depart_px
        self.stationary_vel    = stationary_vel
        self.stationary_frames = stationary_frames
        self.baseline_frames   = baseline_frames
        self.buffer_seconds    = buffer_seconds
        self.post_seconds      = post_seconds

        self.kalman_mgr = TrashKalmanManager(
            max_missed = kalman_max_missed,
            iou_thresh = kalman_iou_thresh,
        )

        self.track_history:   dict = defaultdict(list)
        self.track_stats:     dict = {'person': set(), 'vehicle': set(), 'trash': set()}
        self.dump_tracks:     dict = {}
        self.baseline_ids:    set  = set()
        self.confirmed_dumps: int  = 0
        self.frame_count:     int  = 0
        self.fps:             int  = 25
        self.ring_buffer:     Optional[RingBuffer] = None

        self._post_active:    bool = False
        self._post_remaining: int  = 0
        self._post_frames:    list = []
        self._post_meta:      dict = {}
        self._alert_frames:   int  = 0

    # Helpers
  
    def _centre(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def _dist(self, a, b):
        return float(np.hypot(a[0]-b[0], a[1]-b[1]))

    def _clip_bbox(self, bbox, h, w):
        x1, y1, x2, y2 = bbox
        return (max(0,x1), max(0,y1), min(w,x2), min(h,y2))

    def _color_for(self, label, tid=None):
        base = np.array(self.LABEL_COLOR.get(label, (180,180,180)), dtype=np.float32)
        if tid is not None:
            np.random.seed(int(tid) % (2**31-1))
            jitter = np.random.randint(-25, 25, 3).astype(np.float32)
            return tuple(int(np.clip(c, 40, 255)) for c in base + jitter)
        return tuple(int(c) for c in base)

    def _reset_state(self):
        self.track_history.clear()
        self.track_stats     = {'person': set(), 'vehicle': set(), 'trash': set()}
        self.dump_tracks.clear()
        self.baseline_ids.clear()
        self.confirmed_dumps = 0
        self.frame_count     = 0
        self._post_active    = False
        self._post_remaining = 0
        self._post_frames    = []
        self._alert_frames   = 0
        self.kalman_mgr.reset()

    # YOLO inference
    def _run_tracking(self, frame, update_trails=True):
        coco_res = self.coco_model.track(
            frame, persist=True, tracker=self.tracker_cfg,
            classes=self.TARGET_COCO_IDS, conf=self.conf_threshold, verbose=False)
        trash_res = self.trash_model.track(
            frame, persist=True, tracker=self.tracker_cfg,
            conf=self.trash_conf, verbose=False)

        raw_trash    = self._parse_trash_tracked(trash_res, update_trails=False)
        stable_trash = self.kalman_mgr.update(raw_trash)

        if update_trails:
            for t in stable_trash:
                tid = t['track_id']
                x1, y1, x2, y2 = t['bbox']
                cx, cy = (x1+x2)/2, (y1+y2)/2
                h = self.track_history[tid]
                h.append((cx, cy))
                if len(h) > 30:
                    h.pop(0)
                if t['confidence'] > 0:
                    self.track_stats['trash'].add(tid)

        return self._parse_coco_tracked(coco_res, update_trails) + stable_trash

    def _parse_coco_tracked(self, results, update_trails=True):
        tracks = []
        for result in results:
            if result.boxes.id is None:
                continue
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                tid  = int(box.id[0])
                if cls not in self.COCO_TO_LABEL:
                    continue
                label = self.COCO_TO_LABEL[cls]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                self.track_stats[label].add(tid)
                if update_trails:
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    h = self.track_history[tid]
                    h.append((cx, cy))
                    if len(h) > 30:
                        h.pop(0)
                tracks.append({'bbox':(x1,y1,x2,y2), 'label':label,
                               'confidence':conf, 'track_id':tid})
        return tracks

    def _parse_trash_tracked(self, results, update_trails=True):
        tracks = []
        for result in results:
            if result.boxes.id is None:
                continue
            for box in result.boxes:
                conf    = float(box.conf[0])
                raw_tid = int(box.id[0])
                tid     = raw_tid + TRASH_ID_OFFSET
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                self.track_stats['trash'].add(tid)
                if update_trails:
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    h = self.track_history[tid]
                    h.append((cx, cy))
                    if len(h) > 30:
                        h.pop(0)
                tracks.append({'bbox':(x1,y1,x2,y2), 'label':'trash',
                               'confidence':conf, 'track_id':tid})
        return tracks

    # Dump state machine
  
    def _update_dump_state(self, tracks, raw_frame: np.ndarray):
        actor_tracks = [t for t in tracks
                        if t['label'] in ('person','vehicle')
                        and t['track_id'] is not None]
        trash_list   = [t for t in tracks
                        if t['label'] == 'trash'
                        and t['track_id'] is not None
                        and t['confidence'] > 0]

        seen_tids = {t['track_id'] for t in trash_list}

        # Baseline phase — catalogue pre-existing trash
        if self.frame_count <= self.baseline_frames:
            for tt in trash_list:
                self.baseline_ids.add(tt['track_id'])
                self.dump_tracks[tt['track_id']] = TrashDumpTrack(
                    tid=tt['track_id'], state=DumpState.BASELINE,
                    centre=self._centre(tt['bbox']))
            return

        # Create / update TrashDumpTrack objects
        for tt in trash_list:
            tid = tt['track_id']
            if tid in self.baseline_ids:
                continue
            cx, cy = self._centre(tt['bbox'])
            if tid not in self.dump_tracks:
                self.dump_tracks[tid] = TrashDumpTrack(
                    tid=tid, state=DumpState.NEW,
                    centre=(cx,cy), prev_centre=(cx,cy))
            else:
                tr = self.dump_tracks[tid]
                tr.prev_centre = tr.centre
                tr.centre      = (cx, cy)
                tr.velocity    = self._dist(tr.centre, tr.prev_centre)

        # State transitions
        for tid, tr in list(self.dump_tracks.items()):
            if tr.state in (DumpState.BASELINE, DumpState.DISMISSED, DumpState.DUMPED):
                continue
            if tid not in seen_tids:
                if tr.state == DumpState.ASSOCIATED:
                    tr.state = DumpState.DISMISSED
                continue

            if tr.state == DumpState.NEW:
                self._try_associate(tr, actor_tracks, raw_frame)
            elif tr.state == DumpState.ASSOCIATED:
                self._check_depart(tr, actor_tracks, raw_frame)

    def _try_associate(self, tr: TrashDumpTrack, actor_tracks, raw_frame):
        closest, closest_d = None, float('inf')
        for actor in actor_tracks:
            d = self._dist(tr.centre, self._centre(actor['bbox']))
            if d < closest_d:
                closest_d, closest = d, actor

        if closest is not None and closest_d <= self.proximity_px:
            tr.assoc_frames     += 1
            tr.assoc_actor_id    = closest['track_id']
            tr.assoc_actor_label = closest['label']
            tr.actor_last_centre = self._centre(closest['bbox'])
            self._refresh_best_crop(tr, closest, raw_frame)
            if tr.assoc_frames >= self.assoc_min_frames:
                tr.state = DumpState.ASSOCIATED
                print(f"[dump] Trash {tr.tid - TRASH_ID_OFFSET} ASSOCIATED with "
                      f"{tr.assoc_actor_label} ID:{tr.assoc_actor_id}")
        else:
            tr.assoc_frames = max(0, tr.assoc_frames - 1)

    def _check_depart(self, tr: TrashDumpTrack, actor_tracks, raw_frame):
        actor = next((a for a in actor_tracks
                      if a['track_id'] == tr.assoc_actor_id), None)

        if actor is not None:
            actor_centre = self._centre(actor['bbox'])
            actor_dist   = self._dist(tr.centre, actor_centre)
            tr.actor_last_centre = actor_centre
            self._refresh_best_crop(tr, actor, raw_frame)
            if actor_dist < self.depart_px:
                tr.stationary_frames = 0
                return
        else:
            actor_dist = self._dist(tr.centre, tr.actor_last_centre)

        if tr.velocity <= self.stationary_vel:
            tr.stationary_frames += 1
        else:
            tr.stationary_frames = max(0, tr.stationary_frames - 1)

        if tr.stationary_frames >= self.stationary_frames:
            tr.state = DumpState.DUMPED
            self._on_dump_confirmed(tr)

    def _refresh_best_crop(self, tr: TrashDumpTrack, actor: dict, raw_frame: np.ndarray):
        if actor['confidence'] > tr.best_actor_conf:
            tr.best_actor_conf  = actor['confidence']
            tr.best_actor_bbox  = actor['bbox']
            tr.best_actor_frame = raw_frame.copy()
          
    # Dump confirmed — save clip + crop
   
    def _on_dump_confirmed(self, tr: TrashDumpTrack):
        self.confirmed_dumps += 1
        ts       = time.strftime('%Y%m%d_%H%M%S')
        event_id = f"dump_{ts}_t{tr.tid - TRASH_ID_OFFSET}"

        print(f"\n{'!'*55}")
        print(f"  DUMP CONFIRMED  event: {event_id}")
        print(f"  Trash ID : {tr.tid - TRASH_ID_OFFSET}")
        print(f"  Actor    : {tr.assoc_actor_label}  ID:{tr.assoc_actor_id}")
        print(f"{'!'*55}\n")

        self._alert_frames = self.fps * 5
        self._save_actor_crop(tr, event_id)

        # Start post-event recording
        self._post_active    = True
        self._post_remaining = max(1, self.fps * self.post_seconds)
        self._post_frames    = self.ring_buffer.snapshot()  # pre-event frames from RAM
        self._post_meta      = {'event_id': event_id}

    def _tick_post_recording(self, annotated_frame: np.ndarray):
        if not self._post_active:
            return
        self._post_frames.append(annotated_frame.copy())
        self._post_remaining -= 1
        if self._post_remaining <= 0:
            self._post_active = False
            self._save_clip(self._post_frames, self._post_meta['event_id'])
            self._post_frames = []

    def _save_clip(self, frames: list, event_id: str):
        if not frames:
            return
        h, w   = frames[0].shape[:2]
        path   = str(self.output_dir / f"{event_id}_clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"[dump] Clip saved : {path}")

    def _save_actor_crop(self, tr: TrashDumpTrack, event_id: str):
        if tr.best_actor_frame is None or tr.best_actor_bbox is None:
            return
        frame = tr.best_actor_frame
        h, w  = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_bbox(tr.best_actor_bbox, h, w)
        x1p = max(0, x1 - CROP_PAD)
        y1p = max(0, y1 - CROP_PAD)
        x2p = min(w, x2 + CROP_PAD)
        y2p = min(h, y2 + CROP_PAD)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return
        label     = tr.assoc_actor_label
        crop_path = str(self.output_dir / f"{event_id}_{label}_crop.jpg")
        cv2.imwrite(crop_path, crop)
        print(f"[dump] Crop saved : {crop_path}")

    # Drawing
  
    def draw_tracks(self, image, tracks, draw_trails=True):
        out = image.copy()
        for t in tracks:
            x1, y1, x2, y2 = t['bbox']
            label = t['label']
            conf  = t['confidence']
            tid   = t['track_id']
            is_ghost = (label == 'trash' and conf == 0.0)

            if label == 'trash' and tid in self.dump_tracks:
                color = self.STATE_BORDER.get(
                    self.dump_tracks[tid].state, self._color_for(label, tid))
            else:
                color = self._color_for(label, tid)

            if is_ghost:
                for x in range(x1, x2, 10):
                    cv2.line(out, (x, y1), (min(x+5,x2), y1), color, 1)
                    cv2.line(out, (x, y2), (min(x+5,x2), y2), color, 1)
                for y in range(y1, y2, 10):
                    cv2.line(out, (x1, y), (x1, min(y+5,y2)), color, 1)
                    cv2.line(out, (x2, y), (x2, min(y+5,y2)), color, 1)
            else:
                cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)

            if not is_ghost:
                if label == 'trash' and tid is not None:
                    display_id = tid - TRASH_ID_OFFSET
                    state_name = self.dump_tracks[tid].state.name \
                        if tid in self.dump_tracks else ''
                    text = f"trash ID:{display_id} {conf:.2f} [{state_name}]"
                else:
                    id_str = f" ID:{tid}" if tid is not None else ""
                    text   = f"{label}{id_str} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1-th-bl-4), (x1+tw+4, y1), color, -1)
                cv2.putText(out, text, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            if draw_trails and tid is not None and len(self.track_history[tid]) > 1:
                pts = self.track_history[tid]
                for i in range(1, len(pts)):
                    thick = max(1, int(np.sqrt(i+1)))
                    cv2.line(out,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             color, thick)
        return out

    def _draw_dump_alert(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-52), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame,
                    f"  *** DUMP DETECTED  (total: {self.confirmed_dumps}) ***",
                    (10, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def _draw_state_legend(self, frame):
        items = [('NEW',(200,200,0)),('ASSOCIATED',(0,165,255)),
                 ('DUMPED',(0,0,255)),('BASELINE',(120,120,120))]
        x, y = frame.shape[1] - 165, 10
        for name, color in items:
            cv2.rectangle(frame, (x,y), (x+14,y+14), color, -1)
            cv2.putText(frame, name, (x+18,y+11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
            y += 20

    @staticmethod
    def _draw_overlay(frame, lines):
        y = 25
        for text in lines:
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (5,y-th-4), (12+tw,y+bl+2), (0,0,0), -1)
            cv2.putText(frame, text, (8,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            y += th + bl + 8

    # Public run methods
  
    def detect_video(self, video_path, output_path=None, display=True,
                     draw_trails=True):
        print(f"\n[run] Video: {video_path}")
        self._reset_state()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        self.fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ring_buffer = RingBuffer(self.fps, self.buffer_seconds)

        print(f"[run] {width}x{height} @ {self.fps} fps | {total} frames")
        print(f"[run] Baseline: first {self.baseline_frames} frames")
        print(f"[run] Ring buffer: {self.buffer_seconds}s = "
              f"{self.fps * self.buffer_seconds} frames in RAM")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
            print(f"[run] Output -> {output_path}")

        current_trails = draw_trails
        if display:
            print("[run] Q=quit  S=screenshot  T=trails")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.ring_buffer.push(frame)      # push RAW frame

            tracks    = self._run_tracking(frame, update_trails=current_trails)
            self._update_dump_state(tracks, frame)
            annotated = self.draw_tracks(frame, tracks, draw_trails=current_trails)
            self._draw_state_legend(annotated)

            n_p = sum(1 for t in tracks if t['label'] == 'person')
            n_v = sum(1 for t in tracks if t['label'] == 'vehicle')
            n_t = sum(1 for t in tracks if t['label'] == 'trash'
                      and t['confidence'] > 0)
            phase = "BASELINE" if self.frame_count <= self.baseline_frames else "ACTIVE"

            self._draw_overlay(annotated, [
                f"Phase:{phase}  Frame:{self.frame_count}/{total}",
                f"Now  P:{n_p}  V:{n_v}  Trash:{n_t}",
                f"Total P:{len(self.track_stats['person'])}  "
                f"V:{len(self.track_stats['vehicle'])}  "
                f"Trash:{len(self.track_stats['trash'])}",
                f"Dumps confirmed: {self.confirmed_dumps}",
            ])

            if self._alert_frames > 0:
                self._draw_dump_alert(annotated)
                self._alert_frames -= 1

            # post-event clip collects annotated frames
            self._tick_post_recording(annotated)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow('Dump Detection', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[run] Stopped.")
                    break
                elif key == ord('s'):
                    sp = f'screenshot_{self.frame_count}.jpg'
                    cv2.imwrite(sp, annotated)
                    print(f"[run] Screenshot: {sp}")
                elif key == ord('t'):
                    current_trails = not current_trails

            if self.frame_count % 50 == 0:
                pct = self.frame_count / total * 100 if total else 0
                print(f"[run] {pct:.1f}%  dumps:{self.confirmed_dumps}")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        self._print_summary("Video")

    def detect_webcam(self, camera_id=0, draw_trails=True):
        print(f"\n[run] Webcam {camera_id}")
        print("[run] Q=quit  S=screenshot  T=trails  R=reset")
        self._reset_state()

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        self.ring_buffer = RingBuffer(self.fps, self.buffer_seconds)

        current_trails = draw_trails
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.ring_buffer.push(frame)

            tracks    = self._run_tracking(frame, update_trails=current_trails)
            self._update_dump_state(tracks, frame)
            annotated = self.draw_tracks(frame, tracks, draw_trails=current_trails)
            self._draw_state_legend(annotated)

            n_p = sum(1 for t in tracks if t['label'] == 'person')
            n_v = sum(1 for t in tracks if t['label'] == 'vehicle')
            n_t = sum(1 for t in tracks if t['label'] == 'trash'
                      and t['confidence'] > 0)
            phase = "BASELINE" if self.frame_count <= self.baseline_frames else "ACTIVE"

            self._draw_overlay(annotated, [
                f"Phase:{phase}  Frame:{self.frame_count}",
                f"Now  P:{n_p}  V:{n_v}  Trash:{n_t}",
                f"Dumps confirmed: {self.confirmed_dumps}",
                "Q:Quit  S:Screenshot  T:Trails  R:Reset",
            ])

            if self._alert_frames > 0:
                self._draw_dump_alert(annotated)
                self._alert_frames -= 1

            self._tick_post_recording(annotated)

            cv2.imshow('Dump Detection', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sp = f'webcam_{self.frame_count}.jpg'
                cv2.imwrite(sp, annotated)
                print(f"[run] Screenshot: {sp}")
            elif key == ord('t'):
                current_trails = not current_trails
            elif key == ord('r'):
                self._reset_state()
                self.ring_buffer = RingBuffer(self.fps, self.buffer_seconds)
                self.coco_model.track(frame, persist=False,
                                      classes=self.TARGET_COCO_IDS,
                                      conf=self.conf_threshold, verbose=False)
                self.trash_model.track(frame, persist=False,
                                       conf=self.trash_conf, verbose=False)
                print("[run] Reset.")

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary("Webcam")

    def detect_image(self, image_path, output_path=None):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read: {image_path}")
        coco_res  = self.coco_model(image, classes=self.TARGET_COCO_IDS,
                                    conf=self.conf_threshold, verbose=False)
        trash_res = self.trash_model(image, conf=self.trash_conf, verbose=False)
        detections = []
        for result in coco_res:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls not in self.COCO_TO_LABEL:
                    continue
                x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                detections.append({'bbox':(x1,y1,x2,y2),
                                   'label':self.COCO_TO_LABEL[cls],
                                   'confidence':float(box.conf[0]),
                                   'track_id':None})
        for result in trash_res:
            for box in result.boxes:
                x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                detections.append({'bbox':(x1,y1,x2,y2), 'label':'trash',
                                   'confidence':float(box.conf[0]),
                                   'track_id':None})
        annotated = self.draw_tracks(image, detections, draw_trails=False)
        if output_path:
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved: {output_path}")
        return annotated

    def _print_summary(self, mode):
        print(f"\n{'='*55}")
        print(f" {mode} Summary")
        print(f"{'='*55}")
        print(f" Frames processed      : {self.frame_count}")
        print(f" Persons tracked       : {len(self.track_stats['person'])}")
        print(f" Vehicles tracked      : {len(self.track_stats['vehicle'])}")
        print(f" Trash tracked         : {len(self.track_stats['trash'])}")
        print(f" Baseline trash ignored: {len(self.baseline_ids)}")
        print(f" Dumps confirmed       : {self.confirmed_dumps}")
        if self.confirmed_dumps:
            print(f" Evidence saved in     : {self.output_dir}/")
        print(f"{'='*55}")
# CLI

def main():
    parser = argparse.ArgumentParser(
        description='Illegal Dump Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python person_vehicle_trash_tracker.py --mode video  --input road.mp4 --trash-model best.pt --no-display
  python person_vehicle_trash_tracker.py --mode webcam --trash-model best.pt
  python person_vehicle_trash_tracker.py --mode video  --input road.mp4 --trash-model best.pt --proximity 120 --assoc-frames 10

Saved per event in ./dump_events/:
  dump_<ts>_t<id>_clip.mp4          10-second evidence clip
  dump_<ts>_t<id>_vehicle_crop.jpg  vehicle photo crop
  dump_<ts>_t<id>_person_crop.jpg   person photo crop
        """)

    parser.add_argument('--mode',         required=True, choices=['video','webcam','image'])
    parser.add_argument('--input',        help='Input file path')
    parser.add_argument('--output',       help='Annotated output video path')
    parser.add_argument('--model',        default='yolov8n.pt')
    parser.add_argument('--trash-model',  default='best.pt')
    parser.add_argument('--conf',         type=float, default=0.5)
    parser.add_argument('--trash-conf',   type=float, default=0.25)
    parser.add_argument('--tracker',      default='bytetrack.yaml',
                        choices=['bytetrack.yaml','botsort.yaml'])
    parser.add_argument('--camera',       type=int,   default=0)
    parser.add_argument('--output-dir',   default='dump_events')
    # Kalman
    parser.add_argument('--kalman-missed',type=int,   default=10)
    parser.add_argument('--kalman-iou',   type=float, default=0.08)
    # Dump detection
    parser.add_argument('--proximity',    type=float, default=PROXIMITY_PX,
                        help=f'Proximity px for association (default:{PROXIMITY_PX})')
    parser.add_argument('--assoc-frames', type=int,   default=ASSOC_MIN_FRAMES,
                        help=f'Frames near actor to confirm association (default:{ASSOC_MIN_FRAMES})')
    parser.add_argument('--depart',       type=float, default=DEPART_PX,
                        help=f'Actor depart distance px (default:{DEPART_PX})')
    parser.add_argument('--stat-vel',     type=float, default=STATIONARY_VEL,
                        help=f'Max trash velocity to be stationary (default:{STATIONARY_VEL})')
    parser.add_argument('--stat-frames',  type=int,   default=STATIONARY_FRAMES,
                        help=f'Stationary frames to confirm dump (default:{STATIONARY_FRAMES})')
    parser.add_argument('--baseline',     type=int,   default=BASELINE_FRAMES,
                        help=f'Startup baseline frames (default:{BASELINE_FRAMES})')
    parser.add_argument('--buffer-sec',   type=int,   default=BUFFER_SECONDS,
                        help=f'Ring buffer seconds (default:{BUFFER_SECONDS})')
    parser.add_argument('--post-sec',     type=int,   default=POST_SECONDS,
                        help=f'Post-event recording seconds (default:{POST_SECONDS})')
    # flags
    parser.add_argument('--no-trails',    action='store_true')
    parser.add_argument('--no-display',   action='store_true')
    parser.add_argument('--no-save',      action='store_true')

    args = parser.parse_args()
    if args.mode in ('video','image') and not args.input:
        parser.error(f'--input required for {args.mode} mode')

    tracker = PersonVehicleTrashTracker(
        coco_model_path   = args.model,
        trash_model_path  = args.trash_model,
        conf_threshold    = args.conf,
        trash_conf        = args.trash_conf,
        tracker           = args.tracker,
        kalman_max_missed = args.kalman_missed,
        kalman_iou_thresh = args.kalman_iou,
        output_dir        = args.output_dir,
        proximity_px      = args.proximity,
        assoc_min_frames  = args.assoc_frames,
        depart_px         = args.depart,
        stationary_vel    = args.stat_vel,
        stationary_frames = args.stat_frames,
        baseline_frames   = args.baseline,
        buffer_seconds    = args.buffer_sec,
        post_seconds      = args.post_sec,
    )
    draw_trails = not args.no_trails

    if args.mode == 'video':
        out = None if args.no_save else (
            args.output or f"tracked_{Path(args.input).name}")
        tracker.detect_video(args.input, out,
                             display=not args.no_display,
                             draw_trails=draw_trails)
    elif args.mode == 'webcam':
        tracker.detect_webcam(camera_id=args.camera, draw_trails=draw_trails)
    else:
        out = args.output or f"detected_{Path(args.input).name}"
        tracker.detect_image(args.input, out)


if __name__ == '__main__':
    main()
