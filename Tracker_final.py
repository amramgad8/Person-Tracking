import cv2
import numpy as np
import os
import urllib.request
import sys
import platform
from datetime import datetime, timedelta

# ---------------------------------------------------- #
#  Simple Sound Notification (Cross‑platform)
# ---------------------------------------------------- #
if platform.system() == "Windows":
    import winsound

    def play_beep():
        winsound.Beep(1000, 800)          # Tones with 1 kHz frequency for 250ms
else:
    def play_beep():
        if os.system("command -v paplay > /dev/null 2>&1") == 0:
            os.system("paplay /usr/share/sounds/freedesktop/stereo/message.oga &")
        elif os.system("command -v aplay > /dev/null 2>&1") == 0:
            os.system("aplay /usr/share/sounds/alsa/Front_Center.wav &")
        elif os.system("command -v afplay > /dev/null 2>&1") == 0:
            os.system("afplay /System/Library/Sounds/Pop.aiff &")
        else:
            sys.stdout.write("\a")        # Terminal beep


# ---------------------------------------------------- #
#  Download YOLO files if missing
# ---------------------------------------------------- #
def download_if_missing():
    files = {
        "yolov4-tiny.weights":
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.cfg":
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "coco.names":
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    for fname, url in files.items():
        if not os.path.exists(fname):
            print(f"[+] Downloading {fname} …")
            urllib.request.urlretrieve(url, fname)
            print(f"[✓] {fname} downloaded")

download_if_missing()

# ---------------------------------------------------- #
#  Time Formatting Utility
# ---------------------------------------------------- #
def pretty_time(td: timedelta) -> str:
    tot = int(td.total_seconds())
    h, rem = divmod(tot, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ---------------------------------------------------- #
#  Simple ID Tracker
# ---------------------------------------------------- #
class SimpleIDTracker:
    def __init__(self, max_disappear=15, dist_thresh=100):
        self.next_id = 1
        self.objects = {}   # id → {bbox, centroid, missed}
        self.records = {}   # id → {entry, exit, duration}
        self.max_disappear, self.dist_thresh = max_disappear, dist_thresh

        self.last_msg, self.msg_time = None, None
        self.MSG_DURATION = timedelta(seconds=2)

    def _register(self, box):
        cx, cy = box[0]+box[2]//2, box[1]+box[3]//2
        self.objects[self.next_id] = {'bbox': box, 'centroid': (cx, cy), 'missed': 0}
        self.records[self.next_id] = {'entry': datetime.now(), 'exit': None, 'duration': None}
        self._set_msg(f"Person {self.next_id} entered")
        self.next_id += 1
        
        # Increase total_detected and total_present when a new person enters
        global total_detected, total_present
        total_detected += 1
        total_present += 1

    def _deregister(self, pid):
        rec = self.records[pid]
        rec['exit'] = datetime.now()
        rec['duration'] = rec['exit'] - rec['entry']
        self._set_msg(f"Person {pid} exited")
        self.objects.pop(pid)

    def _set_msg(self, text: str):
        play_beep()  # Play sound on entry/exit
        self.last_msg = text
        self.msg_time = datetime.now()

    def get_msg(self):
        if self.last_msg and datetime.now() - self.msg_time < self.MSG_DURATION:
            return self.last_msg
        return None

    def update(self, detections):
        if not detections:
            for pid in list(self.objects):
                self.objects[pid]['missed'] += 1
                if self.objects[pid]['missed'] > self.max_disappear:
                    self._deregister(pid)
            return self.objects

        if not self.objects:
            for box in detections:
                self._register(box)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centers = np.array([self.objects[i]['centroid'] for i in obj_ids])
        new_centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h in detections])

        D = np.linalg.norm(obj_centers[:, None] - new_centers[None, :], axis=2)
        rows = D.min(1).argsort()
        cols = D.argmin(1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols or D[r, c] > self.dist_thresh:
                continue
            pid = obj_ids[r]
            self.objects[pid] = {'bbox': detections[c], 'centroid': new_centers[c], 'missed': 0}
            used_rows.add(r); used_cols.add(c)

        for c, box in enumerate(detections):
            if c not in used_cols:
                self._register(box)

        for r, pid in enumerate(obj_ids):
            if r not in used_rows:
                self.objects[pid]['missed'] += 1
                if self.objects[pid]['missed'] > self.max_disappear:
                    self._deregister(pid)

        return self.objects

# ---------------------------------------------------- #
#  YOLO‑tiny Person Detection
# ---------------------------------------------------- #
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_people(frame, conf=0.8, nms=0.6):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confid = [], []
    for out in outs:
        for det in out:
            s = det[5:]; cid = np.argmax(s); c = s[cid]
            if cid == 0 and c > conf:  # Class 0 is 'person'
                cx, cy, bw, bh = (det[:4] * np.array([w, h, w, h])).astype(int)
                boxes.append([int(cx-bw/2), int(cy-bh/2), int(bw), int(bh)])
                confid.append(float(c))
    idxs = cv2.dnn.NMSBoxes(boxes, confid, conf, nms)
    return [boxes[i] for i in idxs.flatten()] if len(idxs) else []

# ---------------------------------------------------- #
#  Main Loop with Debugging
# ---------------------------------------------------- #
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = SimpleIDTracker()
    COLORS = np.random.randint(0, 255, (500, 3), dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_id, DETECT_EVERY = 0, 6  # Detect every 2 frames

    global total_detected, total_present
    total_detected = 0
    total_present = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people every N frames
            if frame_id % DETECT_EVERY == 0:
                dets = detect_people(frame)
                objs = tracker.update(dets)
            else:
                objs = tracker.objects
            frame_id += 1

            # Draw boxes and time
            for pid, info in objs.items():
                x, y, w, h = info['bbox']
                color = COLORS[pid % len(COLORS)].tolist()
                elapsed = datetime.now() - tracker.records[pid]['entry']
                label = f"ID {pid}  {pretty_time(elapsed)}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 7), font, 0.55, color, 2)

            # Counters
            cv2.putText(frame, f"Current visitors: {len(objs)}", (10, 25), font, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total visitors today: {len(tracker.records)}", (10, 55), font, 0.8, (0, 255, 0), 2)

            # Notification message
            msg = tracker.get_msg()
            if msg:
                (w_txt, h_txt), _ = cv2.getTextSize(msg, font, 1.0, 3)
                cv2.rectangle(frame, (5, 80), (5 + w_txt + 10, 80 + h_txt + 10), (0, 0, 0), -1)
                cv2.putText(frame, msg, (10, 80 + h_txt), font, 1.0, (0, 255, 255), 3)

            cv2.imshow("Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Output statistics when the program is closed
        print(f"\n[INFO] Total persons detected during session: {total_detected}")
        print(f"[INFO] Total persons present during session: {total_present}")

        cap.release()
        cv2.destroyAllWindows()
