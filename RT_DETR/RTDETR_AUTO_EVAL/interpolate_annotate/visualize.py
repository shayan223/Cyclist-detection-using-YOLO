import cv2
from pathlib import Path
from natsort import natsorted

# run = max(Path(".").glob("run_"), key=lambda p: int(p.name[4:]))
run = Path("./run_002")
images = run / "images"
labels = run / "labels"
names  = ["cyclist", "pedestrian"]
colors = [(0,255,0), (255,0,255)]

for img_path in natsorted(images.glob("*.jpg")):
    frame = cv2.imread(str(img_path))
    h, w  = frame.shape[:2]
    lbl   = labels / img_path.with_suffix(".txt").name

    if lbl.exists():
        for line in lbl.read_text().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            cls = int(cls)
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), colors[cls], 2)
            cv2.putText(frame, names[cls], (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)

    cv2.imshow("YOLO Viewer", frame)
    key = cv2.waitKey(16) & 0xFF
    if key == ord("q"): break
    if key == ord(" "): cv2.waitKey(0)

cv2.destroyAllWindows()