import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

WINDOW = "YOLO Detection"
cv2.namedWindow(WINDOW)
cv2.createTrackbar("Min confidence %", WINDOW, 50, 100, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read threshold from slider (0-100) and convert to 0.0-1.0
    threshold = cv2.getTrackbarPos("Min confidence %", WINDOW) / 100.0

    results = model(frame)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue  # skip boxes below threshold

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )

    # Show current threshold on frame
    cv2.putText(
        frame,
        f"Threshold: {threshold:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 200, 255), 2,
    )

    cv2.imshow(WINDOW, frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
