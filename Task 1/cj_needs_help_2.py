import cv2
import numpy as np

def lucas_kanade_fast(old_gray, new_gray, points, win=5):

    half = win // 2

    old_gray = old_gray.astype(np.float32)
    new_gray = new_gray.astype(np.float32)

    # Precompute gradients
    Ix_full = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy_full = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=3)
    It_full = new_gray - old_gray

    h, w = old_gray.shape

    good_old = []
    good_new = []

    for pt in points:
        x, y = int(pt[0]), int(pt[1])

        # Skip borders
        if x-half < 0 or y-half < 0 or x+half >= w or y+half >= h:
            continue

        # Extract window
        Ix = Ix_full[y-half:y+half+1, x-half:x+half+1].flatten()
        Iy = Iy_full[y-half:y+half+1, x-half:x+half+1].flatten()
        It = It_full[y-half:y+half+1, x-half:x+half+1].flatten()

        # Skip flat regions
        if np.sum(Ix**2 + Iy**2) < 1e-4:
            continue

        A = np.vstack((Ix, Iy)).T
        b = -It

        # Solve least squares
        nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        u, v = nu

        good_old.append([x, y])
        good_new.append([x + u, y + v])

    return np.array(good_old, dtype=np.float32), np.array(good_new, dtype=np.float32)


# ---------------- MAIN ---------------- #

cap = cv2.VideoCapture("Task 1/OPTICAL_FLOW.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

start_time = 35   # start at 35 seconds
end_time = 40    # end at 40 seconds

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Jump to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

# Resize for speed
scale = 0.6
frame = cv2.resize(frame, None, fx=scale, fy=scale)

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initial feature detection
points = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=30,
    qualityLevel=0.2,
    minDistance=10
)

if points is not None:
    points = points.reshape(-1, 2)
else:
    points = np.array([], dtype=np.float32)

frame_id = start_frame

while True:

    ret, frame = cap.read()
    if not ret or frame_id > end_frame:
        break

    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply LK
    old_pts, new_pts = lucas_kanade_fast(old_gray, gray, points)

    filtered_points = []

    for old_pt, new_pt in zip(old_pts, new_pts):

        motion = np.linalg.norm(new_pt - old_pt)

        # motion threshold (eliminates noise)
        if motion > 0.02:
            filtered_points.append(new_pt)

            x, y = int(new_pt[0]), int(new_pt[1])
            x1, y1 = int(old_pt[0]), int(old_pt[1])
            x2, y2 = int(new_pt[0]), int(new_pt[1])

            cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.5)

    # Update points
    points = np.array(filtered_points, dtype=np.float32)

    # Re-detect if too few points
    if len(points) < 8:
        new_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=30,
            qualityLevel=0.2,
            minDistance=10
        )
        if new_pts is not None:
            points = new_pts.reshape(-1, 2)

    cv2.imshow("Manual Lucas-Kanade Tracking", frame)

    old_gray = gray.copy()

    if cv2.waitKey(30) == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
