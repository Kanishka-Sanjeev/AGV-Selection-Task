import numpy as np
import cv2
import submission as sub
import helper as hlp
from odometry_visualizer import TrajectoryVisualizer

# write your implementation here

# Load video frames
def load_video_frames(video_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frames = []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1

    cap.release()
    return frames

# Lucas-Kanade (from Task 1 Subtask 1)
def lucas_kanade_fast(old_gray, new_gray, points, win=5):

    half = win // 2

    old_gray = old_gray.astype(np.float32)
    new_gray = new_gray.astype(np.float32)

    Ix_full = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy_full = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=3)
    It_full = new_gray - old_gray

    h, w = old_gray.shape

    good_old = []
    good_new = []

    for pt in points:
        x, y = int(pt[0]), int(pt[1])

        if x-half < 0 or y-half < 0 or x+half >= w or y+half >= h:
            continue

        Ix = Ix_full[y-half:y+half+1, x-half:x+half+1].flatten()
        Iy = Iy_full[y-half:y+half+1, x-half:x+half+1].flatten()
        It = It_full[y-half:y+half+1, x-half:x+half+1].flatten()

        if np.sum(Ix**2 + Iy**2) < 1e-4:
            continue

        A = np.vstack((Ix, Iy)).T
        b = -It

        nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        u, v = nu

        good_old.append([x, y])
        good_new.append([x + u, y + v])

    return np.array(good_old, dtype=np.float32), np.array(good_new, dtype=np.float32)


# RANSAC PnP using estimate_pose
def pnp_ransac(X, pts2d, K, iterations=100, threshold=5):

    best_inliers = []
    best_P = None

    N = len(X)

    for _ in range(iterations):

        if N < 6:
            break

        idx = np.random.choice(N, 6, replace=False)

        X_sample = X[idx]
        pts_sample = pts2d[idx]

        try:
            P = sub.estimate_pose(pts_sample, X_sample)
        except:
            continue

        # Reprojection
        X_h = np.hstack((X, np.ones((N, 1))))
        proj = (P @ X_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]

        error = np.linalg.norm(proj - pts2d, axis=1)

        inliers = np.where(error < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_P = P

    if best_P is None or len(best_inliers) < 6:
        return None, None

    # refine using all inliers
    P_final = sub.estimate_pose(pts2d[best_inliers], X[best_inliers])

    return P_final, best_inliers

# MAIN VO PIPELINE
def run_vo(images, K):

    trajectory = []

    # Initial feature detection
    prev_img = images[0]
    pts_prev = cv2.goodFeaturesToTrack(prev_img, maxCorners=1000,
                                       qualityLevel=0.01, minDistance=7)

    pts_prev = pts_prev.reshape(-1, 2)

    # Track to second frame
    curr_img = images[1]
    pts_prev, pts_curr = lucas_kanade_fast(prev_img, curr_img, pts_prev)

    # Compute F, E
    M = max(prev_img.shape)
    F = sub.eight_point(pts_prev, pts_curr, M)
    E = sub.essential_matrix(F, K, K)

    # Camera pose from E
    M2s = hlp.camera2(E)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    best_X = None
    best_P2 = None

    for i in range(4):
        M2 = M2s[:, :, i]
        P2 = K @ M2

        X_candidate = sub.triangulate(P1, pts_prev, P2, pts_curr)

        if np.all(X_candidate[:, 2] > 0):
            best_X = X_candidate
            best_P2 = P2
            break

    if best_X is None:
        print("Initialization failed")
        return np.array(trajectory)

    X = best_X

    # initial pose
    Kc, Rc, tc = sub.estimate_params(best_P2)
    C = -Rc.T @ tc
    trajectory.append(C.flatten())

    prev_img = curr_img

    # MAIN LOOP
    for i in range(2, len(images)):

        curr_img = images[i]

        pts_prev_new, pts_curr_new = lucas_kanade_fast(prev_img, curr_img, pts_prev)

        # maintain correspondences
        min_len = min(len(pts_prev_new), len(X))

        pts_prev = pts_prev_new[:min_len]
        pts_curr = pts_curr_new[:min_len]
        X = X[:min_len]

        print("Tracked:", len(pts_curr), "3D:", len(X))

        # Re-detect if needed
        if len(pts_prev) < 50:
            print("Re-detecting")

            new_pts = cv2.goodFeaturesToTrack(curr_img, maxCorners=1000,
                                              qualityLevel=0.01, minDistance=7)

            if new_pts is not None:
                pts_prev = new_pts.reshape(-1, 2)

            prev_img = curr_img
            continue

        # PnP with RANSAC
        P, inliers = pnp_ransac(X, pts_curr, K)

        if P is None:
            print("PnP failed")
            prev_img = curr_img
            continue

        # Extract pose
        Kc, Rc, tc = sub.estimate_params(P)
        C = -Rc.T @ tc

        trajectory.append(C.flatten())

        prev_img = curr_img
        pts_prev = pts_curr

    return np.array(trajectory)

# RUN
if __name__ == "__main__":

    video_path = "Task 6/dataset_video.mp4"

    K = np.array([
        [517.3, 0, 318.6],
        [0, 516.5, 255.3],
        [0, 0, 1]
    ])

    images = load_video_frames(video_path, max_frames=50)

    trajectory = run_vo(images, K)

    print("Trajectory shape:", trajectory.shape)

    vis = TrajectoryVisualizer()

    for pose in trajectory:
        vis.add_pose(pose)
        vis.visualize()