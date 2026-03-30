"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp
import numpy as np
import scipy.linalg

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):

    # Normalize points
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M

    N = pts1.shape[0]

    # Construct matrix A
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]

        A[i] = [
            x2*x1, x2*y1, x2,
            y2*x1, y2*y1, y2,
            x1, y1, 1
        ]

    # Solve Af = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    F = hlp._singularize(F)

    # Refine F
    F = hlp.refineF(F, pts1_norm, pts2_norm)

    # Un-normalize
    T = np.array([
        [1/M, 0, 0],
        [0, 1/M, 0],
        [0, 0, 1]
    ])
    F = T.T @ F @ T

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):

    # Convert to grayscale
    if im1.ndim == 3:
        im1 = np.mean(im1, axis=2)
    if im2.ndim == 3:
        im2 = np.mean(im2, axis=2)

    # Normalize images (VERY important)
    im1 = im1 / 255.0
    im2 = im2 / 255.0

    pts2 = []
    window = 5
    search_range = 40  # restrict search

    H, W = im2.shape

    for (x1, y1) in pts1:
        x1, y1 = int(x1), int(y1)

        # Epipolar line
        l = F @ np.array([x1, y1, 1])
        a, b, c = l

        best_dist = float('inf')
        best_match = (x1, y1)

        # Skip if patch out of bounds
        if (y1 - window < 0 or y1 + window >= im1.shape[0] or
            x1 - window < 0 or x1 + window >= im1.shape[1]):
            pts2.append(best_match)
            continue

        patch1 = im1[y1-window:y1+window+1,
                     x1-window:x1+window+1]

        # Skip low-texture patches
        if np.std(patch1) < 0.01:
            pts2.append(best_match)
            continue

        # Search along epipolar line (restricted)
        for y2 in range(max(y1 - search_range, window),
                        min(y1 + search_range, H - window)):

            # Avoid divide by zero
            if abs(a) < 1e-6:
                continue

            x2 = int(-(b*y2 + c) / a)

            if x2 < window or x2 >= W - window:
                continue

            patch2 = im2[y2-window:y2+window+1,
                         x2-window:x2+window+1]

            # Skip bad patches
            if np.std(patch2) < 0.01:
                continue

            # Normalize patches SAFELY
            patch1_n = (patch1 - np.mean(patch1)) / np.std(patch1)
            patch2_n = (patch2 - np.mean(patch2)) / np.std(patch2)

            # SSD
            dist = np.sum((patch1_n - patch2_n) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_match = (x2, y2)

        pts2.append(best_match)

    return np.array(pts2)

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # Compute Essential Matrix
    E = K2.T @ F @ K1

    # Enforce correct singular values (1, 1, 0)
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt

    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Build matrix A
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        # Convert from homogeneous to 3D
        X = X / X[3]

        pts3d[i] = X[:3]

    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Camera centers
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2

    # New coordinate system
    # x-axis = baseline
    r1 = (c2 - c1).reshape(-1)
    r1 = r1 / np.linalg.norm(r1)

    # z-axis = average viewing direction
    z1 = R1[2, :]
    z2 = R2[2, :]
    r3 = (z1 + z2)
    r3 = r3 / np.linalg.norm(r3)

    # y-axis = orthogonal
    r2 = np.cross(r3, r1)
    r2 = r2 / np.linalg.norm(r2)

    # Recompute z-axis to ensure orthogonality
    r3 = np.cross(r1, r2)

    R = np.vstack((r1, r2, r3))

    # New rotations
    R1p = R
    R2p = R

    # New intrinsics (use K2 or average)
    K1p = K2
    K2p = K2

    # New translations
    t1p = -R1p @ c1
    t2p = -R2p @ c2

    # Rectification matrices
    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    H, W = im1.shape
    dispM = np.zeros((H, W))

    half = win_size // 2

    for y in range(half, H - half):
        for x in range(half, W - half):

            best_disp = 0
            best_dist = float('inf')

            # patch in left image
            patch1 = im1[y-half:y+half+1, x-half:x+half+1]

            # search along same row (rectified images!)
            for d in range(max_disp):
                x2 = x - d

                if x2 - half < 0:
                    continue

                patch2 = im2[y-half:y+half+1, x2-half:x2+half+1]

                if patch2.shape != patch1.shape:
                    continue

                # SSD
                dist = np.sum((patch1 - patch2) ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_disp = d

            dispM[y, x] = best_disp

    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
     # Compute camera centers
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2

    # Compute baseline (distance between cameras)
    B = np.linalg.norm(c1 - c2)

    # Get focal length (fx)
    f = K1[0, 0]

    # Compute depth
    depthM = np.zeros_like(dispM)

    # Avoid division by zero
    mask = dispM > 0

    depthM[mask] = (f * B) / dispM[mask]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    N = x.shape[0]

    # Convert X to homogeneous (Nx4)
    X_h = np.hstack((X, np.ones((N, 1))))

    A = []

    for i in range(N):
        Xi = X_h[i]
        u, v = x[i]

        # Two rows per point
        A.append(np.hstack([Xi, np.zeros(4), -u * Xi]))
        A.append(np.hstack([np.zeros(4), Xi, -v * Xi]))

    A = np.array(A)

    # Solve Ap = 0 using SVD
    _, _, Vt = np.linalg.svd(A)

    # Last row of Vt gives solution
    P = Vt[-1].reshape(3, 4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # Extract M
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Fix signs (make diagonal of K positive)
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Normalize K
    K = K / K[-1, -1]

    # Compute camera center
    _, _, Vt = np.linalg.svd(P)
    c = Vt[-1]
    c = c / c[-1]  # homogeneous → 3D

    # Compute translation
    t = -R @ c[:3]

    return K, R, t.reshape(3, 1)
