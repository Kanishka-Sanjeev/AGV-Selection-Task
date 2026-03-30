import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz

I1 = io.imread('Task 6/data/im1.png').astype(float)
I2 = io.imread('Task 6/data/im2.png').astype(float)
data = np.load('Task 6/data/some_corresp.npz')

I1 = I1 / 255.0
I2 = I2 / 255.0

pts1 = data['pts1']
pts2 = data['pts2']

# 2. Run eight_point to compute F

M = max(I1.shape)
F = sub.eight_point(pts1, pts2, M)

# 3. Load points in image 1 from data/temple_coords.npz

temple = np.load('Task 6/data/temple_coords.npz')
pts1_temple = temple['pts1']

# 4. Run epipolar_correspondences to get points in image 2

pts2_temple = sub.epipolar_correspondences(I1, I2, F, pts1_temple)
# hlp.displayEpipolarF(I1, I2, F)

# 5. Compute the camera projection matrix P1

intrinsics = np.load('Task 6/data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))

# 6. Use camera2 to get 4 camera projection matrices P2

E = sub.essential_matrix(F, K1, K2)

print("E:", E)
print("Type:", type(E))
print("Shape:", np.shape(E))

M2s = hlp.camera2(E)

# 7. Run triangulate using the projection matrices

pts3d_all = []
P2_all = []
M2_all = []

for i in range(4):
    M2 = M2s[:, :, i]
    P2 = K2 @ M2
    pts3d = sub.triangulate(P1, pts1_temple, P2, pts2_temple)

    pts3d_all.append(pts3d)
    P2_all.append(P2)
    M2_all.append(M2)

# 8. Figure out the correct P2

best_pts3d = None
best_M2 = None

for i in range(4):
    pts3d = pts3d_all[i]

    if np.all(pts3d[:, 2] > 0):
        best_pts3d = pts3d
        best_M2 = M2_all[i]
        break
    
# 9. Scatter plot the correct 3D points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], s=5)
ax.set_title("3D Reconstruction of Temple")
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

R1 = np.eye(3)
t1 = np.zeros((3,1))

R2 = best_M2[:, :3]
t2 = best_M2[:, 3].reshape(3,1)

np.savez('Task 6/data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)