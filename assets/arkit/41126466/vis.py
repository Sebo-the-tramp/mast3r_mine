import time
import random

import viser
import viser.transforms as tf
import numpy as np
import imageio.v3 as iio

from scipy.spatial.transform import Rotation as R

server = viser.ViserServer()
# server.scene.world_axes.visible = True

# Load and display the content of the .npy file from ./intrinsics
intrinsics_path = './intrinsics.npy'
intrinsics_data = np.load(intrinsics_path)
print(len(intrinsics_data))
print("Intrinsics Data:")
print(intrinsics_data)

# Load and display the content of the .npy file from ./trajectories
trajectories_path = './trajectories.npy'
trajectories_data = np.load(trajectories_path)
print(len(trajectories_data))
print("Trajectories Data:")
print(trajectories_data)

# Load and display the content of the .npy file from ./pairs
pairs_path = './pairs.npy'
pairs_data = np.load(pairs_path)
print(len(pairs_data))
print("Pairs Data:")
print(pairs_data)

# Load and display the content of the .npy file from ./images
images_path = './images.npy'
images_data = np.load(images_path)
print(len(images_data))
print("Images Data:")
print(images_data)

# Convert images_data to a list if it's a NumPy array
images_data = images_data.tolist()

# Function to find the index of an image
def find_image_index(image_name, images_list):
    try:
        return images_list.index(image_name)
    except ValueError:
        print(f"Image {image_name} not found in images_data.")
        return None

imgs = ["41126466_5184.748.png", "41126466_5194.144.png"]
img_idxs = [find_image_index(img, images_data) for img in imgs]

# for img_idx in range(6):
for img_idx in img_idxs:
    img_name = images_data[img_idx]
    print(f"Index of {img_name}: {img_idx}")

    # Retrieve corresponding trajectory and intrinsic data
    ext = trajectories_data[img_idx]
    int_ = intrinsics_data[img_idx]

    downsample_factor = 1

    # Common image dimensions
    H, W = 384, 512

    # Frustum for image1
    fy = int_[3]
    image1 = iio.imread(f"./vga_wide/{img_name.replace('.png', '.jpg')}")
    image1 = image1[::downsample_factor, ::downsample_factor]

    frustum = server.scene.add_camera_frustum(
        f"/test/frame_{img_idx}/frustum",
        fov=2 * np.arctan2(H / 2, fy),
        aspect=W / H,
        scale=0.15,
        image=image1,
        wxyz=R.from_matrix(ext[:3, :3]).as_quat(scalar_first=True),
        position=ext[:3, 3],
    )

# Keep the server running
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Shutting down the server.")
