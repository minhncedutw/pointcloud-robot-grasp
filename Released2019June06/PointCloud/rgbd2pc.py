import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Read Redwood dataset")
    # color_raw = o3d.io.read_image("../../TestData/RGBD/color/00000.jpg")
    # depth_raw = o3d.io.read_image("../../TestData/RGBD/depth/00000.png")
    source = "E:/CLOUD/GDrive(t105ag8409)/Projects/ARLab/Point Cloud Robot Grasping/camera/orbbec_astra_s/recorded_data/standard_model_May22/"
    color_raw = o3d.io.read_image(source + "RGB/" + "00021.png")
    depth_raw = o3d.io.read_image(source + "D/" + "00021.png")
    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)

    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])