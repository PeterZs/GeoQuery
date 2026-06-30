import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _load_split_json(data_dir: str, n_views: int) -> Optional[Dict[str, List[int]]]:
    """加载 train_test_split_{n_views}.json 文件。"""
    split_file = os.path.join(data_dir, f"train_test_split_{n_views}.json")
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            return json.load(f)
    return None

# 在文件顶部的 _load_split_json 函数后添加一个新函数
def _load_mvs_depth(mvs_depth_dir: str, dataset_type: str, scene_name: str, 
                    n_views: int, image_name: str) -> Optional[np.ndarray]:
    """
    Load a precomputed reference depth map.
    
    Args:
        mvs_depth_dir: Root directory for depth maps.
        dataset_type: 数据集类型 ("mipnerf360" 或 "dl3dv")
        scene_name: 场景名称
        n_views: 视角数量
        image_name: 图像名称（不含路径，如 "_DSC8681.JPG"）
    
    Returns:
        depth: [H, W] numpy array 或 None
    """
    # 构建 depth 文件路径
    # 去掉图像扩展名
    image_base = os.path.splitext(image_name)[0]
    depth_path = os.path.join(
        mvs_depth_dir, 
        dataset_type, 
        scene_name, 
        f"{n_views}_views", 
        "depth", 
        f"{image_base}.npy"
    )
    
    # 尝试加载 confidence
    confidence_path = os.path.join(
        mvs_depth_dir, 
        dataset_type, 
        scene_name, 
        f"{n_views}_views", 
        "confidence", 
        f"{image_base}.npy"
    )
    
    if not os.path.exists(depth_path):
        # print(f"[MVS Depth] Warning: depth file not found: {depth_path}")
        return None
    
    depth = np.load(depth_path)
    
    # 生成 valid mask
    # if os.path.exists(confidence_path):
    #     confidence = np.load(confidence_path)
    #     # 假设 confidence > 0.5 为有效
    #     valid_mask = (confidence > 0.5) & (depth > 0)
    # else:
    #     valid_mask = depth > 0
    
    return depth



class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        n_views: int = 0,
        dataset_type: str = "auto", #"360, DL3DV"
        depth_dir: Optional[str] = None # depth dir
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.n_views = n_views
        self.dataset_type = dataset_type
        self.depth_dir = depth_dir # depth dir

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        # for d in [image_dir, colmap_image_dir]:
        #     if not os.path.exists(d):
        #         raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        
        # 优先使用已存在的下采样目录
        if os.path.exists(image_dir) and len(_get_rel_paths(image_dir)) > 0:
            # 下采样目录已存在，直接使用（可以是 JPG 或 PNG）
            print(f"[Parser] Using existing images from {image_dir}")
            # 如果 COLMAP 使用的也是这个目录，就不需要映射
            if not os.path.exists(colmap_image_dir) or colmap_image_dir == image_dir:
                colmap_image_dir = image_dir
        else:
            # 下采样目录不存在，需要从原始目录创建
            if not os.path.exists(colmap_image_dir):
                raise ValueError(
                    f"Image directory {colmap_image_dir} does not exist. "
                    f"Please provide at least one image directory."
                )
            # 检查是否需要 resize 和格式转换
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            if len(colmap_files) == 0:
                raise ValueError(f"Image folder {colmap_image_dir} is empty.")
            
            # 如果原始图像是 JPG，转换为 PNG 并 resize
            if os.path.splitext(colmap_files[0])[1].lower() == ".jpg":
                print(f"[Parser] Resizing JPG images to PNG in {image_dir}_png")
                image_dir = _resize_image_folder(
                    colmap_image_dir, image_dir + "_png", factor=factor
                )
            else:
                # 其他格式直接 resize
                print(f"[Parser] Resizing images to {image_dir}")
                image_dir = _resize_image_folder(
                    colmap_image_dir, image_dir, factor=factor
                )

        # 映射 COLMAP 图像名称到实际图像路径
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        
        if colmap_image_dir == image_dir:
            # 同一个目录，直接映射
            colmap_to_image = {f: f for f in colmap_files}
        else:
            # 不同目录，需要建立映射关系
            colmap_to_image = dict(zip(colmap_files, image_files))
        
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        
        all_indices = np.arange(len(self.image_names))
        self.train_indices, self.test_indices = self._split_train_test(all_indices, self.image_names)
    
    
    def get_normalize_scale(self) -> float:
        """
            get scale factor of scene when normalize is true.
        """
        if not self.normalize:
            return 1.0
        
        transform = self.transform
        R_scale = transform[:3, :3]
        scale_factor = np.linalg.norm(R_scale[:,0])
        
        return scale_factor
        
    
    
    
    def _split_train_test(
        self, all_indices: np.ndarray, image_names: List[str]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据数据集类型和参数进行训练/测试分割。
        
        对于 mipnerf360 数据集：从 train_test_split_{n_views}.json 读取
        对于 dl3dv 数据集：使用 llffhold 方式分割，然后均匀采样训练视角
        """
        dataset_type = self.dataset_type
        
        # 自动检测数据集类型
        if dataset_type == "auto":
            # 检查是否存在 train_test_split_{n_views}.json
            if self.n_views > 0:
                split_data = _load_split_json(self.data_dir, self.n_views)
                if split_data is not None:
                    dataset_type = "mipnerf360"
                else:
                    dataset_type = "dl3dv"
            else:
                dataset_type = "dl3dv"  # 默认使用 dl3dv 方式
        
        if dataset_type == "mipnerf360" and self.n_views > 0:
            return self._split_mipnerf360(all_indices, image_names)
        else:
            return self._split_dl3dv(all_indices)
        


    def _split_mipnerf360(
        self, all_indices: np.ndarray, image_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MipNeRF360 数据集的分割方式：从 train_test_split_{n_views}.json 读取。
        """
        split_data = _load_split_json(self.data_dir, self.n_views)
        
        if split_data is None:
            print(f"[Parser] Warning: train_test_split_{self.n_views}.json not found, "
                  f"falling back to dl3dv split method.")
            return self._split_dl3dv(all_indices)
        
        train_ids = split_data.get("train_ids", [])
        test_ids = split_data.get("test_ids", [])
        
        # 验证索引有效性
        max_idx = len(all_indices) - 1
        train_ids = [idx for idx in train_ids if idx <= max_idx]
        test_ids = [idx for idx in test_ids if idx <= max_idx]
        
        train_indices = np.array(train_ids, dtype=np.int32)
        test_indices = np.array(test_ids, dtype=np.int32)
        
        print(f"[Parser] MipNeRF360 split: loaded {len(train_indices)} train, "
              f"{len(test_indices)} test views from JSON.")
        
        return train_indices, test_indices
    
    def _split_dl3dv(self, all_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        DL3DV-Benchmark 数据集的分割方式：
        1. 使用 llffhold (test_every) 分割训练和测试集
        2. 如果指定了 n_views，则从训练集中均匀采样
        """
        llffhold = self.test_every
        
        # 首先按 llffhold 分割
        train_indices = np.array([idx for idx in all_indices if idx % llffhold != 0])
        test_indices = np.array([idx for idx in all_indices if idx % llffhold == 0])
        
        # 如果指定了 n_views，从训练集中均匀采样
        if self.n_views > 0 and len(train_indices) > self.n_views:
            idx_sub = np.linspace(0, len(train_indices) - 1, self.n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_indices = train_indices[idx_sub]
            
            assert len(train_indices) == self.n_views, \
                f"Expected {self.n_views} train views, got {len(train_indices)}"
            
            print(f"[Parser] DL3DV split: uniformly sampled {self.n_views} train views "
                  f"from {len(all_indices)} total views.")
        else:
            print(f"[Parser] DL3DV split: {len(train_indices)} train, "
                  f"{len(test_indices)} test views (test every={llffhold}).")
        
        return train_indices, test_indices
    
    def filter_points_sparse(self, train_indices: List[int]):
        """
        过滤点云，只保留在训练视角中可见的点。
        这对于稀疏视角重建很重要，可以减少点云数量并提高训练效率。
    
        Args:
            train_indices: 训练视角的索引列表
        """
        visible_points = set()
    
        for idx in train_indices:
            image_name = self.image_names[idx]
            if image_name in self.point_indices:
                point_indices = self.point_indices[image_name]
                visible_points.update(point_indices)
    
        keep_indices = list(visible_points)
    
        # 更新点云数据
        self.points = self.points[keep_indices]
        self.points_err = self.points_err[keep_indices]
        self.points_rgb = self.points_rgb[keep_indices]
    
        # 更新 point_indices 映射
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.point_indices = {
            k: np.array([old_to_new[idx] for idx in v if idx in old_to_new])
            for k, v in self.point_indices.items()
        }
    
        print(f"[Parser] Filtered points: {len(keep_indices)}/{len(visible_points)} "
          f"points visible in {len(train_indices)} training views")
        
    def verify_image_camera_alignment(self, sample_indices: Optional[List[int]] = None, 
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        验证图像和相机参数是否匹配对应。
    
            Args:
            sample_indices: 要验证的图像索引列表，如果为None则验证所有图像
            verbose: 是否打印详细信息
    
        Returns:
            包含验证结果的字典
        """
        if sample_indices is None:
            sample_indices = list(range(len(self.image_names)))
    
        results = {
            "total_images": len(sample_indices),
            "size_mismatches": [],
            "principal_point_issues": [],
            "projection_issues": [],
            "missing_files": [],
            "all_valid": True
        }
    
        for idx in sample_indices:
            image_path = self.image_paths[idx]
            image_name = self.image_names[idx]
            camera_id = self.camera_ids[idx]
            K = self.Ks_dict[camera_id]
            expected_size = self.imsize_dict[camera_id]  # (width, height)
        
            # 1. 检查文件是否存在
            if not os.path.exists(image_path):
                results["missing_files"].append({
                "index": idx,
                "image_name": image_name,
                "path": image_path
                })
                results["all_valid"] = False
                continue
        
            # 2. 检查图像尺寸
            try:
                actual_image = imageio.imread(image_path)[..., :3]
                actual_height, actual_width = actual_image.shape[:2]
                expected_width, expected_height = expected_size
            
                size_ratio_w = actual_width / expected_width
                size_ratio_h = actual_height / expected_height
            
                # 允许5%的误差
                if abs(size_ratio_w - 1.0) > 0.05 or abs(size_ratio_h - 1.0) > 0.05:
                    results["size_mismatches"].append({
                    "index": idx,
                    "image_name": image_name,
                    "expected": (expected_width, expected_height),
                    "actual": (actual_width, actual_height),
                    "ratio": (size_ratio_w, size_ratio_h)
                    })
                    results["all_valid"] = False
            except Exception as e:
                results["size_mismatches"].append({
                "index": idx,
                "image_name": image_name,
                "error": str(e)
                })
                results["all_valid"] = False
                continue
        
        # 3. 检查主点位置
            cx, cy = K[0, 2], K[1, 2]
            if cx < 0 or cx > actual_width or cy < 0 or cy > actual_height:
                results["principal_point_issues"].append({
                "index": idx,
                "image_name": image_name,
                "principal_point": (cx, cy),
                "image_size": (actual_width, actual_height)
                })
                results["all_valid"] = False
        
            # 4. 验证3D点投影（如果该图像有对应的3D点）
            if image_name in self.point_indices:
                point_indices = self.point_indices[image_name]
                if len(point_indices) > 0:
                    points_world = self.points[point_indices]
                    camtoworld = self.camtoworlds[idx]
                    worldtocam = np.linalg.inv(camtoworld)
                
                    # 转换到相机坐标系
                    points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
                
                    # 投影到图像平面
                    points_proj = (K @ points_cam.T).T
                    points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2)
                    depths = points_cam[:, 2]
                
                    # 过滤有效点（深度>0且在图像范围内）
                    valid_mask = (
                    (points_2d[:, 0] >= 0) & (points_2d[:, 0] < actual_width) &
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < actual_height) &
                    (depths > 0)
                    )
                
                    valid_ratio = valid_mask.sum() / len(points_2d)
                
                    # 如果有效点比例太低（<50%），可能有问题
                    if valid_ratio < 0.5:
                        results["projection_issues"].append({
                        "index": idx,
                        "image_name": image_name,
                        "valid_ratio": valid_ratio,
                        "total_points": len(points_2d),
                        "valid_points": valid_mask.sum()
                        })
                        results["all_valid"] = False
    
        if verbose:
            print("\n" + "="*60)
            print("Image-camera parameter validation")
            print("="*60)
            print(f"Total images: {results['total_images']}")
            print(f"Missing files: {len(results['missing_files'])}")
            print(f"Size mismatches: {len(results['size_mismatches'])}")
            print(f"Principal-point issues: {len(results['principal_point_issues'])}")
            print(f"Projection issues: {len(results['projection_issues'])}")
            print(f"Overall: {'passed' if results['all_valid'] else 'failed'}")
        
            if results['size_mismatches']:
                print("\nImages with size mismatches:")
                for item in results['size_mismatches'][:5]:
                    print(f"  [{item['index']}] {item['image_name']}: "
                          f"expected {item['expected']}, got {item['actual']}")
        
            if results['principal_point_issues']:
                print("\nPrincipal-point issues:")
                for item in results['principal_point_issues'][:5]:
                    print(f"  [{item['index']}] {item['image_name']}: "
                      f"principal point {item['principal_point']}, image size {item['image_size']}")
        
            if results['projection_issues']:
                print("\nProjection validation issues:")
                for item in results['projection_issues'][:5]:
                    print(f"  [{item['index']}] {item['image_name']}: "
                      f"valid-point ratio {item['valid_ratio']:.2%}")
            print("="*60 + "\n")
    
        return results
    
    def verify_train_test_alignment(self, verbose: bool = True, 
                                    visualize: bool = False,
                                    save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify that train/test camera parameters match the corresponding images.
        
        Args:
            verbose: Print a summary.
            visualize: Save projected-point overlays.
            save_dir: Directory for overlay images.
    
        Returns:
            Validation summary.
        """
        results = {
            "train": {"total": 0, "passed": 0, "failed": [], "projection_stats": []},
            "test": {"total": 0, "passed": 0, "failed": [], "projection_stats": []},
            "all_valid": True
        }
        
        if visualize and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            import matplotlib.pyplot as plt
    
        if verbose:
            print("\n" + "="*60)
            print("Validating train image-camera alignment")
            print("="*60)
        for item_idx, global_idx in enumerate(self.train_indices):
            result = self._verify_single_view(
                global_idx=global_idx,
                split="train",
                item_idx=item_idx,
                verbose=verbose,
                visualize=visualize,
                save_dir=save_dir
            )
            results["train"]["total"] += 1
            if result["valid"]:
                results["train"]["passed"] += 1
            else:
                results["train"]["failed"].append(result)
                results["all_valid"] = False
            results["train"]["projection_stats"].append(result.get("projection_stats", {}))
    
        if verbose:
            print("\n" + "="*60)
            print("Validating test image-camera alignment")
            print("="*60)
        for item_idx, global_idx in enumerate(self.test_indices):
            result = self._verify_single_view(
                global_idx=global_idx,
                split="test",
                item_idx=item_idx,
                verbose=verbose,
                visualize=visualize,
                save_dir=save_dir
            )
            results["test"]["total"] += 1
            if result["valid"]:
                results["test"]["passed"] += 1
            else:
                results["test"]["failed"].append(result)
                results["all_valid"] = False
            results["test"]["projection_stats"].append(result.get("projection_stats", {}))
    
        if verbose:
            print("\n" + "="*60)
            print("Train/test view validation summary")
            print("="*60)
            print(f"Train views: {results['train']['passed']}/{results['train']['total']} passed")
            print(f"Test views: {results['test']['passed']}/{results['test']['total']} passed")
            print(f"Overall: {'passed' if results['all_valid'] else 'failed'}")
            
            if results["train"]["failed"]:
                print(f"\nFailed train views ({len(results['train']['failed'])}):")
                for fail in results["train"]["failed"][:5]:
                    print(f"  index {fail['global_idx']} ({fail['image_name']}): {fail['reason']}")
            
            if results["test"]["failed"]:
                print(f"\nFailed test views ({len(results['test']['failed'])}):")
                for fail in results["test"]["failed"][:5]:
                    print(f"  index {fail['global_idx']} ({fail['image_name']}): {fail['reason']}")
            
            train_stats = results["train"]["projection_stats"]
            test_stats = results["test"]["projection_stats"]
            if train_stats:
                valid_ratios = [s.get("valid_ratio", 0) for s in train_stats if s and "valid_ratio" in s]
                if valid_ratios:
                    avg_valid_ratio = np.mean(valid_ratios)
                    print(f"\nAverage train projection valid-point ratio: {avg_valid_ratio:.2%}")
            if test_stats:
                valid_ratios = [s.get("valid_ratio", 0) for s in test_stats if s and "valid_ratio" in s]
                if valid_ratios:
                    avg_valid_ratio = np.mean(valid_ratios)
                    print(f"Average test projection valid-point ratio: {avg_valid_ratio:.2%}")
            print("="*60 + "\n")
    
        return results

    def _verify_single_view(self, global_idx: int, split: str, item_idx: int,
                           verbose: bool = False, visualize: bool = False, 
                           save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify one view's image-camera alignment.
        
        Returns:
            Validation result.
        """
        result = {
            "valid": True,
            "global_idx": global_idx,
            "item_idx": item_idx,
            "split": split,
            "image_name": self.image_names[global_idx],
            "image_path": self.image_paths[global_idx],
            "reason": "",
            "projection_stats": {}
        }
        
        if not os.path.exists(self.image_paths[global_idx]):
            result["valid"] = False
            result["reason"] = f"Image file does not exist: {self.image_paths[global_idx]}"
            return result
        
        try:
            image = imageio.imread(self.image_paths[global_idx])[..., :3]
            actual_height, actual_width = image.shape[:2]
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"Unable to read image: {str(e)}"
            return result
        
        camera_id = self.camera_ids[global_idx]
        K = self.Ks_dict[camera_id].copy()
        camtoworld = self.camtoworlds[global_idx]
        image_name = self.image_names[global_idx]
        
        cx, cy = K[0, 2], K[1, 2]
        if cx < 0 or cx >= actual_width or cy < 0 or cy >= actual_height:
            result["valid"] = False
            result["reason"] = f"Invalid principal point: ({cx:.1f}, {cy:.1f}), image size: ({actual_width}, {actual_height})"
            return result
        
        # COLMAP points should project into the image when the pairing is correct.
        if image_name in self.point_indices:
            point_indices = self.point_indices[image_name]
            if len(point_indices) > 0:
                points_world = self.points[point_indices]
                worldtocam = np.linalg.inv(camtoworld)
                
                points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
                depths = points_cam[:, 2]
                
                points_proj = (K @ points_cam.T).T
                points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2)
                
                valid_mask = (
                    (points_2d[:, 0] >= 0) & (points_2d[:, 0] < actual_width) &
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < actual_height) &
                    (depths > 0)
                )
                
                valid_ratio = valid_mask.sum() / len(points_2d)
                valid_points_2d = points_2d[valid_mask]
                valid_points_rgb = self.points_rgb[point_indices][valid_mask]
                
                result["projection_stats"] = {
                    "total_points": len(points_2d),
                    "valid_points": valid_mask.sum(),
                    "valid_ratio": valid_ratio,
                    "mean_depth": depths[depths > 0].mean() if (depths > 0).any() else 0
                }
                
                if valid_ratio < 0.7:
                    result["valid"] = False
                    result["reason"] = f"3D point projection failed: valid ratio {valid_ratio:.2%} (expected >= 70%)"
                    result["projection_stats"]["failed"] = True
                else:
                    result["projection_stats"]["failed"] = False
                
                if visualize and save_dir and valid_points_2d.shape[0] > 0:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    
                    axes[0].imshow(image)
                    axes[0].scatter(valid_points_2d[:, 0], valid_points_2d[:, 1], 
                                   c=valid_points_rgb / 255.0, s=1, alpha=0.6)
                    axes[0].set_title(f"{split} view {item_idx} (global {global_idx})\n"
                                     f"{image_name}\n"
                                     f"Valid ratio: {valid_ratio:.2%}")
                    axes[0].axis('off')
                    
                    axes[1].imshow(image)
                    scatter = axes[1].scatter(valid_points_2d[:, 0], valid_points_2d[:, 1],
                                            c=depths[valid_mask], cmap='jet', s=1, alpha=0.6)
                    plt.colorbar(scatter, ax=axes[1], label='Depth')
                    axes[1].set_title(f"Depth visualization\n"
                                    f"Mean depth: {result['projection_stats']['mean_depth']:.2f}")
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f"{split}_{item_idx:04d}_idx{global_idx}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
            else:
                result["projection_stats"] = {"total_points": 0, "valid_points": 0, 
                                             "valid_ratio": 0, "warning": "No 3D points"}
        else:
            result["projection_stats"] = {"warning": "Image not in point_indices"}
            if verbose:
                print(f"  Warning: {image_name} has no associated 3D points; skipping projection validation")
        
        if result["valid"] and verbose:
            stats = result["projection_stats"]
            if stats.get("valid_ratio", 0) > 0:
                print(f"  ✓ [{item_idx:3d}] {result['image_name']:30s} "
                      f"投影有效点: {stats['valid_ratio']:.2%} ({stats['valid_points']}/{stats['total_points']})")
        
        return result


class Dataset:
    """A simple dataset class for sparse view reconstruction."""

    parser: Parser
    split: str = "train"
    patch_size: Optional[int] = None
    load_depths: bool = False

    def __init__(self,
                parser: Parser,
                split: str = "train",
                patch_size: Optional[int] = None,
                load_depths: bool = False):
        
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        
        self.reference_depths = {}
        
        
        
        if self.split == "train":
            self.indices = self.parser.train_indices
            #* filter points
            self.parser.filter_points_sparse(self.indices)
            
            if parser.depth_dir is not None and self.parser.n_views > 0:
                self._load_reference_depths()
            
        elif self.split == "test":
            self.indices = self.parser.test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.indices)
    
    
    def _load_reference_depths(self):
        """Load precomputed depth maps for all training views."""
        mvs_depth_dir = self.parser.depth_dir
        n_views = self.parser.n_views
        
        data_dir = self.parser.data_dir.rstrip('/')
        
        dataset_type = self.parser.dataset_type
        if dataset_type == "auto":
            if "mipnerf360" in data_dir.lower() or "360" in data_dir.lower():
                dataset_type = "mipnerf360"
            else:
                dataset_type = "dl3dv"
        
        if dataset_type == "mipnerf360":
            scene_name = os.path.basename(data_dir)
        else:
            parts = data_dir.split('/')
            if 'gaussian_splat' in parts[-1]:
                scene_name = parts[-2]
            else:
                scene_name = parts[-1]
        
        print(f"[Dataset] Loading reference depth from: {mvs_depth_dir}")
        print(f"[Dataset] Dataset type: {dataset_type}, Scene: {scene_name}, n_views: {n_views}")
        
        loaded_count = 0
        for item_idx, global_idx in enumerate(self.indices):
            image_name = self.parser.image_names[global_idx]
            
            depth = _load_mvs_depth(
                mvs_depth_dir=mvs_depth_dir,
                dataset_type=dataset_type,
                scene_name=scene_name,
                n_views=n_views,
                image_name=image_name
            )
            
            if depth is not None:
                self.reference_depths[item_idx] = depth
                loaded_count += 1
        
        print(f"[Dataset] Loaded {loaded_count}/{len(self.indices)} reference depths")
    
    def update_reference_depth(self, image_id: int, depth: np.ndarray):
        """Update the cached depth for a training view."""
        self.reference_depths[image_id] = depth
    
    def get_reference_depth(self, image_id: int) -> Optional[np.ndarray]:
        """Return the cached depth for a training view."""
        return self.reference_depths.get(image_id, None)
    
    def has_reference_depth(self, image_id: int) -> bool:
        """Check whether a training view has cached depth."""
        return image_id in self.reference_depths

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", type=str, default="data/scene")
    arg_parser.add_argument("--factor", type=int, default=4)
    arg_parser.add_argument("--n_views", type=int, default=3)
    arg_parser.add_argument("--dataset_type", type=str, default="dl3dv")
    args = arg_parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, 
        factor=args.factor, 
        normalize=False, 
        test_every=8,
        n_views=args.n_views,
        dataset_type=args.dataset_type,
    )
    
    print(f"Total images: {len(parser.image_names)}")
    print(f"Train images: {len(parser.train_indices)}, they are {parser.train_indices}")
    print(f"Test images: {len(parser.test_indices)}, they are {parser.test_indices}")
    
    train_dataset = Dataset(parser, split="train")
    test_dataset = Dataset(parser, split="test")
    
    print(f"Train dataset: {len(train_dataset)} images.")
    print(f"Test dataset: {len(test_dataset)} images.")
    
    print("Scene scale is: ", parser.scene_scale)
    
