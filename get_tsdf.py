import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM
import src.fusion as fusion
import open3d as o3d
from src.utils.datasets import get_dataset

import matplotlib.pyplot as plt
import cv2

def update_cam(cfg):
    """
    Update the camera intrinsics according to pre-processing config, 
    such as resize or edge crop.
    """
    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
    # resize the input images to crop_size (variable name used in lietorch)
    if 'crop_size' in cfg['cam']:
        crop_size = cfg['cam']['crop_size']
        H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        sx = crop_size[1] / W
        sy = crop_size[0] / H
        fx = sx*fx
        fy = sy*fy
        cx = sx*cx
        cy = sy*cy
        W = crop_size[1]
        H = crop_size[0]

        

    # croping will change H, W, cx, cy, so need to change here
    if cfg['cam']['crop_edge'] > 0:
        H -= cfg['cam']['crop_edge']*2
        W -= cfg['cam']['crop_edge']*2
        cx -= cfg['cam']['crop_edge']
        cy -= cfg['cam']['crop_edge']
    
    return H, W, fx, fy, cx, cy

def init_tsdf_volume(cfg, args):
    # scale the bound if there is a global scaling factor
    scale = cfg['scale']
    bound = torch.from_numpy(
        np.array(cfg['mapping']['bound'])*scale)
    bound_divisible = cfg['grid_len']['bound_divisible']
    # enlarge the bound a bit to allow it divisible by bound_divisible
    bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
                        bound_divisible).int()+1)*bound_divisible+bound[:, 0]

    # TSDF volume
    H, W, fx, fy, cx, cy = update_cam(cfg)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix # (3, 3)

    print("Initializing voxel volume...")
    vol_bnds = np.array(bound)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=4/256) #4.0/512)

    frame_reader = get_dataset(cfg, args, scale)

    # load est cam pose
    #est_cam_ls = torch.load('est_cam.pt')


    # tsdf fusion in open3d
    '''
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 256.0,
                sdf_trunc=80.0 * scale / 256.0,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for idx in range(len(frame_reader)):
        print(f'frame: {idx}')
        _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
        #est_c2w = est_cam_ls[idx]

        # convert to open3d camera pose
        c2w = gt_c2w.cpu().numpy()
        # convert to open3d camera pose
        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        w2c = np.linalg.inv(c2w)
        #cam_points.append(c2w[:3, 3])
        depth = gt_depth.cpu().numpy()
        color = gt_color.cpu().numpy()

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(np.array(
            (color * 255).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1,
            depth_trunc=1000,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh('open3d_fusion.ply', mesh)
    '''
    loss_ls = []
    
    for idx in range(len(frame_reader)):
        print(f'frame: {idx}')
        _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
        #est_c2w = est_cam_ls[idx]

        # convert to open3d camera pose
        c2w = gt_c2w.cpu().numpy()

        # loss_cam = torch.sum(est_c2w - c2w)
        # loss_ls.append(loss_cam.item())
        #c2w = est_c2w.cpu().numpy()
        #print(c2w)
            
        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0

        depth = gt_depth.cpu().numpy() # (368, 496, 3)
        color = gt_color.cpu().numpy()
        depth = depth.astype(np.float32)
        color = np.array((color * 255).astype(np.uint8))

        tsdf_vol.integrate(color, depth, intrinsic, c2w, obs_weight=1.)

    print('Getting TSDF volume')
    tsdf_volume, _, bounds = tsdf_vol.get_volume()

    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("tsdf_volume/room1_tsdf.ply", verts, faces, norms, colors)

    # print('Getting cam loss stat ...')
    # loss = np.array(loss_ls).reshape(1, 2000)
    # print(loss.shape)
    # x_axis = np.arange(0, loss.shape[1], 1).reshape(1, 2000)
    # plt.plot(x_axis, loss, 'r*')
    # plt.savefig(f'camloss.jpg')
    # plt.cla()

    # mask = (tsdf_volume == 1.0)
    # tsdf_volume[mask] = 0


    tsdf_volume = torch.tensor(tsdf_volume)
    tsdf_volume = tsdf_volume.reshape(1, 1, tsdf_volume.shape[0], tsdf_volume.shape[1], tsdf_volume.shape[2])
    tsdf_volume = tsdf_volume.permute(0, 1, 4, 3, 2)
    
    return tsdf_volume, bounds

def get_tsdf():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    tsdf_volume, bounds = init_tsdf_volume(cfg, args)

    # torch.save(tsdf_volume, 'tsdf_volume/office0_tsdf_volume.pt')
    # torch.save(bounds, 'tsdf_volume/office0_bounds.pt')




get_tsdf()
