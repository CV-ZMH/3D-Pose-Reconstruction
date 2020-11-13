# -*- coding: utf-8 -*-
import argparse

def pose_opts():
    """
    Optional Argument for 2D pose estimatino
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['video', 'image'],
                    help='vid or img mode for demo')
    ap.add_argument('--source', required=True,
                    help='input file for pose estimation')

    ap.add_argument('--multiview', action='store_true', help='Final 3D image is in multiview or not.')

    ap.add_argument('--show', action='store_true', help='Final 3D image is in multiview or not.')

    ap.add_argument('--json', default='model_weights/human_pose.json',
                    help='json file for pose estimation')

    ap.add_argument('--model_file', default='model_weights/densenet121_trtpose.pth',
                    help='model file path')

    return ap.parse_args()

def depth_opts():
    """
    Default params for depth estimation model
    """
    opts = {
        'checkpoint' : 'model_weights/nyu_resnet50',
        'max_depth' : 10,
        'dataset' : 'nyu',
        'height' : 480,
        'width' : 640,
        }

    return opts
