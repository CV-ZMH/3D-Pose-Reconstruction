# -*- coding: utf-8 -*-
import cv2
import warnings
warnings.filterwarnings('ignore')

from pose import Pose
from depth import Depth
from utils.opts import pose_opts, depth_opts
from make_3d import predict_3D_pose


def main():

    model_type = 'trt' if 'trt' in pose_args.model_file else 'torch'
    backbone = 'densenet121' if 'densenet121' in pose_args.model_file else 'resnet18'
    width, height = depth_args['width'], depth_args['height']

    # initialize trtpose and depth prediction object
    pose = Pose(pose_args, model_type=model_type, backbone=backbone)
    depth = Depth(depth_args)

    if pose_args.mode == 'image':
        bgr_img = cv2.imread(pose_args.source)
        image = predict_3D_pose(bgr_img, pose, depth, \
                                width, height, multiview=pose_args.multiview)

        cv2.imwrite(pose_args.source[:-4]+'_3D_{}.jpg'.format('multiview' if pose_args.multiview \
                                                              else 'frontview'), image)
        if pose_args.show:
            cv2.imshow('3D-Pose estimation', image)
            cv2.waitKey(0)

    elif pose_args.mode == 'video':
        cap = cv2.VideoCapture(pose_args.source)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        num_frames = 0
        while True:
            ret, bgr_img = cap.read()
            if not ret: break
            image = predict_3D_pose(bgr_img, pose, depth, \
                                    width, height, multiview=pose_args.multiview)

            if num_frames == 0:
                out = cv2.VideoWriter(
                    pose_args.source[:-4] +'_3d_{}.avi'.format('multiview' if pose_args.multiview else 'frontview'),
                    fourcc, 20.0, (image.shape[1], image.shape[0]))

            num_frames +=1
            out.write(image)
            if pose_args.show:
                cv2.imshow('3D-Pose estimation', image)
                k = cv2.waitKey(1)
                if k==ord('q') or k == 27: break
        out.release()
        cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    depth_args = depth_opts()
    pose_args = pose_opts()
    main()
