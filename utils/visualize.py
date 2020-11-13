# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

joints = [(0, 1), (0, 2), (0, 17), (17, 6), (17, 5), (5, 7), (7, 9), (6, 8),
          (8, 10), (17, 12), (12, 14), (14, 16), (17, 11), (11, 13), (13, 15)]


def draw_subplot(fig, index, points, title='front view'):
    ax = fig.add_subplot(index, projection='3d')
    # angle rotation of each view
    degs = (-70, -90) if title=='front view' else (-3, -87) \
        if title=='top view' else (-90, -90)

    ax.view_init(elev= degs[0], azim=degs[1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.title.set_text(title.title())
    ax.set_title(title.title(), y=1, fontstyle='oblique' )
    ax.set_aspect('equal')
    ax.dist = 6.5

    if title=='front view':
        draw_front_view(ax, points)

    elif title=='top view':
        draw_top_view(ax, points)

    elif title=='side view':
        draw_side_view(ax, points)

def draw_top_view(ax, points):
    for xyz in points:
        missed_joints = np.unique(np.argwhere(xyz== 0)[:, 1])
        # draw line (point1 to point2)
        for joint in joints:
            if  joint[0] in missed_joints or joint[1] in missed_joints: continue
            # draw 3D line plot
            ax.plot(
                [xyz[0, joint[0]], xyz[0, joint[1]]],
                [xyz[1, joint[0]], xyz[1, joint[1]]],
                [xyz[2, joint[0]], xyz[2, joint[1]]],
                c='b', zdir='z', linewidth=2
            )
            # draw 3D scatter
            x = xyz[0][joint[1]]
            y = xyz[1][joint[1]]
            z = xyz[2][joint[1]]
            ax.scatter(x, y, z, marker='o', c='limegreen')
            # ax.text(x, y, z, int(joint[1]))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H-20)
    ax.set_zlim(0, 15)

def draw_side_view(ax, points):
    points[:, 0] = points[:, 0] / (W/10)
    points[:, 1] = points[:, 1] / (H/10)

    for xyz in points:
        missed_joints = np.unique(np.argwhere(xyz== 0)[:, 1])
        # draw line (point1 to point2)
        for joint in joints:
            if  joint[0] in missed_joints or joint[1] in missed_joints: continue
            # draw 3D line plot
            ax.plot(
                [xyz[2, joint[0]], xyz[2, joint[1]]],
                [xyz[1, joint[0]], xyz[1, joint[1]]],
                [xyz[0, joint[0]], xyz[0, joint[1]]],
                c='b', zdir='z', linewidth=2
            )
            x = xyz[0][joint[1]]
            y = xyz[1][joint[1]]
            z = xyz[2][joint[1]]
            ax.scatter(z, y, x, marker='o', c='limegreen', markersize=2)

    ax.set_xlim(-1, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

def draw_front_view(ax, points):
    for xyz in points:
        missed_joints = np.unique(np.argwhere(xyz== 0)[:, 1])
        # draw line (point1 to point2)
        for joint in joints:
            if  joint[0] in missed_joints or joint[1] in missed_joints: continue
            # draw 3D line plot
            ax.plot(
                [xyz[0, joint[0]], xyz[0, joint[1]]],
                [xyz[1, joint[0]], xyz[1, joint[1]]],
                [xyz[2, joint[0]], xyz[2, joint[1]]],
                c='b', zdir='z', linewidth=3
            )
            # draw 3D scatter
            x = xyz[0][joint[1]]
            y = xyz[1][joint[1]]
            z = xyz[2][joint[1]]
            ax.scatter(x, y, z, marker='o', c='limegreen', s=9*2)
            # ax.text(x, y, z, int(joint[1]))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H-20)
    ax.set_zlim(0, 15)

def draw_3d(image, all_xyz, multiview=False):
    global H, W

    H, W = image.shape[:2]
    # init main figure
    fig = plt.figure(figsize=(12, 5))
    ax_in = fig.add_subplot(1, 4, 1) if multiview \
        else fig.add_subplot(1, 2, 1)

    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.imshow(image[...,::-1], aspect='equal')

    # scale x,y coordinate to max depth parameter=10

    if multiview:
        draw_subplot(fig, 142, all_xyz, title='front view')
        draw_subplot(fig, 143, all_xyz, title='top view')
        draw_subplot(fig, 144, all_xyz, title='side view')
    else:
        draw_subplot(fig, 122, all_xyz, title='front view')

    # plt.tight_layout()
    fig.canvas.draw()

    return fig

def draw_2d(image, all_keypoints):
    """
    draw all persons keypoints
    """
    for keypoints in all_keypoints:
        visibilities = []
        # draw points on image
        for kp in keypoints:
            if kp[1]==0 or kp[2]==0 or kp[0]==3 or kp[0]==4:
                visibilities.append(kp[0])
                continue

            cv2.circle(image, (int(kp[1]), int(kp[2])), 2, (0, 0, 0), 2)
        #     cv2.putText(res, str(kp[0]), center, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255) , 1)
        # draw line on image
        for joint in joints:
            if joint[0] in visibilities or joint[1] in visibilities: continue
            cv2.line(
                image,
                tuple(keypoints[joint[0]][1:].astype('int32')),
                tuple(keypoints[joint[1]][1:].astype('int32')),
                (0, 255, 0), 2
            )

    return image
