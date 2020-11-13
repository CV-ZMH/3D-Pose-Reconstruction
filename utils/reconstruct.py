import numpy as np

def get_keypoints(humans, counts, peaks, height, width):
    """
    Get all persons keypoint
    """
    all_keypoints = np.zeros((counts, 18, 3), dtype=np.float32) #  counts contain num_persons
    for count in range(counts):

        human = humans[0][count]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]
                peak = (j, float(peak[1] * width), float(peak[0] * height))
                all_keypoints[count, j] = peak
            else:
                peak = (j, 0., 0.)
                all_keypoints[count, j] = peak

    return all_keypoints


def to_xyz(all_keypoints, depth):
    """
    2d keypoints + depth image to 3D keypoints (XYZ)
    """
    all_keypoints = all_keypoints.astype(np.int32)
    points_3d = np.zeros((len(all_keypoints), 18, 3), dtype=np.float32)

    for person in range(all_keypoints.shape[0]):
        keypoints = all_keypoints[person]
        for idx, xy in enumerate(keypoints):
            if 0 in xy[1:]:
                z = 0
            else:
                z = depth[xy[2], xy[1]]
            xyz = [xy[1], xy[2], z]
            points_3d[person, idx] = xyz

    points_3d = np.transpose(points_3d, (0, 2, 1)).astype(np.float32)

    return points_3d
