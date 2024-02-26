import numpy as np
import typing
import os

class GeometryException():
    pass


UmeyamaResult = typing.Tuple[np.ndarray, np.ndarray, float]


def umeyama_alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = False) -> UmeyamaResult:
    """
    Provides generic geometry algorithms.
    author: Michael Grupp

    This file is part of evo (github.com/MichaelGrupp/evo).

    evo is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    evo is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with evo.  If not, see <http://www.gnu.org/licenses/>.

    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise GeometryException("Degenerate covariance rank, "
                                "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def arc_len(x: np.ndarray) -> float:
    """
    :param x: nxm array of points, m=dimension
    :return: the (discrete approximated) arc-length of the point sequence
    """
    return np.sum(np.linalg.norm(x[:-1] - x[1:], axis=1))


def accumulated_distances(x: np.ndarray) -> np.ndarray:
    """
    :param x: nxm array of points, m=dimension
    :return: the accumulated distances along the point sequence
    """
    return np.concatenate(
        (np.array([0]), np.cumsum(np.linalg.norm(x[:-1] - x[1:], axis=1))))

def align_and_transform(gt_pose_se3, pred_pose_se3):
    '''
    use umeyama alignment to find the transformation and scale between gt_pose and predicted pose

    args:
        gt_pose_se3: SE3 pose for ground truth (seq_len X 4 X 4)
        pred_pose_se3: SE3 pose for ground truth (seq_len X 4 X 4)
    returns:
        transformed_final: scaled and transformed predicted pose

    '''
    ref_um = []
    pred_um = []
    for gt_se3, pred_se3 in zip(gt_pose_se3, pred_pose_se3):
        ref_um.append(gt_se3[0:3, 3].reshape(3,1))
        pred_um.append(pred_se3[0:3, 3].reshape(3,1))


    ref_um = np.hstack(ref_um)
    pred_um = np.hstack(pred_um)

    rot, trans, scale = umeyama_alignment(pred_um, ref_um, with_scale=True)
    pred_um_scaled = pred_um*scale 
    transformation = np.vstack((np.hstack((rot, trans.reshape(3,1))),np.array([0,0,0,1])))
    transformed_pred = transformation[0:3, 0:3]@pred_um_scaled
    transformed_final = transformed_pred+ transformation[0:3,3].reshape(3,1)

    return transformed_final 

def align_trajectory(predicted_pose_file, gt_pose_file, seq_length = 5):
    '''
    takes path of two txt file containing predicted and ground truth pose.
    returns the scaled and aligned trajectory using umeyema transformation
    args: 
        predicted_pose_file(path): path to predicted pose file (Kitti format)
        gt_pose_file(path): path to ground truth pose file (Kitti format)
        seq_length(int): sequence length by which the poses are to be aligned and scaled

    returns:
        transformed_pred: prediction scaled and aligned with gt
        ult_gt: final formatted ground truth
        
    '''
    assert os.path.isfile(predicted_pose_file)
    assert os.path.isfile(gt_pose_file)

    with open(predicted_pose_file) as f:
        prediction_file = f.readlines()

    with open(gt_pose_file) as f:
        gt_file = f.readlines()

    assert len(prediction_file)== len(gt_file)
    # print(len(prediction))
    # print(len(gt))
    gts = []
    preds = []
    pred_pose_se3 = []
    gt_pose_se3 = []
    
    transformed_pred = np.zeros((3,1))
    for i, (gt_line, pred_line) in enumerate(zip(gt_file, prediction_file)):

        gt_pose = np.array(gt_line.split(' ')).astype(float).reshape(3,4)
        gt_pose_se3.append(np.concatenate((gt_pose,np.array([0,0,0,1]).reshape(1,4)),axis=0))

        pred_pose = np.array(pred_line.split(' ')).astype(float).reshape(3,4)
        
        pred_pose_se3.append(np.concatenate((pred_pose,np.array([0,0,0,1]).reshape(1,4)),axis=0))
        
        gts.append(gt_pose[:,3])
        preds.append(pred_pose[:,3])

        if (i+1)%seq_length==0:
            # print(align_and_transform(gt_pose_se3, pred_pose_se3).shape)
            transformed_pred =np.hstack((transformed_pred, align_and_transform(gt_pose_se3, pred_pose_se3)))
            pred_pose_se3 = []
            gt_pose_se3 = []

    if len(pred_pose_se3)!=0:
        transformed_pred =np.hstack((transformed_pred, align_and_transform(gt_pose_se3, pred_pose_se3)))

    ult_gt = np.vstack(gts).T

    return transformed_pred[:,1:], ult_gt

def find_ATE(pred:np.ndarray, gt:np.ndarray)->tuple():
    '''
    finds the Absolute Trajectory Error between two trajectories.
    args:
        pred(np.ndarray): 3 x m array, where m is number of datapoints
        gt(np.ndarray): 3 x m array, where m is number of datapoints

    returns:
        Total_ATE(float): total sum of the distance between each pair 
        Mean_ATE(float): mean ATE-> total_ate/m

    '''
    assert gt.shape == pred.shape

    diff = gt.T - pred.T
    Total_ATE = np.linalg.norm(diff, axis=1).sum()
    Mean_ATE = np.linalg.norm(diff, axis=1).mean()
    return (float(Total_ATE), float(Mean_ATE))


def mean_squared_error_depth(predicted: np.ndarray, gt: np.ndarray, normalize=True)->float:
    '''
    Takes two 2D np array and calculates the pixelwise MSE error between them 
    (normalized by max value or scaled using median)
    Args:
        predicted: predicted depth map
        gt: ground truth depth map
        normalize(bool): normalize if true scale using median if False
    
    Returns:
        mse: mean squared error pixelwise

    '''
    assert predicted.shape == gt.shape 
    if normalize:
        #normalize both
        gt = gt/gt.max()
        mask = np.array(gt, dtype=bool).astype(int)

        predicted = (predicted*mask) #element wise multiply the mask
        predicted = predicted/predicted.max()

    else:
        gt_median = median_of_non_zero_values(gt)
        predicted_median = median_of_non_zero_values(predicted)

        scale = gt_median/predicted_median
        print(scale)
        predicted = predicted*scale


    mse = ((gt-predicted)**2).mean()
    return float(mse)

def median_of_non_zero_values(arr):
    """
    Calculates the median of the non-zero values in a flattened array.

    Args:
    arr: A NumPy array of any shape.

    Returns:
    The median of the non-zero values in the flattened array, or np.nan if there are no non-zero values.
    """

    flat_arr = arr.flatten()
    non_zero_elements = flat_arr[flat_arr != 0]
    if len(non_zero_elements) > 0:
        return np.median(non_zero_elements)
    else:
        return np.nan

if __name__=='__main__':
    a = np.random.random_integers(0,40, size=(500,500))
    b = np.random.random_integers(0,40, size=(500,500))

    print(np.sqrt(mean_squared_error_depth(a,b, normalize=False)))
    
