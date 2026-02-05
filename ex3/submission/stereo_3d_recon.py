import numpy as np

from calibration import compute_kx_ky, estimate_f_b
from extract_patches import extract_patches


def triangulate(u_left, u_right, v, calib_dict):
    """
    Triangulate (determine 3D world coordinates of) a set of points given their projected coordinates in two images.
    These equations are according to the simple setup, where C' = (b, 0, 0)

    Parameters
    ----------
        u_left: NumPy array of shape (num_points,)
            Projected u-coordinates of the 3D-points in the left image
        u_right: NumPy array of shape (num_points,)
            Projected u-coordinates of the 3D-points in the right image
        v: NumPy array of shape (num_points,)
            Projected v-coordinates of the 3D-points (same for both images)
        calib_dict: dict
            Dict containing complete set of camera parameters.
            (Expected to contain kx, ky, f, b)
    
    Returns
    -------
        NumPy array of shape (num_points, 3):
            Triangulated 3D coordinates of the input - in units of [mm]
    """
    b = calib_dict['b']
    f = calib_dict['f']
    kx = calib_dict['kx']
    ky = calib_dict['ky']
    ox = calib_dict['o_x'] 
    oy = calib_dict['o_y']
    ul_norm = u_left - ox
    ur_norm = u_right - ox
    v_norm = v - oy
    d = ul_norm - ur_norm
    epsilon = 1e-16 
    d[d == 0] = epsilon
    z = (f * kx * b) / d
    x = (ul_norm * b) / d
    y = (v_norm * kx * b) / (d * ky)
    returnable = np.column_stack((x, y, z))
    return returnable


def compute_ncc(img_l, img_r, p):
    """
    Calculate normalized cross-correlation (NCC) between patches at the same row in two images.
    
    The regions near the boundary of the image, where the patches go out of image, are ignored.
    That is, for an input image, "p" number of rows and columns will be ignored on each side.

    For input images of size (H, W, C), the output will be an array of size (H - 2*p, W - 2*p, W - 2*p)

    Parameters
    ----------
        img_l: NumPy array of shape (H, W, C)
            Left image
        img_r: NumPy array of shape (H, W, C)
            Right image
        p: int
            Defines square neighborhood. Patch-size is (2*p+1, 2*p+1)
                              
    Returns
    -------
        NumPy array of shape (H - 2*p, W - 2*p, W - 2*p):
        The value output[r, c_l, c_r] denotes the NCC between the patch centered at (r + p, c_l + p) 
        in the left image and the patch centered at  (r + p, c_r + p) at the right image.
    """

    assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
    assert img_l.shape == img_r.shape, "Shape mismatch."
    
    H, W, C = img_l.shape

    # Extract patches - patches_l/r are NumPy arrays of shape H, W, C * (2*p+1)**2
    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)
    
    # Standardize each patch
    epsilon = 1e-8
    mean_l = patches_l.mean(axis=2, keepdims=True)
    std_l = patches_l.std(axis=2, keepdims=True)
    patches_l = (patches_l - mean_l) / (std_l + 1e-8)

    mean_r = patches_r.mean(axis=2, keepdims=True)
    std_r = patches_r.std(axis=2, keepdims=True)
    patches_r = (patches_r - mean_r) / (std_r + 1e-8)


    # Compute correlation (using matrix multiplication) - corr will be of shape H, W, W
    #to save time i should search in less pixels
    corr = np.zeros((H, W, W))
    for i in range(H):
        corr[i] = np.matmul(patches_l[i], patches_r[i].T) / (C * (2*p+1)**2)

    # Ignore boundaries
    return corr[p:H-p, p:W-p, p:W-p]


class Stereo3dReconstructor:
    def __init__(self, p=6, w_mode='none'):
        """
        Feel free to add hyper parameters here, but be sure to set defaults
        
        Args:
            p       ... Patch size for NCC computation
            w_mode  ... Weighting mode. I.e. method to compute certainty scores
        """
        self.p = p
        self.w_mode = w_mode

    def fill_calib_dict(self, calib_dict, calib_points):
        """ Fill missing entries in calib dict - nothing to do here """
        calib_dict['kx'], calib_dict['ky'] = compute_kx_ky(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        
        return calib_dict

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        """
        Compute point correspondences for two images and perform 3D reconstruction.

        Parameters
        ----------
            img_l: NumPy array of shape (H, W, C)
                Left image
            img_r: NumPy array of shape (H, W, C)
                Right image
            calib_dict: dict
                Dict containing complete set of camera parameters.
                (Expected to contain kx, ky, f, b)
        
        Returns
        -------
            NumPy array of shape (H, W, 4):
                Array containing the re-constructed 3D world coordinates for each pixel in the left image.
                Boundary points - which are not well defined for NCC - may be padded with 0s.
                4th dimension holds the certainties, for now just 1.
        """

        assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."
    
        H, W, C = img_l.shape
        ncc = compute_ncc(img_l, img_r, self.p)
        ncc = np.pad(ncc, ((self.p, self.p), (self.p, self.p), (self.p, self.p)),
                    mode='constant', constant_values=0)
        u_r = np.argmax(ncc, axis=2)
        v, u_l = np.indices((H, W))
        idx_left  = np.clip(u_r - 1, 0, W - 1)
        idx_right = np.clip(u_r + 1, 0, W - 1)
        c_prev = ncc[v, u_l, idx_left]
        c_curr = ncc[v, u_l, u_r]
        c_next = ncc[v, u_l, idx_right]
        denom = 2 * (2 * c_curr - c_next - c_prev)
        denom[denom == 0] = 1e-10 
        delta = (c_next - c_prev) / denom
        delta = np.clip(delta, -0.5, 0.5)
        valid_refinement = (u_r > 0) & (u_r < W - 1)
        delta[~valid_refinement] = 0
        u_r = u_r.astype(float) + delta
        points_3d = triangulate(
            u_left=u_l.ravel(),
            u_right=u_r.ravel(),
            v=v.ravel(),
            calib_dict=calib_dict
        ).reshape(H, W, 3)

        certainty = c_curr.copy()
        certainty[~valid_refinement] = 0.0
        certainty[certainty < 0] = 0.0
        certainty = certainty.reshape(H, W, 1)

        return np.concatenate((points_3d, certainty), axis=2)