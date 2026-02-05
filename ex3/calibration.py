import numpy as np


def compute_kx_ky(calib_dict):
    """
    Given a calibration dictionary, compute kx and ky (in units of [px/mm]).
    
    kx -> Number of pixels per millimeter in x direction (ie width)
    ky -> Number of pixels per millimeter in y direction (ie height)
    """
    
    ky = calib_dict['height']/calib_dict['aperture_h']
    kx = calib_dict['width']/calib_dict['aperture_w']
    
    return kx, ky


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal lenght f and baseline b from provided calibration points.

    Note:
    In real life multiple points are useful for calibration - in case there are erroneous points.
    Here, this is not the case. It's OK to use a single point to estimate f, b.
    
    Parameters
    ----------
        calib_dict: dict
            Incomplete calibaration dictionary
        calib_points: pd.DataFrame
            Calibration points provided with data. (Units are given in [mm])
        n_points: int
            Number of points used for estimation
        
    Returns
    -------
        f: float
            Focal lenght [mm]
        b: float
            Baseline [mm]
    """
    # Choose n_points from DataFrame
    if n_points is not None:
        calib_points = calib_points.head(n_points)
    else: 
        n_points = len(calib_points)
    ox = calib_dict['o_x']
    oy = calib_dict['o_y']
    ky = calib_dict['height']/calib_dict['aperture_h']
    kx = calib_dict['width']/calib_dict['aperture_w']
    # Get all values from the first row (at position 0) as a numpy array
    first_row_values = calib_points.iloc[0].values

    # Unpack those values into your variables
    ul, vl, ur, vr, x, y, z = first_row_values[0:7]
    f = ((ul - ox) * z) / (kx * x)
    b = x - (((ur - ox) * z) / (f * kx))
    return f, b
