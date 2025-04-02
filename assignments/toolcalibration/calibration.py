# Copyright (c) 2025 University of Bern. All rights reserved.
import numpy as np


def pivot_calibration(transforms):
    """
    Pivot calibration

    Keyword arguments:
    transforms -- A list of 4x4 transformation matrices from the tracking system (Fi)
                  representing the tracked tool's position and orientation at
                  different instances.

    Returns:
    T          -- The calibration matrix T (in homogeneous coordinates) that defines
                  the offset (p_t) from the tracked part to the pivot point (tool tip).
    """

    ## TODO: Implement pivot calibration as discussed in the lecture

    A = []
    b = []
    I = np.eye(3)

    for F in transforms:
        Ri = F[:3, :3]   # extract rotation
        pi = F[:3, 3]    # extract translation

        A.append(np.hstack((Ri, -I)))
        b.append(-pi)

    A = np.vstack(A)
    b = np.vstack(b).reshape(-1,1)

    # use least square methode
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Ri*pt -pp = pi -> solve pt,pp

    pt, pp = p[:3], p[3:]

    T = np.eye(4)
    T[:3, 3] = pt.flatten()

    return T


def calibration_device_calibration(camera_T_reference, camera_T_tool, reference_T_tip):
    """
    Tool calibration using calibration device

    Keyword arguments:
    camera_T_reference -- Transformation matrix from reference (calibration device) to camera.
    camera_T_tool      -- Transformation matrix from tool to camera.
    reference_T_tip    -- Transformation matrix from tip to reference (calibration device).

    Returns:
    T                  -- Calibration matrix from tool to tip.
    """

    ## TODO: Implement a calibration method which uses a calibration device

    tool_T_tip = np.linalg.inv(camera_T_tool) @ camera_T_reference @ reference_T_tip

    T = tool_T_tip
    
    return T
