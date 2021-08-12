from torch import logical_and, sum
from vector_functions import area_para


def interpolate2d(coords_in, values_in, coords_out):
    """
    Linearly interpolate ordered values from one set of coordinates to a new set of coordinates.

    The coordinates are assumed to be ordered in 2D, i.e. neighboring indices correspond to
    neighboring coordinates, and are assumed to lie on a distorted 'grid'. This assumption is used
    to pick vertices and form triangles. The full derivation can be found in
    'Basis Tranformation for interpolation.ipynb'.

    Inputs
    ------
        coords_in       Vector array Nx x Ny x 2. Input coordinates.
        values_in       Scalar array Nx x Ny x 1. Input values.
        coords_out      Vector array Mx x My x 2. Output coordinates.

    Outputs
    -------
        values_out              Scalar array Mx x My x 1. Output values.
        values_out_unfolded     Scalar array Mx x My x Nx x Ny x 1. Masked unfolded output values.
                                This contains the uncollapsed interpolated masked values and hence
                                contains lot's of zeros. This data can be useful if results from
                                overlapping triangles should be combined in a different way than
                                summation.
    """
    # Dimensions of input and output
    Ndim, Nx, Ny = coords_in.shape
    Mdim, Mx, My = coords_out.shape

    # Reshape input coordinates & values_in, and define A,B,C,D points
    A  = coords_in[:-1, :-1, :].view(1, 1, Nx-1, Ny-1, 2)
    B  = coords_in[ 1:, :-1, :].view(1, 1, Nx-1, Ny-1, 2)
    C  = coords_in[:-1,  1:, :].view(1, 1, Nx-1, Ny-1, 2)
    D  = coords_in[ 1:,  1:, :].view(1, 1, Nx-1, Ny-1, 2)

    VA = values_in[:-1, :-1, :].view(1, 1, Nx-1, Ny-1, 2)
    VB = values_in[ 1:, :-1, :].view(1, 1, Nx-1, Ny-1, 2)
    VC = values_in[:-1,  1:, :].view(1, 1, Nx-1, Ny-1, 2)
    VD = values_in[ 1:,  1:, :].view(1, 1, Nx-1, Ny-1, 2)

    # Reshape target coordinates
    T = coords_out.view(Mx, My, 1, 1, 2)

    # Compute difference vectors
    AB = B - A
    AC = C - A
    AT = T - A
    DB = B - D
    DC = C - D
    DT = T - D

    # Compute coefficients
    denom_A = area_para(AB, AC)
    b_A = area_para( AT, AC) / denom_A
    c_A = area_para(-AT, AB) / denom_A

    denom_D = area_para(DB, DC)
    b_D = area_para( DT, AC) / denom_D
    c_D = area_para(-DT, AB) / denom_D

    # Mask coefficients to limit to corresponding triangles
    # Furthermore,
    # in order to correctly count the edges, the A- and D-masks are slightly
    # different: A uses ≥ & ≤, while D uses >. Lastly, bc_A is used both for
    # the A- and D-masks, to prevent rounding errors causing excluding or
    # counting points in both masks. Some rounding errors can still occur
    # for two nearby A- or D-triangles, though these should be virtually
    # nonexistent in practice.
    bc_A = b_A + c_A
    bc_D = b_D + c_D
    mask_A = logical_and(logical_and(b_A > 0, c_A > 0), bc_A < 1)
    mask_D = logical_and(logical_and(b_D > 0, c_D > 0), bc_D < 1)

    # Compute interpolated values
    values_out_unfolded = mask_A * (VA + (VB-VA)*b_A + (VC-VA)*c_A)\
                        + mask_D * (VD + (VB-VD)*b_D + (VC-VD)*c_D)

    values_out = sum(values_out_unfolded, (2, 3))

    return values_out, values_out_unfolded
