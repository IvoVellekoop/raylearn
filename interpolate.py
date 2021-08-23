from torch import logical_and, stack
from vector_functions import area_para


def interpolate2d(coords_in, values_in, coords_out):
    """
    Linearly interpolate ordered values from one set of coordinates to a new set of coordinates.

    The input coordinates are assumed to be ordered in 2D, i.e. neighboring indices correspond to
    neighboring coordinates, and are assumed to lie on a distorted 'grid'. This assumption is used
    to pick neighboring vertices and form triangles. The full derivation can be found in
    'Basis Tranformation for interpolation.ipynb'. The triangles from the input distorted grid may
    overlap. Since the final output values from overlapping triangles can be interpreted in
    different ways, depending on what the data represents, the last step of computing the final
    output values is not handled in this function. Lastly, the output coordinates do not need to be
    in an ordered grid.

    Inputs
    ------
        coords_in       Vector array Nx x Ny x 2. Input coordinates. Nx and Ny are grid dimensions.
                        2 denotes the vector dimension.
        values_in       Scalar array Nx x Ny x 1. Input values. Nx and Ny are grid dimensions.
                        2 denotes the vector dimension.
        coords_out      Vector array Mx x My x 2. Output coordinates. Mx and My are grid dimensions.
                        2 denotes the vector dimension.

    Outputs
    -------
        values_out_unfolded     Scalar array 2 x Nx x Ny x Mx x My x 1. Unfolded output values.
                                This contains the uncollapsed interpolated masked values and hence
                                contains lots of zeros. This data can be useful if results from
                                overlapping triangles should be combined in a different way than
                                summation. The 0th dimension denotes the A/D-triangles. The 1th and
                                2th dimension denote the input coordinates. Depending on the data,
                                overlapping triangles might be interpreted in different ways.
                                Therefore, this last step of computing the final output values is
                                not done in this function. For instance, if the contributions
                                should be summed, this could be done with:
                                    sum(mask * values_out_unfolded, (0, 1, 2))
        mask                    Logical mask 2 x Nx x Ny x Mx x My x 1. Unfolded output mask. This
                                marks which output values lie within each corresponding triangle.

    Notes
    -----
        Each grid quad is referred to by its four vertices A,B,C,D. The quad is then further
        divided into 2 triangles. Values are interpolated as points lying inside these triangles,
        and then are masked to throw out computed values that lie outside the triangles.
        These grid triangles are referred to as A and D, corresponding to their respective
        vertices.  Furthermore, in order to correctly count the edges, the A- and D-masks are
        slightly different: A uses ≥ & ≤, while D uses > & <. Some rounding errors can occur for
        two neighbouring A- or D-triangles, though these should be virtually nonexistent in real
        world data. However, output coordinates that lie exactly on an edge can sometimes be
        counted double or not at all due to these rounding errors.
    """
    # Dimensions of input and output
    Nx, Ny, Ndim = coords_in.shape
    Mx, My, Mdim = coords_out.shape

    # Reshape input coordinates & values_in, and define A,B,C,D points
    A  = coords_in[:-1, :-1, :].view(Nx-1, Ny-1, 1, 1, 2)
    B  = coords_in[ 1:, :-1, :].view(Nx-1, Ny-1, 1, 1, 2)
    C  = coords_in[:-1,  1:, :].view(Nx-1, Ny-1, 1, 1, 2)
    D  = coords_in[ 1:,  1:, :].view(Nx-1, Ny-1, 1, 1, 2)

    VA = values_in[:-1, :-1, :].view(Nx-1, Ny-1, 1, 1, 1)
    VB = values_in[ 1:, :-1, :].view(Nx-1, Ny-1, 1, 1, 1)
    VC = values_in[:-1,  1:, :].view(Nx-1, Ny-1, 1, 1, 1)
    VD = values_in[ 1:,  1:, :].view(Nx-1, Ny-1, 1, 1, 1)

    # Reshape target coordinates
    T = coords_out.view(1, 1, Mx, My, 2)

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
    b_D = area_para( DT, DC) / denom_D
    c_D = area_para(-DT, DB) / denom_D

    # Compute masks
    bc_A = b_A + c_A
    bc_D = b_D + c_D
    mask_A = logical_and(logical_and(b_A >= 0, c_A >= 0), bc_A <= 1)
    mask_D = logical_and(logical_and(b_D >  0, c_D >  0), bc_D <  1)
    mask = stack((mask_A, mask_D))

    ### Masked could perhaps be applied using masked_select and two for loops
    ### Could be faster for large vectors
    ### Or sparse Tensors

    # Compute interpolated values
    values_out_unfolded = stack(
        ((VA + (VB-VA)*b_A + (VC-VA)*c_A),
         (VD + (VB-VD)*b_D + (VC-VD)*c_D)))

    return values_out_unfolded, mask
