function tiff = load_tiff(tiffpath, calibration_data, loadstr, create_corrected)
    % load_tiff
    % Load scanimage tiff file containing a stack of frames
    %
    % Input:
    % tiffpath              Path to tiff file
    % calibrationdata       Struct containing loaded calibration data
    % loadstr               String to be displayed during loading of frames
    % create_corrected      Create copy of framestack_raw data with corrected offset (median is
    %                       used to find noise peak) and negative values clipped to 0.
    %
    % Output:
    % tiff                  Struct containing 3D voxel data and metadata
    % tiff.framestack_raw   Raw data as 3D array
    % tiff.framestack       Data as 3D array, with offset corrected and negative values clipped to
    %                       0. Only created if create_corrected == true.
    % tiff.offset           Offset applied to tiff.framestack. Based on median. Only created if
    %                       create_corrected == true.
    % tiff.zstep_um         Z-step size in micrometers
    % tiff.num_of_frames    Number of frames in the framestack (a.k.a. z-slices)
    % tiff.zoom             Zoom setting of measurement
    % tiff.pixels_per_um    Number of pixels per micrometer in x- and y-direction.
    % tiff.FOV_um           Field of View in micrometers. (Center to center first to last pixel.)
    %                       Frames are assumed to be square.
    % tiff.stack_depth_um   Depth of the framestack in micrometers. (Center to center first to last
    %                       voxel.)

    arguments
        tiffpath {mustBeTextScalar}
        calibration_data struct
        loadstr {mustBeTextScalar} = "Loading frames..."
        create_corrected (1, 1) logical = true
    end

    % Retrieve zstep from filename
    zstep_um_cellstr = regexp(tiffpath, 'zstep([.\d]+)um', 'tokens');
    assert(length(zstep_um_cellstr), "zstep not defined in filename")
    tiff.zstep_um = str2double(zstep_um_cellstr{1}{1});
    
    % Load frame info
    tifinfo = imfinfo(tiffpath);         % Get info about image
    tiff.num_of_frames = length(tifinfo);    % Number of frames
    tiff.framestack_raw = zeros(tifinfo(1).Width, tifinfo(1).Height, tiff.num_of_frames);

    assert(tifinfo(1).Width == tifinfo(1).Height, 'Frames are not square.')
    
    % Retrieve zoom
    tif_scanimage_metadata = tifinfo.Artist;
    zoom_cellstr = regexp(tif_scanimage_metadata, '"scanZoomFactor": (\d+),', 'tokens');
    tiff.zoom = str2double(zoom_cellstr{1}{1});
    
    % Compute resolution, FOV and stack depth of this measurement
    pixels_per_um_calibration = sqrt(sum(calibration_data.M(:,1).^2));
    scan_resolution_factor = tifinfo(1).Width / calibration_data.scanimage_details.pixels_per_line;
    tiff.pixels_per_um = pixels_per_um_calibration * scan_resolution_factor * tiff.zoom;

    tiff.FOV_um = (tifinfo(1).Width  - 1) ./ tiff.pixels_per_um;            % Field of View (x&y)
    tiff.stack_depth_um = (tiff.num_of_frames - 1) .* tiff.zstep_um;        % Depth in z

    % Define x-, y-, z-range
    tiff.xrange = [-tiff.FOV_um/2 tiff.FOV_um/2];
    tiff.yrange = [-tiff.FOV_um/2 tiff.FOV_um/2];
    tiff.zrange = [-tiff.stack_depth_um/2 tiff.stack_depth_um/2];
    tiff.dimdata = {tiff.xrange, tiff.yrange, tiff.zrange};
    
    % Loop over frames and read
    starttime = now;
    for n = 1:tiff.num_of_frames
        tiff.framestack_raw(:,:,n) = imread(tiffpath, n);
        eta(n, tiff.num_of_frames, starttime, 'cmd', loadstr, 10);
    end
    
    % Create corrected copy of the data with median subtracted, and negatives clipped to 0
    if create_corrected
        tiff.offset = median(tiff.framestack_raw(:));
        tiff.framestack = tiff.framestack_raw - tiff.offset;
        tiff.framestack(tiff.framestack < 0) = 0;
    end

end