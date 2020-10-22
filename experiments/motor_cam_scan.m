function [scan, thresholds] = motor_cam_scan(motor, scanrange, cam, long_exposure_time, dark_frame, min_exposure_time_us)
    % motor_cam_scan
    % Use a camera on a 1D stage to make a 3D scan using HDR pictures.
    % 
    % Inputs:
    % motor          Zaber object of the motorized 1D stage.
    % scanrange      1D array containing positions (um).
    % cam            Camera object of the camera.
    % exposure_times Longest (initial) exposure time (us). HDR algorithm 
    %                will take additional pictures with shorter exposure.
    % dark_frame     Dark frame intensity image.
    %
    % Outputs:
    % scan           3D singles array containing the 3D scan.
    

    % Check inputs
    assert(isa(motor, 'Zaber'), 'Argument motor must be of class Zaber.')
    assert(isa(cam, 'Camera'), 'Argument cam must be of class Camera.')
    validateattributes(scanrange, {'numeric'}, {'vector'}, '', 'scanrange');
    validateattributes(long_exposure_time, {'numeric'}, {'scalar'}, '', 'long_exposure_time');
    
    waittime = 0.05;     % Time to wait for camera pole to settle in seconds
    
    % Initialize scan array
    framesize = cam.getSize;
    Nz = numel(scanrange);
    scan = zeros(framesize(1), framesize(2), Nz, 'single');
    thresholds = zeros(Nz, 'single');
    
    % Set long exposure time
    cam.setExposureTime(long_exposure_time);
    
    % Move motor to starting position
    disp('Moving motor to initial position...')
    try
        motor.moveTo(scanrange(1))    %%%%% sometimes gives error of no data returned
    catch err
        warning(err.message);
    end
    while motor.getStatus  % Wait for motor to stop. 0 means idle
        pause(0.1)
    end
    
    starttime = now;
    
    % Perform 3D scan; loop over positions
    for n = 1:Nz
        % Move motor to position
        motor.moveTo(scanrange(n));
        
        wait_starttime = now;
        fprintf('Waiting for motor...       \n')
        motorstatus = 20;
        while motorstatus  % Wait for motor to stop. 0 means idle
            pause(0.05)
            waittime_motor = (now-wait_starttime)*86400;
            if waittime_motor > 1
                fprintf('\b\b\b\b\b\b\b\b%3.0f sec\n', waittime_motor)
            end
            motorstatus = motor.getStatus;
        end
        
        eta(n, Nz, starttime, 'console', 'Performing 3D scan...', 0);
        pause(waittime)    % Wait for a bit for the camera pole to settle

        % Take HDR picture
        try
            [new_frame, threshold_HDR] = cam.HDR(dark_frame, min_exposure_time_us);
        catch err
            % Sometimes the grab times out, so retry
            pause(0.5)
            warning(err.message)
            [new_frame, threshold_HDR] = cam.HDR(dark_frame, min_exposure_time_us);
        end
        scan(:,:,n) = single(new_frame);
        thresholds(n) = threshold_HDR;
    end
    
end

