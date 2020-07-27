function scan = motor_cam_scan(motor, scanrange, cam, long_exposure_time)
    % motor_cam_scan
    % Use a camera on a 1D stage to make a 3D scan using HDR pictures.
    % 
    % Inputs:
    % motor          Zaber object of the motorized 1D stage.
    % scanrange      1D array containing positions.
    % cam            Camera object of the camera.
    % exposure_times Longest (initial) exposure time. HDR algorithm will
    %                take additional pictures with shorter exposure.
    %
    % Outputs:
    % scan           3D singles array containing the 3D scan.
    

    % Check inputs
    assert(strcmp(class(motor), 'Zaber'), 'Argument motor must be of class Zaber.')
    assert(strcmp(class(cam), 'Camera'), 'Argument cam must be of class Camera.')
    validateattributes(scanrange, {'numeric'}, {'vector'}, '', 'scanrange');
    validateattributes(long_exposure_time, {'numeric'}, {'scalar'}, '', 'long_exposure_time');
    
    % Initialize scan array
    framesize = cam.getSize;
    Nz = numel(scanrange);
    scan = zeros(framesize(1), framesize(2), Nz, 'single');
    
    % Set long exposure time
    cam.setExposureTime(long_exposure_time);
    
    % Move motor to starting position
    motor.moveTo(scanrange(1))
    disp('Moving motor to initial position...')
    while motor.getStatus  % Wait for motor to stop. 0 means idle
        pause(0.1)
    end
    
    starttime = now;
    
    % Perform 3D scan; loop over positions
    for n = 1:Nz
        % Move motor to position
        motor.moveTo(scanrange(n));
        while motor.getStatus  % Wait for motor to stop. 0 means idle
            pause(0.1)
        end

        % Take HDR picture
        new_frame = single(cam.HDR());
        scan(:,:,n) = new_frame;
        
        eta(n, Nz, starttime, 'console', 'Performing 3D scan...', 0);
    end
    
end

