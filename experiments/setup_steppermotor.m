%% Connect to stepper motors
disp('Setting up stepper motors...')

init_sample_motors = false;             % Initialise sample motors as well

comport = 'COM4';
baudrate = 9600;
cam_mot_id = 6210;                      % ID of the Zaber T-LSM050A
cam_mot_index = 2;                      % Daisy chain index
mot_cam_target_speed = 4000;

mot_cam = Zaber(Serial(comport, baudrate), cam_mot_index);
mot_cam.setTargetSpeed(mot_cam_target_speed);

assert(mot_cam.ID == cam_mot_id,...     % Check for motor type ID
    sprintf(['Expected ID %i at index %i (%s), but found ID %i.\n',...
             'Renumbering the devices might be required. This can be',...
             'done by sending command #2 to all daisy chained devices,',...
             'or through the Zaber Console software.'],...
    cam_mot_id, cam_mot_index, comport, mot_cam.ID))


if init_sample_motors
    % Initialise sample motors as well
    s = Serial('COM3',9600);
    mot1 = Zaber(s);
    mot1.setTargetSpeed(500);

    s = Serial('COM4',9600);
    mot2 = Zaber(s);
    mot2.setTargetSpeed(500);
end


% move motor to starting position
% start_pos = 10000;      % starting position (in um)
% step = 1;               % step size of motor (in um)
% mot1.moveTo(start_pos);
disp('Done setting up stepper motors.')
