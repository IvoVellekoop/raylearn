function [field, phase_step_frames] = phase_step_measurement(cam, camopt, slm, phase_steps)
    % Perform phase stepping measurement
    %
    % Input:
    % cam          Camera object to use
    % camopt       Camera options struct corresponding to cam
    % slm          SLM object to use
    % phase_steps  Number of phase steps
    % 
    % Output:
    % field        Phase stepped field measurement
    
    % Check inputs
    if nargin < 4
        phase_steps = 4;
    end
    
    assert(phase_steps >= 3, 'phase_steps must be at least 3')
    assert(isa(cam, 'Camera'), 'Argument cam must be of class Camera.')
    assert(isa(slm, 'SLM'), 'Argument slm must be of class SLM.')
    
    
    % Initialize frames and phase steps
    phase_patch_id = 1;
    phase_step_frames = zeros(camopt.Width, camopt.Height, phase_steps);
    phase_set = zeros(1,1,phase_steps);
    phase_set(1,1,:) = round( (0:phase_steps-1) * 256/phase_steps);

    % Initialize SLM
    slm.setRect(phase_patch_id, [0 0 1 1]); % Cover full SLM with phase patch
    slm.setData(phase_patch_id, 0)
    slm.update;
    
    % Perform phase steps
    for p = 1:phase_steps
        % Set new value to SLM segment
        slm.setData(phase_patch_id, phase_set(p)); 
        slm.update;

        % record frame
        cam.trigger();
        phase_step_frames(:,:,p) = single(cam.getData());
    end
    
    field = sum(phase_step_frames .* exp(1.0i*phase_set*2*pi/256), 3);
end
