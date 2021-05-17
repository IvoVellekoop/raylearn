function [x_sample_mm, y_sample_mm, x_obj, y_obj, z_obj] = galvoslm_to_rays(Ux_galvo, Uy_galvo, x_SLM, y_SLM)
    % Galvo SLM to rays
    % Converts Galvo voltages and SLM relative coords to ray coordinates
    % Ray position in mm and direction coordinates
    %
    % Input:
    % %%%
    %
    % Output:
    % %%%
    %
    
    % Constants
    obj_tubelength = 200;       % Objective standard tubelength (mm)
    obj_magnification = 16;     % Objective magnification
    f1 = 100;                   % Lens 1 focal length (mm)
    f2 = 100;                   % Lens 2 focal length (mm)
    f3 = 400;                   % Lens 3 focal length (mm)
    f4 = 400;                   % Lens 4 focal length (mm)
    f5 = 200;                   % Lens 5 focal length (mm)
    f7 = 400;                   % Lens 7 focal length (mm)
    c_galvo = 0.5;                 % Galvo tilt constant (volt/degree)
    SLM_height = 10.7;          % Physical height of SLM pixel array (mm)
    
    % Computed constants
    fobj = obj_tubelength / obj_magnification;
    M_GM_sample = (f3*f5*fobj) / (f2*f4*f7);
    
    % Compute position
    x_sample_mm = M_GM_sample * f1 * tand(Ux_galvo / c_galvo);
    y_sample_mm = M_GM_sample * f1 * tand(Uy_galvo / c_galvo);
    
    % Compute direction unit vector
    x_obj = x_SLM * f7/f5 * SLM_height;
    y_obj = y_SLM * f7/f5 * SLM_height;
    z_obj = fobj;
end
