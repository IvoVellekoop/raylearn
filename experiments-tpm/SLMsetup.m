function [slm, sopt] = SLMsetup(lambda, sopt)
%%% Function used to set up the Meadowlark SLM in the two-photon
%%% microscope.
%%%
%%% input variables
%%% lambda: laser wavefront
%%% z_slm:  SLM axial position (conjugation depth)
%%% sopt:   slm options
%%%
%%% output variables
%%% slm:    slm handle
%%% N:      number of controllable SLM segments within illuminated part

% find correct displayPort based on frame rate
displays = SLM.enumerateDevices;
for d = 1:numel(displays)                   % note starts at display 2 (1 is preview window)
    if displays(d).CurrentRefreshRate == 31 % Meadowlark SLM runs at 31 fps
        sopt.displayPort = d-1;
        break;
    end
end

% create SLM handle and set background to zero value
slm = SLM(sopt.displayPort);
slm.setData(0,0); slm.update;

% set correct LUT for given lambda
sopt.alpha = 0.2623*lambda - 23.33;          % SLM gray value corresponding with 2pi
GV_skip = 2;                            % first two gray values are skipped because they introduce a 0.4 rad jump
LUT = [GV_skip*ones(1,GV_skip),GV_skip:255]*sopt.alpha/256;
slm.setLUT(LUT);                        % set look up table for specific SLM at specific wavelength

% set SLM timing parameters
slm.t_idle = 2;                         % time for SLM to start responding to new signal (in number of vsyncs)
slm.t_settle = 2;                       % time for SLM settle to new value (in number of vsyncs)

% set beam center coordinates and diameter
sopt.cx = 0.027;       %%%% should these go in a separate variable? they're not slm options
sopt.cy = 0.01;
sopt.diameter = 0.95;

% define SLM axes (center of beam is [0,0]);
Npix = slm.getSize();
sopt.y = (-Npix(1)/2+1:Npix(1)/2)/Npix(1);
sopt.x = (-Npix(2)/2+1:Npix(2)/2)/Npix(1);
end