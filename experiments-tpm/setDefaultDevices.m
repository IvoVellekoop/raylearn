function active_devices = setDefaultDevices(active_devices)
% function used to set default values to active_devices. Matlab will
% establish a connection with every active device which is set to true
devices = {'slm','cam','cam_slm','pmt_gain','sample_stage'};
default_devices = {true, false, false, false, false};

for i_device = 1:numel(devices)
    field = devices{i_device};
    if ~isfield(active_devices,field)
        active_devices.(field) = default_devices{i_device};
    end
end
end