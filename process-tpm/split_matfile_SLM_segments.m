% Split file by SLM segments
function split_matfile_SLM_segments(matfilepath, savedir, consoletext)
    % Split file by SLM segments
    %
    % Some measurement data is packed into one big file. This function splits it up and saves
    % into mat files of manageable size.
    %
    % Input:
    % matfile       Char array or string. Path to mat file to be processed.
    % outfolder     Char array or string. Path to folder where output mat files will be saved.
    % consoletext   Char array or string. Text to display on console during process. Optional.
    %
    % Requires: eta function from the utilities repo.
    
    if nargin < 3
        consoletext = 'Split mat file SLM segments';
    end
    
    [~, filename, ~] = fileparts(matfilepath);
    try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end       % Create savedir if needed

    for plane = ["ft", "img"]                       % Do this for both conjugated camera planes
        frames_plane = strcat('frames_', plane);
        copt_plane = strcat('copt_', plane);
        
        V = load(matfilepath, frames_plane, copt_plane, 'sopt', 'p');   % Load data as struct
        [~,~,S,~] = size(V.(frames_plane));                             % Number of segments
        copt = V.(copt_plane);                                          % Camera options
        sopt = V.sopt;                                                  % SLM options
        p = V.p;                                                        % Other parameters

        % Loop over segments
        starttime = now;
        for s = 1:S
            frames_galvo = V.(frames_plane)(:,:,s,:);                   % Extract segment data
            savepath = fullfile(savedir, ...
                sprintf('%s_%s_%03i.mat', filename, plane, s));         % Output folder
            save(savepath, 'frames_galvo', 's', 'copt', 'sopt', 'p', 'plane'); % Save
            
            % Estimated Time for Arrival
            eta(s, S, starttime, 'console',...
                sprintf('%s\nSplitting file: %s\nPlane: %s\nOutput file: %s',...
                consoletext, matfilepath, plane, savepath), 0);
        end
        
        clear V                                                         % Free up memory instead of downloading more RAM
    end
end
