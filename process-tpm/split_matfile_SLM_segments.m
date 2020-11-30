% Split file by SLM segments
function split_matfile_SLM_segments(matfilepath, savedir)
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
    
    [~, infilename, ~] = fileparts(matfilepath);
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
    try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end   % Create savedir if needed

    numchars = fprintf('Loading input file...\n%s.mat', infilename);
    load(matfilepath, 'frames_ft', 'frames_img', 'copt_ft', 'copt_img', 'sopt', 'p'); % Load data
    [~,~,S,~] = size(frames_ft);                                    % Number of segments

    % Loop over segments
    starttime = now;
    for s = 1:S
        % Construct output path
        outfilename = sprintf('%s_%03i.mat', infilename, s);
        savepath = fullfile(savedir, outfilename);
        
        % Print writing progress
        fprintf(repmat('\b', [1 numchars]))
        numchars = fprintf('Output file %i/%i\n%s', s, S, outfilename);
        
        % Extract segment data
        frames_galvo_ft  = frames_ft(:,:,s,:);
        frames_galvo_img = frames_img(:,:,s,:);
        
        % Save the extracted data
        save(savepath, '-v7.3', 's', 'frames_galvo_ft', 'frames_galvo_img', 'copt_ft', 'copt_img', 'sopt', 'p');
    end
end
