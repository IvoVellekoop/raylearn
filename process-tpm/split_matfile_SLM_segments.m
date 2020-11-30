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

    numchars = 0;
    load(matfilepath, 'copt_ft', 'copt_img', 'sopt', 'p');          % Load metadata
    inputfile = matfile(matfilepath);                               % Create matfile object
    [~,~,S,G] = size(inputfile, 'frames_img');                      % Number of segments
    
    if G < 1
        % Note: For some corrupt files, an empty dimension might be reported
        warning('Empty array in %s\nCorrupt file?', matfilepath);
        return
    end
    
    % Loop over segments
    for s = 1:S
        % Construct output path
        outfilename = sprintf('%s_%03i.mat', infilename, s);
        savepath = fullfile(savedir, outfilename);
        
        % Print progress
        fprintf(repmat('\b', [1 numchars]))
        numchars = fprintf('Output file %i/%i\n%s\n', s, S, outfilename);
        
        % Extract segment data
        frames_slmslice_ft  = inputfile.frames_ft(:,:,s,1:G);
        frames_slmslice_img = inputfile.frames_img(:,:,s,1:G);
        
        % Save the extracted data and metadata
        save(savepath, '-v7.3', 's', 'frames_slmslice_ft', 'frames_slmslice_img',...
            'copt_ft', 'copt_img', 'sopt', 'p');
    end
end

