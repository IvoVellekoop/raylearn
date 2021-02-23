% Split file by SLM segments
function split_matfile_SLM_segments(matfilepath, savedir, chunksize_bytes)
    % Split file by SLM segments
    %
    % Some measurement data is packed into one big file. This function splits it up and saves
    % into mat files of manageable size.
    %
    % Input:
    % matfilepath   Char array or string. Path to mat file to be processed.
    % savedir       Char array or string. Path to folder where output mat files will be saved.
    % chunksize     Positive Integer. The big arrays will be loaded in chunks of this size.
    
    [~, infilename, ~] = fileparts(matfilepath);
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
    try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end   % Create savedir if needed

    numchars = 0;
    load(matfilepath, 'copt_ft', 'copt_img', 'sopt', 'p');          % Load metadata
    inputfile = matfile(matfilepath);                               % Create matfile object
    [Nx,Ny,S,G] = size(inputfile, 'frames_img');                    % Number of segments
    
    if G < 1
        % Note: For some corrupt files, an empty dimension might be reported
        warning('Empty array in %s\nCorrupt file?', matfilepath);
        return
    end

    totalsize_bytes = 2 * Nx*Ny*G*S * 4;                            % Total size of the arrays
    chunksize = max(1, floor(S*chunksize_bytes/totalsize_bytes));   % Size of an array chunk
    numchunks = ceil(S / chunksize);                                % Number of chunks
    s = 1;                                                          % Segment number
    
    % Loop over chunks
    for c = 1:numchunks
        % Prepare chunk loading
        chunk = s:min(s+chunksize-1, S);                            % Index of chunk end
        thischunksize = length(chunk);                              % Size of this chunk
        bytestr = bytes2str(2*Nx*Ny*G*thischunksize*4, '%.0f');     % Formatted char array
        numchars = numchars + fprintf('Loading chunk (%s) %i/%i...\n', bytestr, c, numchunks);
        
        % Load data chunk
        frames_chunk_ft  = inputfile.frames_ft(:, :, chunk, 1:G);   % Load chunk ft frames
        frames_chunk_img = inputfile.frames_img(:,:, chunk, 1:G);   % Load chunk img frames
        
        
        for sc = 1:thischunksize    
            % Construct output path
            outfilename = sprintf('%s_%03i.mat', infilename, s);
            savepath = fullfile(savedir, outfilename);

            % Print progress
            fprintf(repmat('\b', [1 numchars]))
            numchars = fprintf('Output file %i/%i\n%s\n', s, S, outfilename);

            % Extract segment data
            frames_slmslice_ft  = frames_chunk_ft(:,:,sc,:);
            frames_slmslice_img = frames_chunk_img(:,:,sc,:);

            % Save the extracted data and metadata
            save(savepath, '-v7.3', 's', 'frames_slmslice_ft', 'frames_slmslice_img',...
                'copt_ft', 'copt_img', 'sopt', 'p');
            
            s = s+1;
        end
    end
end