function [rects, N, inside_circle_mask, X, Y] = BlockedCircleSegments(Nd, D, cx, cy, segmentwidth, segmentheight, vis)
% Computes rectangles of segments of a block geometry. Each
% segment is a square region with its center inside the circle
% with diameter D and center cx, cy.
% Segments will be numbered like this:
%      .....
%      ..4..
%      .135.
%      ..2..
%      .....
%
% === Usage: ===
% [rects, N] = BlockedCircleSegments(Nd, D, cx, cy, vis,
% relativesegmentsize)
% Input arguments D, cx, cy, vis, segmentwidth and segmentheight
% are optional. Defaults are given below.
% BlockedCircleSegments will mimic functionality of
% SLM.setBlockGeometry, however, rather than directly setting
% the SLM geometry, it returns 4-element rect vectors which can
% be used as input for SLM.setRect. This would yield
% functionality similar to using SLM.setBlockGeometry, but with
% 'transparency' for undefined segments, instead of forcibly
% overwriting pixels of the layer below.
%
% === Input: ===
% Nd:   Number of elements along diameter.
% D:    Diameter of circle. Default: 1
% cx:   Center x coordinate. Default: 0
% cy:   Center y coordinate. Default: 0
% segmentwidth, segmentheight: Size of the segments in relative coordinates. Default: 0.05.
% vis:  Visualisation flag (0 or 1). If set to 1, visualise
%       numbered segments in a plot. For debugging only.
%       Default: 0
%
% === Output: ===
% N:                    Number of segments
% rects:                N-by-4-matrix. Each row represents a segment
%                       rectangle: [center_x, center_y, width, height]
% inside_circle_mask:   


% === Check if input is valid ===
narginchk(1,7)      % Check if 1 <= nargin <= 7

% Defaults
if nargin < 2
    D = 1;          % Diameter of circle
end
if nargin < 3
    cx = 0;         % Center x coordinate
end
if nargin < 4
    cy = 0;         % Center y coordinate
end
if nargin < 5       % Segment width
    segmentwidth = 0.05;
end
if nargin < 6       % Segment height
    segmentwidth = 0.05;
end
if nargin < 7
    vis = 0;        % Visualisation flag
end

% Check argument types and sizes
validateattributes(D, {'numeric'}, {'size', [1 1]})
validateattributes(cx, {'numeric'}, {'size', [1 1]})
validateattributes(cy, {'numeric'}, {'size', [1 1]})
validateattributes(vis, {'numeric'}, {'size', [1 1]})
validateattributes(segmentwidth, {'numeric'}, {'size', [1 1]})
validateattributes(segmentheight, {'numeric'}, {'size', [1 1]})


% === Compute segment geometries ===

% Compute constants
R = D/2;            % Radius of circular area (normalized -> 0.5)
couterx = R - segmentwidth/2;   % Distance from center to outer segment center
coutery = R - segmentheight/2;  % Distance from center to outer segment center

% Construct x and y coordinates on grid
x  = linspace(cx-couterx, cx+couterx, Nd);
y  = linspace(cy-coutery, cy+coutery, Nd);
[X, Y] = meshgrid(x, y);

% Compute radii of gridpoints
r2 = (X - cx).^2 + (Y - cy).^2;

% Check which gridpoints lie inside circle
inside_circle_mask = (r2 <= R^2);

% Select gridpoints inside circle and create 1D array
Xin = X(inside_circle_mask);
Yin = Y(inside_circle_mask);
rcx = Xin(:);
rcy = Yin(:);

% Create 1D arrays for
N = length(rcx);
rw = segmentwidth * ones(N,1);
rh = segmentheight * ones(N,1);

% Package into collection of rectangle vectors
rects = [rcx rcy rw rh];


% === Visualise the numbered segments in a plot if requested ===
if vis
    % Spawn figure and draw border rectangle
    figure;
    rectangle('Position', [-0.8 -0.5 1.6 1])
    hold on
    
    % Loop over segments
    for n = 1:N
        % Compute corner coordinates
        xcorn = rcx(n) - rw(n)/2;
        ycorn = rcy(n) - rh(n)/2;
        
        % Draw rectangle and annotate with segment number
        rectangle('Position', [xcorn ycorn rw(n) rh(n)], 'FaceColor', [.85 .85 .85])
        text(rcx(n), rcy(n), num2str(n), 'HorizontalAlignment', 'center')
    end
    hold off
    
    % Give title including parameters and x-/y-labels
    title(sprintf('BlockRects\nNd = %i | D = %.2f | cx = %.2f | cy = %.2f', Nd, D, cx, cy))
    xlabel('X'); ylabel('Y')
end
end