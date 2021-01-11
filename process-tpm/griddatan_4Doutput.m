function [yi1,yi2,yi3,yi4] = griddatan_4Doutput(x,y1,y2,y3,y4,xi,method,options)
%   This is a version of the griddatan function from Matlab, modified to
%   interpolate 4 output variables Y1,Y2,Y3,Y4, using the same X and XI.
%   In the following description, Y corresponds to any of the variables
%   Y1,Y2,Y3,Y4.
%
%   === Original griddatan description: ===
%GRIDDATAN Data gridding and hyper-surface fitting (dimension >= 2).
%   YI = GRIDDATAN(X,Y,XI) fits a hyper-surface of the form Y = F(X) to the
%   data in the (usually) nonuniformly-spaced vectors (X, Y).  GRIDDATAN
%   interpolates this hyper-surface at the points specified by XI to
%   produce YI. XI can be nonuniform.
%
%   X is of dimension m-by-n, representing m points in n-D space. Y is of
%   dimension m-by-1, representing m values of the hyper-surface F(X). XI
%   is a vector of size p-by-n, representing p points in the n-D space
%   whose surface value is to be fitted. YI is a vector of length p
%   approximating the values F(XI).  The hyper-surface always goes through
%   the data points (X,Y).  XI is usually a uniform grid (as produced by
%   MESHGRID).
%
%   YI = GRIDDATAN(X,Y,XI,METHOD) where METHOD is one of
%       'linear'    - Triangulation-based linear interpolation (default)
%       'nearest'   - Nearest neighbor interpolation
%   defines the type of surface fit to the data. 
%   All the methods are based on a Delaunay triangulation of the data.
%   If METHOD is [], then the default 'linear' method will be used.
%
%   YI = GRIDDATAN(X,Y,XI,METHOD,OPTIONS) specifies a cell array of strings 
%   OPTIONS to be used as options in Qhull via DELAUNAYN. 
%   If OPTIONS is [], the default options will be used.
%   If OPTIONS is {''}, no options will be used, not even the default.
%
%   Example:
%      X = 2*rand(5000,3)-1; Y = sum(X.^2,2);
%      d = -0.8:0.05:0.8; [x0,y0,z0] = meshgrid(d,d,d);
%      XI = [x0(:) y0(:) z0(:)];
%      YI = griddatan(X,Y,XI);
%   Since it is difficult to visualize 4D data sets, use isosurface at 0.8:
%      YI = reshape(YI, size(x0));
%      p = patch(isosurface(x0,y0,z0,YI,0.8));
%      isonormals(x0,y0,z0,YI,p);
%      set(p,'FaceColor','blue','EdgeColor','none');
%      view(3), axis equal, axis off, camlight, lighting phong      
%
%   See also scatteredInterpolant, delaunayTriangulation, DELAUNAYN, MESHGRID.

%   Copyright 1984-2018 The MathWorks, Inc.

if nargin < 6
    error(message('MATLAB:griddatan:NotEnoughInputs'));
end
if ~ismatrix(x) || ~ismatrix(xi)
    error(message('MATLAB:griddatan:HigherDimArray'));
end
[m,n] = size(x);
if m < n+1
    error(message('MATLAB:griddatan:NotEnoughPts'));
end
if (m ~= size(y1,1)) || (m ~= size(y2,1)) || (m ~= size(y3,1)) || (m ~= size(y4,1))
    error(message('MATLAB:griddatan:InputSizeMismatch'));
end
if n <= 1
    error(message('MATLAB:griddatan:XLowColNum'));
end
if any(~isfinite(x),'all')
    error(message('MATLAB:griddatan:CannotTessellateInfOrNaN'));
end
if ( nargin < 7 || isempty(method) )
    method = 'linear';
end
if ~ischar(method) && ~(isstring(method) && isscalar(method))
    error(message('MATLAB:griddatan:InvalidMethod'));
end
if nargin == 8
    if ~iscellstr(options) && ~isstring(options)
        error(message('MATLAB:griddatan:OptsNotStringCell'));
    end
    opt = options;
else
    opt = [];
end

% Average the duplicate points before passing to delaunay
[x,ind1,ind2] = unique(x,'rows');
if size(x,1) < m
    warning(message('MATLAB:griddatan:DuplicateDataPoints'));
    y1 = accumarray(ind2,y1,[size(x,1),1],@mean);
    y2 = accumarray(ind2,y2,[size(x,1),1],@mean);
    y3 = accumarray(ind2,y3,[size(x,1),1],@mean);
    y4 = accumarray(ind2,y4,[size(x,1),1],@mean);
else
    y1 = y1(ind1);
    y2 = y2(ind1);
    y3 = y3(ind1);
    y4 = y4(ind1);
end

switch lower(method)
    case 'linear'
        [yi1,yi2,yi3,yi4] = linear(x,y1,y2,y3,y4,xi,opt);
    case 'nearest'
        [yi1,yi2,yi3,yi4] = nearest(x,y1,y2,y3,y4,xi,opt);
    otherwise
        error(message('MATLAB:griddatan:UnknownMethod'));
end

%------------------------------------------------------------
function [zi1,zi2,zi3,zi4] = linear(x,y1,y2,y3,y4,xi,opt)
%LINEAR Triangle-based linear interpolation

%   Reference: David F. Watson, "Contouring: A guide
%   to the analysis and display of spacial data", Pergamon, 1994.

% Triangularize the data
if isempty(opt)
  tri = delaunayn(x);
else
  tri = delaunayn(x,opt);
end
if isempty(tri)
  warning(message('MATLAB:griddatan:CannotTriangulate'));
  zi1 = NaN*zeros(size(xi));
  zi2 = zi1;
  zi3 = zi1;
  zi4 = zi1;
  return
end

% Find the nearest triangle (t)
[t,p] = tsearchn(x,tri,xi);

m1 = size(xi,1);
zi1 = NaN*zeros(m1,1);
zi2 = zi1;
zi3 = zi1;
zi4 = zi1;

for i = 1:m1
  if ~isnan(t(i))
     zi1(i) = p(i,:)*y1(tri(t(i),:));
     zi2(i) = p(i,:)*y2(tri(t(i),:));
     zi3(i) = p(i,:)*y3(tri(t(i),:));
     zi4(i) = p(i,:)*y4(tri(t(i),:));
  end
end
%------------------------------------------------------------
function [zi1,zi2,zi3,zi4] = nearest(x,y1,y2,y3,y4,xi,opt)
%NEAREST Triangle-based nearest neightbor interpolation

%   Reference: David F. Watson, "Contouring: A guide
%   to the analysis and display of spacial data", Pergamon, 1994.

% Triangularize the data
if isempty(opt)
  tri = delaunayn(x);
else
  tri = delaunayn(x,opt);
end
if isempty(tri) 
  warning(message('MATLAB:griddatan:CannotTriangulate'));
  zi1 = NaN(size(xi));
  zi2 = zi1;
  zi3 = zi1;
  zi4 = zi1;
  return
end

% Find the nearest vertex
k = dsearchn(x,tri,xi);

zi1 = k;
zi2 = k;
zi3 = k;
zi4 = k;
d = find(isfinite(k));
zi1(d) = y1(k(d));
zi2(d) = y2(k(d));
zi3(d) = y3(k(d));
zi4(d) = y4(k(d));
