%% Test to fit 4D polynomial, to model camera coords to SLM/Galvo coords
close all; clc; clear

%% Create basis polynomial functions
% Create coordinate arrays for x,y,kx,ky
Nxy = 9;
Nk  = 7;
Ntot = Nxy*Nxy*Nk*Nk;

x  = linspace(-1, 1, Nxy)';
y  = linspace(-1, 1, Nxy)';
kx = linspace(-1, 1, Nk)';
ky = linspace(-1, 1, Nk)';

% [Xo, Yo, KXo, KYo] = meshgrid(xo, yo, kxo, kyo);


% Create 2D arrays containing 1, x, x², ..., 1, y, y², ..., 1, kx, kx², ...
npowers = 3;                                    % Polynomial powers (including 0)

powers = 0:(npowers-1);                         % Array containing all powers
xpowers = x.^powers;
ypowers = y.^powers;
kxpowers = kx.^powers;
kypowers = ky.^powers;
xykxky_powers = zeros(Nxy, Nxy, Nk, Nk, npowers.^4);     % Initialize basis

% Loop over all powers in x,y,kx,ky
m = 1;
for xpow = xpowers
    for ypow = ypowers
        for kxpow = kxpowers
            for kypow = kypowers
                % Add 4D polynomial to set of basis functions
                xykxky_powers(:, :, :, :, m) = xpow .* ypow' .* permute(kxpow, [3 2 1]) .* permute(kypow, [4 3 2 1]);
                m = m+1;
            end
        end
    end
end

xykxky_cam_basis = reshape(xykxky_powers, Ntot, npowers^4);    % Reshape 4D basis set to matrix

%% Create test data X
% Generate random coefficients for creating test data
a = randn(npowers.^4, 1) ./ linspace(1, 3, npowers.^4)'.^2;
noisefactor = 0.15;

% Create test data from coefficients
Xgtlin = xykxky_cam_basis * a;                      % Ground truth, linear array
Xgalvolin = Xgtlin + noisefactor * randn(Ntot,1);   % 'Measured' = ground truth + noise, linear array
Xgt = reshape(Xgtlin, Nxy, Nxy, Nk, Nk);            % Ground truth
Xgalvo = reshape(Xgalvolin, Nxy, Nxy, Nk, Nk);      % 'Measured' = ground truth + noise


%% Fit data
cf = xykxky_cam_basis \ Xgalvolin;                  % Compute coefficients
Xfitlin = xykxky_cam_basis * cf;                    % Compute fit
Xfit = reshape(Xfitlin, Nxy, Nxy, Nk, Nk);          % Reshape to 4D array


%% Plot slices of fit data vs ground truth
figure(1)
plot(Xgtlin(1:80), 'or')
hold on
plot(Xfitlin(1:80), '+b')
title('First 80 values')
xlabel('index')
ylabel('X_{galvo} value')
legend('Ground Truth', 'Fit')
hold off

%% Plot bar graph of coefficients
figure(2)
bar([a cf])
legend('Ground Truth', 'Fit')
RMSE = mean((a - cf).^2);
nRMSE = RMSE / mean(a.^2);
title(sprintf('noisefactor = %.2f | 4D polynomial degree %i\nCoefficients: RMSE = %.2g  nRMSE = %.3g', noisefactor, npowers-1, RMSE, nRMSE))
