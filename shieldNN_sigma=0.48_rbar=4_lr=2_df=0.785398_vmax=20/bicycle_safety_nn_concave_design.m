close all
%% Paremeters and Definitions

% Bicycle parameters:
lr = 2; % total vehicle length = 2*lr = 0.46 
% NOTE: THIS IS THE ~MAX VEHICLE LENGTH FOR SUBSEQUENT PARAMETERS!!
%   To allow a longer vehicle you may:
%       Increase \sigma;
%       INcrease rBar; or 
%       INcrease deltaFMax
deltaFMax = pi/4; % maximum steering angle: pi/4 = 45 degrees
betaMax = atan(0.5*tan(deltaFMax));
vmax = 20; % Maximum linear velocity

% Barrier function parameters:
rBar = 4; % Mimum physical distance to origin
sigma = 0.48; % Scaling factor that determines minimum dist. to 
		  % origin at xi = +-pi (see definition of 'barrier' below)

% The barrier function h:
h = @(v,r,xi) ( ...
		(sigma.*cos(xi/2) + 1 - sigma)/rBar - 1./r ...
	);

% The Lie derivative of h along trajectories of the (radial) bicycle dynamics:
Lh = @(v,r,xi,beta) (...
		v*( ...
			(1./r.^2).*cos(xi - beta) + ...
			sigma.*sin(xi/2).*sin(xi - beta)./(2*rBar*r) + ...
			sigma.*sin(xi/2).*sin(beta)./(2*rBar*lr) ...
		)...
	);

% 'barrier' computes h(v,r,xi) == 0 as a funciton of xi; i.e. the actual
% **physical barrier**.
barrier = @(xi) ( ...
		rBar./(sigma.*cos(xi/2) + 1 - sigma) ...
	);

K = max(1,1/rBar) * ( (sigma/(2*rBar)) + 2 + 1);

% (Relaxation) class K function for CBF (this is just a linear function,
% scaled to account for the maximum velocity):
% alpha = @(x) ( ...
% 		K*vmax*sigma*x./(2*rBar*lr) ...
% 	);
alpha = @(x) ( ...
		K*vmax*x ...
	);

% Derivative along constant-level curves (from Mathematica):
%     (Note: r = 0 is the actual barrier, i.e. along the function 'barrier'
%     from above.)
derivZBF = @(xi,beta,r,v,sigma,lr,rBar,vmax,K) ((v).^(-1).*(-(sin(beta - xi)/(r + rBar/(1 - sigma + sigma.*cos(xi/2))).^2) + (sigma.*(cos(beta)/lr - cos(beta - xi)/(r + rBar/(1 - sigma + sigma.*cos(xi/2)))).*sin(xi/2))/(2.*rBar)).^(-1).*((K.*r.*vmax.*sigma.*(1 - sigma + sigma.*cos(xi/2)).*(r + 2.*rBar - r.*sigma + r.*sigma.*cos(xi/2)).*(rBar).^(-1).*(r + rBar - r.*sigma + r.*sigma.*cos(xi/2)).^(-2).*sin(xi/2))/2 - (v.*(sigma.*cos(xi/2).*(lr).^(-1).*(rBar).^(-1).*sin(beta) + 4.*(r + rBar/(1 - sigma + sigma.*cos(xi/2))).^(-2).*sin(beta - xi) - sigma.*cos(xi/2).*(rBar).^(-1).*(r + rBar/(1 - sigma + sigma.*cos(xi/2))).^(-1).*sin(beta - xi) + (sigma).^(2).*(r + rBar - r.*sigma + r.*sigma.*cos(xi/2)).^(-2).*(sin(xi/2)).^(2).*sin(beta - xi) - 4.*rBar.*sigma.*cos(beta - xi).*(1 - sigma + sigma.*cos(xi/2)).*(r + rBar - r.*sigma + r.*sigma.*cos(xi/2)).^(-3).*sin(xi/2) + 2.*sigma.*cos(beta - xi).*(rBar).^(-1).*(r + rBar/(1 - sigma + sigma.*cos(xi/2))).^(-1).*sin(xi/2)))/4));



%% Plotting code for verification purposes
figure;
[XI,BETA] = meshgrid(linspace(-pi,pi,200),linspace(-betaMax,betaMax,200));
surf(XI,BETA,Lh(2,barrier(XI),XI,BETA)+alpha(h(2,barrier(XI),XI)));
hold on;
surf(XI,BETA,0*XI)
hold off;

figure;
[XI,BETA] = meshgrid(linspace(-pi,pi,200),linspace(-betaMax,betaMax,200));
surf(XI,BETA,Lh(1,barrier(XI)+1,XI,BETA)+alpha(h(1,barrier(XI)+1,XI)));
hold on;
surf(XI,BETA,0*XI)
hold off;

%% Train ReLU approximation safety network
safetyConst = 0.01;
numSegments = 10;

layers = [
	sequenceInputLayer(1)
	fullyConnectedLayer(numSegments)
	reluLayer
	fullyConnectedLayer(1)
	regressionLayer
];

% Find a radius after which all controls are admissible:
freeControlThreshold = 10;
for rr=0:0.1:5
	if Lh(vmax,barrier(pi)+rr,pi,-betaMax)+alpha(h(vmax,barrier(pi)+rr,pi)) >= 0
		freeControlThreshold = rr;
		break;
	end
end

% Radius thresholds:
%	Each element should be interpreted as a 'physical' barrier of the form:
%		barrier(xi) + threshold(index).
%	(0 and freeControlThreshold are required.)
radiusThresholds = [0 0.25 0.5 freeControlThreshold];

% Each column will contain the training data for a single ReLU network:

betaVals = zeros(numSegments,length(radiusThresholds)-1);
S1 = zeros(numSegments,length(radiusThresholds)-1);
xI = zeros(numSegments,length(radiusThresholds)-1);
yI = zeros(numSegments,length(radiusThresholds)-1);
xInt = zeros(numSegments-1,length(radiusThresholds)-1);
yInt = zeros(numSegments-1,length(radiusThresholds)-1);
W1 = zeros(numSegments,length(radiusThresholds)-1);
b1 = zeros(numSegments,length(radiusThresholds)-1);
W2 = zeros(numSegments,length(radiusThresholds)-1);
b2 = zeros(numSegments,length(radiusThresholds)-1);
betaValsOffset = zeros(numSegments,length(radiusThresholds)-1);
repThresh = zeros(length(radiusThresholds)-1,1);

for ii = 1:length(radiusThresholds)-1
	% Find the first angle where there is a constraint on the control:
	startAngle = fzero( ...
		@(XI) (Lh(vmax,barrier(XI)+radiusThresholds(ii),XI,-betaMax)+alpha(h(vmax,barrier(XI)+radiusThresholds(ii),XI))), ...
		pi ...
	);
    xi = linspace(startAngle,pi,numSegments+1);
	for jj = 2:length(xi)
        betaVals(jj-1,ii) = ...
            fzero( ...
                @(beta) ( ...
                    Lh(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj),beta) + ...
                    alpha(h(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj))) ...
                ), ...
                -betaMax ...
            );
        S1(jj-1,ii) = derivZBF(xi(jj),betaVals(jj-1,ii),radiusThresholds(ii), vmax, sigma, lr, rBar, vmax, K);
        
        eps1 = -sign(S1(jj-1,ii))*sqrt( safetyConst^2 / ( 1 + 1/(S1(jj-1,ii)^2) ) );
        xI(jj-1,ii) = xi(jj) - eps1;
        yI(jj-1,ii) = -(1/S1(jj-1,ii)) * eps1 + betaVals(jj-1,ii);
        if jj == 2
            %in = (-betaMax - betaVals(jj-1,ii))/S1(jj-1,ii) + xI(jj-1,ii);
            W1(jj-1,ii) = abs(S1(jj-1,ii));
            b1(jj-1,ii) = -W1(jj-1,ii)*xI(jj-1,ii) + yI(jj-1,ii) + betaMax;
            W2(jj-1,ii) = sign(S1(jj-1,ii));
            b2(jj-1,ii) = -betaMax;
        end
        if jj >= 3
           xInt(jj-2,ii) = (yI(jj-2,ii) - yI(jj-1,ii) + S1(jj-1,ii)*xI(jj-1,ii) - S1(jj-2,ii)*xI(jj-2,ii) )/ ( S1(jj-1,ii) - S1(jj-2,ii));
           yInt(jj-2,ii) = S1(jj-1,ii) * ( xInt(jj-2,ii) - xI(jj-1,ii) ) + yI(jj-1,ii);
           W1(jj-1,ii) = abs(S1(jj-1,ii) - S1(jj-2,ii));
           b1(jj-1,ii) = -W1(jj-1,ii) * xInt(jj-2,ii);
           W2(jj-1,ii) = sign(S1(jj-1,ii) - S1(jj-2,ii));
           b2(jj-1,ii) = 0;
        end
    end
end

testFn = @(xi,ii) ( W2(:,ii)'*max( W1(:,ii)*xi + b1(:,ii), 0) + b2(1,ii) );
tt = linspace(0,pi,1000);
yy = testFn(tt,1);

trainedNets = {};
% options = trainingOptions('adam', ...
%     'InitialLearnRate',0.001, ...
% 	'Shuffle','every-epoch', ...
% 	'MiniBatchSize',2000, ...
%     'Verbose',false, ... 
% 	'MaxEpochs',500, ...
% 	'Plots','none'); % 'training-progress'
for ii = 1:length(radiusThresholds)-1
    layers(2).Weights = W1(:,ii);
    layers(2).Bias = b1(:,ii);
    layers(4).Weights = W2(:,ii)';
    layers(4).Bias = [b2(1,ii)];
	trainedNets{ii} = assembleNetwork(layers);
end
% 
%% Add betaMax clipping to NN:

for ii = 1:length(radiusThresholds)-1
    % Add a clipping layer to force the NN output into [-betaMax,betaMax]
    trainedNets{ii} = assembleNetwork([ ...
        trainedNets{ii}.Layers(1:length(trainedNets{ii}.Layers)-1); ...
        fullyConnectedLayer(1, ...
            'Weights',[1], ...
            'Bias',[betaMax] ...
        ); ...
        reluLayer; ...
        fullyConnectedLayer(1, ...
            'Weights',[-1], ...
            'Bias',[2*betaMax] ...
        ); ...
        reluLayer; ...
        fullyConnectedLayer(1, ...
            'Weights',[-1], ...
            'Bias',[betaMax] ...
        ); ...
        trainedNets{ii}.Layers(length(trainedNets{ii}.Layers)) ...
    ]);
end


%% Plot ReLU approximation compared to actual \beta threshold

meshPoints = 2000;
xi = linspace(-pi,pi,meshPoints);
betaThresh = zeros(length(xi),length(radiusThresholds)-1);
repThresh = zeros(length(radiusThresholds)-1,1);

for ii = 1:length(radiusThresholds)-1
	% Find the first angle where there is a constraint on the control:
	startAngle = fzero( ...
		@(XI) (Lh(vmax,barrier(XI)+radiusThresholds(ii),XI,-betaMax)+alpha(h(vmax,barrier(XI)+radiusThresholds(ii),XI))), ...
		pi ...
	);
	for jj = 1:length(xi)
		if xi(jj) >= startAngle
			if repThresh(ii) == 0
				repThresh(ii) = jj;
			end
			betaThresh(jj,ii) = ...
				fzero( ...
					@(beta) ( ...
						Lh(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj),beta) + ...
						alpha(h(vmax,barrier(xi(jj))+radiusThresholds(ii),xi(jj))) ...
					), ...
					-betaMax ...
				);
        else
            betaThresh(jj,ii) = -betaMax;
		end
    end
end

figure
legText = {};
hold on
for ii = 1:length(radiusThresholds)-1
    plot( ...
        xi,betaThresh(:,ii),  ...
        xi,predict(trainedNets{ii},xi),  ...
        -xi,-fliplr(betaThresh(:,ii)),  ...
        xi,-predict(trainedNets{ii},-xi) ...
    );
    legText{4*ii - 3} = strcat('Lh(v,\xi,r) + \alpha(h(v,\xi,r)) == 0 for r = barrier(\xi) + ',  string(radiusThresholds(ii)));
    legText{4*ii - 2} = strcat('ReLU Approximation for r = barrier(\xi) + ',string(radiusThresholds(ii)));
    legText{4*ii - 1} = strcat('Lh(v,\xi,r) + \alpha(h(v,\xi,r)) == 0 for r = barrier(\xi) + ',  string(radiusThresholds(ii)));
    legText{4*ii    } = strcat('ReLU Approximation for r = barrier(\xi) + ',string(radiusThresholds(ii)));
end
plot(xi,betaMax + 0*xi,'g')
legText{length(legText)+1} = '\beta_{max}';
hold off
xlabel('\xi (radians)')
ylabel('\beta = tan^{-1}( 1/2 \cdot tan({\delta}_f))')
legend(legText, 'Location','northwest')


%% Augment NN to get high/low value filtering:

finalNets = trainedNets;

for ii = 1:length(radiusThresholds)-1
    % Add a clipping layer to force the NN output into [-betaMax,betaMax]
    path1 = finalNets{ii}.Layers(2:length(finalNets{ii}.Layers)-1);
    for kk = 1:length(path1)
        path1(kk).Name = strcat(path1(kk).Name,'_XiPath');
    end
    path1(length(path1)).Name = 'XiPathOut';
    path1 = [fullyConnectedLayer(1,'Name','XiPathIn','Weights',[1 0],'Bias',[0]); path1];
    
    path2 = finalNets{ii}.Layers(2:length(finalNets{ii}.Layers)-1);
    for kk = 1:length(path2)
        path2(kk).Name = strcat(path2(kk).Name,'_NegXiPath');
    end
    path2(length(path2)).Name = 'NegXiPathOut';
    path2 = [fullyConnectedLayer(1,'Name','NegXiPathIn','Weights',[-1 0],'Bias',[0]); path2];
    lgraph = layerGraph(path1);
    lgraph = addLayers(lgraph,path2);
    lgraph = addLayers(lgraph,fullyConnectedLayer(1,'Name','BetaPathIn','Weights',[0 1],'Bias',[0]));
    lgraph = addLayers(lgraph,sequenceInputLayer(2,'Name','InputLayer'));
    lgraph = connectLayers(lgraph,'InputLayer','XiPathIn');
    lgraph = connectLayers(lgraph,'InputLayer','NegXiPathIn');
    lgraph = connectLayers(lgraph,'InputLayer','BetaPathIn');
    lgraph = addLayers(lgraph,concatenationLayer(1,3,'Name','ConcatenationLayer'));
    lgraph = connectLayers(lgraph,'XiPathOut','ConcatenationLayer/in1');
    lgraph = connectLayers(lgraph,'NegXiPathOut','ConcatenationLayer/in2');
    lgraph = connectLayers(lgraph,'BetaPathIn','ConcatenationLayer/in3');
    lgraph = addLayers(lgraph, [...
        fullyConnectedLayer(4, 'Name', 'PathDifferences1', ...
            'Weights', [-1 0 1; 1 0 0; 0 -1 0; 0 0 1], ...
            'Bias', [0;betaMax;betaMax;betaMax] ...
        ) ...
        reluLayer('Name','relu_Differences1') ...
        fullyConnectedLayer(4, 'Name', 'PathDifferences2', ...
            'Weights', [-1 -1 1 0; 0 1 0 0; 0 0 1 0; 0 0 0 1], ...
            'Bias', [0;0;0;0] ...
        ) ...
        reluLayer('Name','relu_Differences2') ...
        fullyConnectedLayer(2, 'Name', 'PathDifferences3', ...
            'Weights', [-1 0 1 0; 0 0 0 1], ...
            'Bias', [-betaMax;-betaMax] ...
        )
    ]);
    lgraph = connectLayers(lgraph,'ConcatenationLayer','PathDifferences1');
    lgraph = addLayers(lgraph,regressionLayer('Name','FinalOutputLayer'));
    lgraph = connectLayers(lgraph,'PathDifferences3','FinalOutputLayer');
    finalNets{ii} = assembleNetwork(lgraph);
    exportONNXNetwork(finalNets{ii},strcat(num2str(ii), '.onnx'))
end


%% Final output:

% Safety "control-filter" NNs are stored in finalNets
%
% For each threshold in radiusThresholds, there is one safety
% "control-filter" network in finalNets that is valid for *all* radii
% larger than the barrier plus the associated radius threshold.
%
% That is:
%
% finalNet{ii} is valid for all r >= barrier(\xi) + radiusThresholds(ii)
%
% This can be relaxed to:
%
% finalNet{ii} is valid for all r >= barrier(PI) + radiusThresholds(ii)
%
% (so that the threshold is independent of \xi).
%
% NOTE: there is no need to test the *lower* bound associated with
% finalNet{1} because the barrier will ensure that it remains applicable.
for file_index = 1:3
    filename = strcat(num2str(file_index) , '.onnx')
    filename2 = strcat('tf', num2str(file_index) )
    exportONNXNetwork(finalNets{1,file_index},filename)
    exportNetworkToTensorFlow(finalNets{1,file_index},filename2)
end





