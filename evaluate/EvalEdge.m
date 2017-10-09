

for i = 1:1
    tic;

    addpath(genpath('edges'));
    resDir = fullfile('./nms_results/');
    fprintf('%s\n',resDir);
    gtDir = 'gts/';
    edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
    figure; edgesEvalPlot(resDir,'ConGAN');

    toc
end