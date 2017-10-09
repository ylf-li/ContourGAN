addpath(genpath('edges'))
root = fullfile('./rootmat/');
root_res = fullfile('./nms_results/');

files = dir(root);
files = files(3:end,:);
filenames = cell(1,size(files, 1));
res_names = cell(1,size(files, 1));
for i = 1:size(files, 1),
    filenames{i} = files(i).name;
    res_names{i} = [files(i).name(1:end-4), '.png'];
end

for i = 1:size(filenames,2)

    edge=imread([root,filenames{i}]);
    edge = 1-single(edge)/255;
    [Ox, Oy] = gradient2(convTri(edge, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    edge = edgesNmsMex(edge, O, 2, 5, 1.01, 8);
    imwrite(edge,[root_res, res_names{i}]);
end