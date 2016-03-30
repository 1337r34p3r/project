clear all;
close all;
clc;

n=5;
m=4;

BASE_URL = 'images\';
EXTENSN  = '.tiff';

finalFeatures = [];
targ = [];

images = {'Happy', 'Sad', 'Fear', 'Surprise'};
for i = 1:m
    NEW_URL = strcat(BASE_URL, images{i}, '\');
    for j = 1:n
        targ = [targ; i];
        img = imread(strcat(NEW_URL, int2str(j), EXTENSN));
        % img = rgb2gray(img); 
        bounds = faceDetect(img);
        img = imcrop(img, bounds);
        img = imresize(img, [100, 100]);

        gaborArray = gaborFilterBank(3,4,39,39);  % Generates the Gabor filter bank
        featureVector = gaborFeatures(img, images{i}, gaborArray,6, 6);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.

        finalFeatures = [finalFeatures; featureVector];
    end
end

disp('==========================================');

p=finalFeatures.'; % transposed feature vector
t=targ.'; % transposed output vector

net=newff(minmax(p),[1 size(t,1)],{'logsig' 'purelin'});  % newff initialize the network  5 is no. of hidden layers keep it 1 or 2 for ur code

% network parameters
net.trainFcn = 'traingdx';
% net.trainFcn = 'trainlm';
net.trainParam.epochs = 5000;
net.trainParam.goal = .001;

net = train(net,p,t); % to train the initialized network and net contains the trained network

Y=sim(net,p); % sim is used to test the network on an image here testing is done on trained images

tSize = size(Y, 2);

for j = 1:5
    for i = j:5:tSize
        if( Y(i) <= 1.5 && Y(i) > 0.5 )
            fprintf('Image %d is a HAPPY IMAGE\n', i);
        elseif( Y(i)<=2.5 && Y(i) >1.5 )
            fprintf('Image %d is a SAD IMAGE\n', i);
        elseif( Y(i)<=3.5 && Y(i) >2.5 )
            fprintf('Image %d is a FEAR IMAGE\n', i);
        elseif( Y(i)<=4.5 && Y(i) >3.5 )
            fprintf('Image %d is a SURPRISE IMAGE\n', i);
        end 
    end
end