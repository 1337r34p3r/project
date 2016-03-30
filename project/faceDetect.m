function [ bounds ] = faceDetect( img )
%FACEDETECT Summary of this function goes here
%   Detailed explanation goes here

%Detect objects using Viola-Jones Algorithm

%To detect Face
FDetect = vision.CascadeObjectDetector;

%Returns Bounding Box values based on number of objects
bounds = step(FDetect,img);

end

