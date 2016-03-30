clear all
clc

img = imread('..\images\1.tiff');
bounds = faceDetect(img);
img = imcrop(img, bounds);
figure, title('Face Cropped'), imshow(img);