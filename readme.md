# HW 2 Grey Level and Color - Parameters


## How to compile

     cd /code/iptools/;
     make clean;
     make;
     cd ../project/;
     make clean;
     make;
     cd bin/;
     ./iptool parameters.txt;



## Single operations

- source_img   outut_img  operation operation input (if needed)

Example:

- baboon.pgm baboon_1.pgm  stretch 0 255

## ROI parameters

Multiple sequential operations must have ROI parameters

- i j i_size j_size

## ROI Operations

Common operations merge back to the source image. So if bianrize was performed over an ROI it will merge it back to the original image. If you just want the ROI as the output Image, extend the operation paramter with ROI.
example:

- binarizeROI: Will only produce the ROI image
- binarize: will produce the original image with the modified ROI merged

## Multiple operations

- source_img   output_img  numberOfPperations operation ROI_INPUTS operation operation input

Example:

- baboon.pgm baboon_3_1.pgm 2 0 0 400 500 stretchROI 0 255 140 0 0 250 250 equalizeGreyROI
