# HW 3 Fourier Transform and Frequency filteting - Parameters

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

- baboon.pgm baboon_1.pgm  lowPass 30

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

- portrait.pgm sample_output.pgm 3 0 0 500 500 lowPassROI 15 0 0 340 340 edgeSharpROI 60 0 0 300 300 rotateROI 270
