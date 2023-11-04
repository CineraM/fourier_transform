#ifndef UTILITY_H
#define UTILITY_H

#include "../image/image.h"
#include <sstream>
#include <string>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <cstdio>

#define WRAPPER_PARAMS  image &src, image &tgt, int fnc_input, int roi_i, int roi_j, int roi_i_size, int roi_j_size

class utility
{
	public:
		utility();
		virtual ~utility();

		static std::string intToString(int number);
		static int checkValue(int value);

		static void roi(image &src, image &tgt, 
		int roi_i, int roi_j, int roi_i_size, int roi_j_size);

		static void mergeRoi(image &src, image &roi,image &tgt, int roi_i, int roi_j, 
		int roi_i_size, int roi_j_size);

		static void binarize(image &src, image &tgt, int threshold);
		static void addGrey(image &src, image &tgt, int value);
		static void scale(image &src, image &tgt, float ratio);
		static void rotate(image &src, image &tgt, int angle);
		static void addColorBrightness(image &src, image &tgt, int value);
		static void addColor(image &src, image &tgt, double value);

		static void binarizeWrapper(WRAPPER_PARAMS);
		static void addGreyWrapper(WRAPPER_PARAMS);
		static void scaleWrapper(WRAPPER_PARAMS);
		static void rotateWrapper(WRAPPER_PARAMS);
		static void addColorWrapper(WRAPPER_PARAMS);
		static void addColorBrightnessWrapper(WRAPPER_PARAMS);
		
		static void binarizeROI(WRAPPER_PARAMS);
		static void addGreyROI(WRAPPER_PARAMS);
		static void scaleROI(WRAPPER_PARAMS);
		static void rotateROI(WRAPPER_PARAMS);
		static void addColorROI(WRAPPER_PARAMS);
		static void addColorBrightnessROI(WRAPPER_PARAMS);
		
		// Project 2 functions
		static void histogramStretching(image &src, image &tgt, int A, int B);

		static void histogramStretchingROI(image &src, image &tgt, int A, int B, 
			int roi_i, int roi_j, int roi_i_size, int roi_j_size);


		static void equalizeGrey(string src, image &tgt);
		static void equalizeGreyROI(image &src, image &tgt, string tgtfile,
			int roi_i, int roi_j, int roi_i_size, int roi_j_size);
		static void equalizeGreyWrapper(image &src, image &tgt, string tgtfile,
			int roi_i, int roi_j, int roi_i_size, int roi_j_size);

		static void equalizeColor(string src, image &tgt, int RGB_VAL);
		static void equalizeColorROI(image &src, image &tgt, string tgtfile,
			int RGB_VAL, int roi_i, int roi_j, int roi_i_size, int roi_j_size);
		static void equalizeColorWrapper(image &src, image &tgt, string tgtfile,
			int RGB_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size);

		static void equalizeHSV(string src, image &tgt, int HSV_VAL);
		static void equalizeHSVROI(image &src, image &tgt, string tgtfile,
			int HSV_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size);
		static void equalizeHSVWrapper(image &src, image &tgt, string tgtfile,
			int HSV_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size);

		// EC
		static void equalizeT(image &src, string srcfile, image &tgt, int threshold);
};

#endif

