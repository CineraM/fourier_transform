#include "utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MAXRGB 255
#define MINRGB 0

using namespace cv;
using namespace std;
image temp1, temp2;
string TEMP_PGM = "temp.pgm";
string TEMP_PPM = "temp.ppm";
string TEMP_PNG = "temp.png";

cv::Mat computeDFT(Mat image);
void fftShift(Mat magI);
void lowpassFilter(const cv::Mat &dft_Filter, int distance);

int utility::checkValue(int value)
{
	if (value > MAXRGB)
		return MAXRGB;
	if (value < MINRGB)
		return MINRGB;
	return value;
}

bool check_roi(int i, int j, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	if(i >= roi_i && i <= roi_i+roi_i_size-1)
		if(j >= roi_j && j <= roi_j+roi_j_size-1) 
			return 1;
	return 0;
}


void utility::roi(image &src, image &tgt, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	tgt.resize(roi_i_size, roi_j_size);
	int rows = src.getNumberOfRows();
	int cols = src.getNumberOfColumns();

	for (int i=0; i<roi_i_size; i++)
	{
		for (int j=0; j<roi_j_size; j++)
		{
			if((i+roi_i) >= rows || (j+roi_j) >= cols)
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, MINRGB);
			}
			else
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, src.getPixel(i+roi_i, j+roi_j, channel));
			}
		}
	}

}


void utility::mergeRoi(image &src, image &roi,image &tgt, int roi_i, int roi_j, 
int roi_i_size, int roi_j_size)
{
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			
			if(check_roi(i, j, roi_i, roi_j, roi_i_size, roi_j_size))
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, roi.getPixel(i-roi_i, j-roi_j, channel) ); 
			}
			else
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, src.getPixel(i,j, channel) );
			}
				
		}
	}
}


void utility::binarize(image &src, image &tgt, int threshold)
{
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			if (src.getPixel(i,j) < threshold)
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, MINRGB);
			}
			else
			{
				for(int channel = 0; channel<3; channel++)
					tgt.setPixel(i, j, channel, MAXRGB);
			}
		}
	}
}


void utility::addGrey(image &src, image &tgt, int value)
{
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			for(int channel = 0; channel<3; channel++)
				tgt.setPixel(i, j, channel, checkValue(src.getPixel(i, j, channel)+value));
		}
}


void utility::scale(image &src, image &tgt, float ratio)
{
	if(ratio < 10) ratio = 10;
	if(ratio > 20) ratio = 20;
	ratio/=10;
	int rows = (int)((float)src.getNumberOfRows() * ratio);
	int cols  = (int)((float)src.getNumberOfColumns() * ratio);
	tgt.resize(rows, cols);
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{	
			/* Map the pixel of new image back to original image */
			int i2 = (int)floor((float)i/ratio);
			int j2 = (int)floor((float)j/ratio);
			// Directly copy the value 
			for(int channel = 0; channel<3; channel++)
				tgt.setPixel(i, j, channel, checkValue(src.getPixel(i2, j2, channel)));
			}
	}
}


void utility::rotate(image &src, image &tgt, int angle)
{
	int rows = src.getNumberOfRows();
	int cols = src.getNumberOfColumns();


	tgt.resize(rows, cols);
	if(angle == 90 || angle == 270)
		tgt.resize(cols, rows);
		
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{
			int new_i, new_j;
			if(angle == 90)
			{
				new_i = j;
				new_j = rows - 1 - i;
			}
			else if(angle == 180)
			{
				new_i = rows - 1 - i;
				new_j = cols - 1 - j;	
			}
			else if(angle == 270)
			{
				new_i = cols - 1 - j;
				new_j = i;	
			}
			
			for(int channel = 0; channel<3; channel++)
				tgt.setPixel(new_i, new_j, channel, src.getPixel(i, j, channel));
		}
	}
}

void utility::addColor(image &src, image &tgt, double value)
{
	if(value < 10) value = 10;
	if(value > 20) value = 20;

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			for(int channel = 0; channel<3; channel++)
				tgt.setPixel(i, j, channel, checkValue( src.getPixel(i, j, channel)*(value/10) ));
		}
}

void utility::addColorBrightness(image &src, image &tgt, int value)
{
	if(value < -50) value = -50;
	if(value > 50) value = 50;

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			for(int channel = 0; channel<3; channel++)
				tgt.setPixel(i, j, channel, checkValue(src.getPixel(i, j, channel) + value));
		}
}


void utility::binarizeWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	binarize(temp1, temp2, fnc_input);
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::addGreyWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addGrey(temp1, temp2, fnc_input);
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::scaleWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	scale(temp1, tgt, fnc_input);
}


void utility::rotateWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	rotate(temp1, temp2, fnc_input);
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::addColorWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addColor(temp1, temp2, fnc_input);
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);

}


void utility::addColorBrightnessWrapper(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addColorBrightness(temp1, temp2, fnc_input);
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}

// ONLY ROI FUNCTIONS

void utility::binarizeROI(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	binarize(temp1, tgt, fnc_input);
}


void utility::addGreyROI(WRAPPER_PARAMS)
{
	
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addGrey(temp1, tgt, fnc_input);
}


void utility::scaleROI(WRAPPER_PARAMS)
{
	scale(temp1, tgt, fnc_input);
}


void utility::rotateROI(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	rotate(temp1, tgt, fnc_input);
}


void utility::addColorROI(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addColor(temp1, tgt, fnc_input);
}


void utility::addColorBrightnessROI(WRAPPER_PARAMS)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	addColorBrightness(temp1, tgt, fnc_input);
}

// Project 2 functions
void utility::histogramStretching(image &src, image &tgt, int A, int B)
{
	int min_pixel = 255;
	int max_pixel = 0;

	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			int pixel_val = src.getPixel(i, j);
			if(pixel_val > max_pixel) max_pixel = pixel_val;
			if(pixel_val < min_pixel) min_pixel = pixel_val;
		}
	}
			
	int C = min_pixel*1.05;
	int D = max_pixel*0.95;

	unordered_map<int, int> new_pixel_vals;
	
	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			int old_val = src.getPixel(i, j);
			if(new_pixel_vals.find(old_val) == new_pixel_vals.end())
			{
				double new_val = (old_val-C) * (B-A) / (D-C) + A;
				// double new_val = ((old_val - C) / (D - C)) * (B - A) + A; 
				new_pixel_vals[old_val] = checkValue(new_val);
				tgt.setPixel(i, j, new_pixel_vals[old_val]);	
			}
			else
				tgt.setPixel(i, j, new_pixel_vals[old_val]);
		}
	}

}


void utility::histogramStretchingROI(image &src, image &tgt, int A, int B, 
	int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	histogramStretching(temp1, tgt, A, B);
}


void save_to_tgt(string src, image &tgt)
{
	char* new_src = new char[src.length() + 1];  // +1 for null-terminator
	strcpy(new_src, src.c_str());
	tgt.read(new_src);
}


void utility::equalizeGrey(string src, image &tgt)
{
	Mat src_image = imread(src, IMREAD_GRAYSCALE);
    // Apply histogram equalization
    Mat equalized_image;
    equalizeHist(src_image, equalized_image);
    // save img to new file
    imwrite(TEMP_PGM, equalized_image);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}


void utility::equalizeGreyROI(image &src, image &tgt, string tgtfile,
	int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	equalizeGrey(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}

void utility::equalizeGreyWrapper(image &src, image &tgt, string tgtfile,
	int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	equalizeGrey(TEMP_PGM, temp2);
	std::remove(TEMP_PGM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::equalizeColor(string src, image &tgt, int RGB_VAL)
{
	vector<Mat> channels;
	Mat src_image = imread(src, IMREAD_COLOR);
    split(src_image, channels);

	if(RGB_VAL == 3)
	{
		for (int i = 0; i < 3; ++i)
			equalizeHist(channels[i], channels[i]);
	}
	else
		equalizeHist(channels[RGB_VAL], channels[RGB_VAL]);

	Mat equalized_image;
    merge(channels, equalized_image);
	// color conversion to save img
	cvtColor(equalized_image, equalized_image, COLOR_RGB2BGR);
	imwrite(TEMP_PPM, equalized_image);
	save_to_tgt(TEMP_PPM, tgt);
	std::remove(TEMP_PPM.c_str());
}

void utility::equalizeColorROI(image &src, image &tgt, string tgtfile,
	int RGB_VAL, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	equalizeColor(TEMP_PPM, tgt, RGB_VAL);
	std::remove(TEMP_PPM.c_str());
}

void utility::equalizeColorWrapper(image &src, image &tgt, string tgtfile,
	int RGB_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	equalizeColor(TEMP_PPM, temp2, RGB_VAL);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::equalizeHSV(string src, image &tgt, int HSV_VAL)
{
    Mat input_image = imread(src, IMREAD_COLOR);

    Mat hsv_image;
    cvtColor(input_image, hsv_image, COLOR_BGR2HSV);

    std::vector<Mat> channels;
    split(hsv_image, channels);

	if(HSV_VAL == 3)
	{
		for (int i = 0; i < 3; ++i)
			equalizeHist(channels[i], channels[i]);
	}
	else
		equalizeHist(channels[HSV_VAL], channels[HSV_VAL]);

    merge(channels, hsv_image);
    Mat equalized_bgr;
    cvtColor(hsv_image, equalized_bgr, COLOR_HSV2BGR);

	imwrite(TEMP_PPM, equalized_bgr);
	save_to_tgt(TEMP_PPM, tgt);
	std::remove(TEMP_PPM.c_str());
}

void utility::equalizeHSVROI(image &src, image &tgt, string tgtfile,
	int HSV_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	equalizeHSV(TEMP_PPM, tgt, HSV_VAL);
	std::remove(TEMP_PPM.c_str());
}

void utility::equalizeHSVWrapper(image &src, image &tgt, string tgtfile,
	int HSV_VAL ,int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	equalizeHSV(TEMP_PPM, temp2, HSV_VAL);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


// extra credit question
void utility::equalizeT(image &src, string srcfile, image &tgt, int threshold)
{	
	image t_image;
	string temp_file = "temp_file.pgm";

	t_image.resize(src.getNumberOfRows(), src.getNumberOfColumns());

	std::vector<std::vector<bool>> t_mapping(src.getNumberOfRows(), 
		std::vector<bool>(src.getNumberOfColumns(), false));

	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			if (src.getPixel(i,j) < threshold)
			{
				t_mapping[i][j] = true;
				t_image.setPixel(i, j, src.getPixel(i, j));
			}
			else
			{
				t_image.setPixel(i, j, threshold);
			}
				
		}
	}
	t_image.save("debug_threshold_pixel_selection.pgm");
	t_image.save(temp_file.c_str());
	
	Mat src_image = imread(temp_file, IMREAD_GRAYSCALE);
    Mat equalized_image;
    equalizeHist(src_image, equalized_image);

    imwrite("debug_threshold_pixel_selection2.pgm", equalized_image);
    imwrite(temp_file, equalized_image);

	tgt.resize(src.getNumberOfRows(), src.getNumberOfColumns());
	image eq_temp_image;

	char* temp_file_dir = new char[temp_file.length() + 1];  // +1 for null-terminator
	strcpy(temp_file_dir, temp_file.c_str());

	eq_temp_image.read(temp_file_dir);

	for (int i=0; i<src.getNumberOfRows(); i++)
	{
		for (int j=0; j<src.getNumberOfColumns(); j++)
		{
			if (t_mapping[i][j])
				tgt.setPixel(i, j, eq_temp_image.getPixel(i, j));
			else
				tgt.setPixel(i, j, src.getPixel(i, j));
		}
	}

	std::remove(temp_file.c_str());
}

// reference opencv documentation
// https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
void utility::fourierTrans(string src, image &tgt)
{
	Mat src_image = imread(src, IMREAD_GRAYSCALE);

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( src_image.rows );
    int n = getOptimalDFTSize( src_image.cols ); // on the border add zero values
    copyMakeBorder(src_image, padded, 0, m - src_image.rows, 0, n - src_image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magnitudeImage = planes[0];
    magnitudeImage += Scalar::all(1);                    // switch to logarithmic scale
    log(magnitudeImage, magnitudeImage);
    // crop the spectrum, if it has an odd number of rows or columns
    magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magnitudeImage.cols/2;
    int cy = magnitudeImage.rows/2;
    Mat q0(magnitudeImage, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magnitudeImage, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magnitudeImage, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magnitudeImage, magnitudeImage, 0, 255, NORM_MINMAX); 
	// Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

    imwrite(TEMP_PGM, magnitudeImage);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}

void utility::fourierTransROI(image &src, image &tgt, string tgtfile,
	int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	fourierTrans(TEMP_PGM, tgt);
	// std::remove(TEMP_PGM.c_str());
}

void utility::fourierTransWrapper(image &src, image &tgt, string tgtfile,
	int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	fourierTrans(TEMP_PPM, temp2);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


// Compute the Discrete fourier transform
cv::Mat computeDFT(Mat image) {
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);         // Add to the expanded another plane with zeros
    dft(complex, complex, DFT_COMPLEX_OUTPUT);  // fourier transform
    return complex;
}

// https://datahacker.rs/opencv-discrete-fourier-transform-part2/#id1
// create an ideal low pass filter
void lowpassFilter(const cv::Mat &dft_Filter, int distance)
{
    Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
    double radius;

    for(int i = 0; i < dft_Filter.rows; i++)
    {
        for(int j = 0; j < dft_Filter.cols; j++)
        {
            radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
            if(radius>distance){
                tmp.at<float>(i,j) = (float)0;
            }else{
                tmp.at<float>(i,j) = (float)1;
            }

        }
    }

    Mat toMerge[] = {tmp, tmp};
    merge(toMerge, 2, dft_Filter);
}

void fftShift(Mat magI) 
{
    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void highpassFilter(Mat &dft_Filter, int distance)
{
    Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
    double radius;

    for(int i = 0; i < dft_Filter.rows; i++)
    {
        for(int j = 0; j < dft_Filter.cols; j++)
        {
            radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
            if(radius>distance){
                tmp.at<float>(i,j) = (float)1;
            }else{
                tmp.at<float>(i,j) = (float)0;
            }

        }
    }

    Mat toMerge[] = {tmp, tmp};
    merge(toMerge, 2, dft_Filter);
}

// Function to apply inverse Fourier transform to an image
void utility::lowPass(string src, image &tgt, int radius) 
{
    cv::Mat img, complexImg, filter, filterOutput, imgOutput, planes[2];

    img = imread(src, 0);

    complexImg = computeDFT(img);
    filter = complexImg.clone();

    lowpassFilter(filter, radius); // create an ideal low pass filter

    fftShift(complexImg); // rearrage quadrants
    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
    fftShift(complexImg); // rearrage quadrants

    // compute inverse
    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, NORM_MINMAX);

    split(filter, planes);
    normalize(planes[1], filterOutput, 0, 1, NORM_MINMAX);

    // imwrite("Filter.pgm", filterOutput*255);			// Debug
    // imwrite("Low_pass_filter.pgm", imgOutput*255);	// Debug
    imwrite(TEMP_PGM, imgOutput*255);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}

void utility::lowPassROI(image &src, image &tgt, string tgtfile,
	int radius, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	lowPass(TEMP_PGM, tgt, radius);
	// std::remove(TEMP_PGM.c_str());
}

void utility::lowPassWrapper(image &src, image &tgt, string tgtfile,
	int radius, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	lowPass(TEMP_PPM, temp2, radius);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::highPass(string src, image &tgt, int radius) 
{
    cv::Mat img, complexImg, filter, filterOutput, imgOutput, planes[2];

    img = imread(src, 0);

    complexImg = computeDFT(img);
    filter = complexImg.clone();

    highpassFilter(filter, radius); // create an ideal high pass filter

    fftShift(complexImg); // rearrage quadrants
    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
    fftShift(complexImg); // rearrage quadrants

    // compute inverse
    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, NORM_MINMAX);

    split(filter, planes);
    normalize(planes[1], filterOutput, 0, 1, NORM_MINMAX);

    // imwrite("Filter.pgm", filterOutput*255);			// Debug
    // imwrite("Low_pass_filter.pgm", imgOutput*255);	// Debug
    imwrite(TEMP_PGM, imgOutput*255);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}


void utility::highPassROI(image &src, image &tgt, string tgtfile,
	int radius, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	highPass(TEMP_PGM, tgt, radius);
	// std::remove(TEMP_PGM.c_str());
}

void utility::highPassWrapper(image &src, image &tgt, string tgtfile,
	int radius, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	highPass(TEMP_PPM, temp2, radius);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


void utility::edgeSharp(string src, image &tgt, int T) 
{
    Mat gray = imread(src, 0);

    // Apply Gaussian blur to the input image
    Mat blurred;
    GaussianBlur(gray, blurred, Size(0, 0), 1.5);

    // Calculate the difference between the input and blurred images
    Mat highFreq = gray - blurred;

    // Multiply high-frequency components by the user-provided constant
    Mat sharpened = gray + T * highFreq;

	imwrite(TEMP_PGM, sharpened);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}

void utility::edgeSharpROI(image &src, image &tgt, string tgtfile,
	int T, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PGM.c_str());
	edgeSharp(TEMP_PGM, tgt, T);
	// std::remove(TEMP_PGM.c_str());
}

void utility::edgeSharpWrapper(image &src, image &tgt, string tgtfile,
	int T, int roi_i, int roi_j, int roi_i_size, int roi_j_size)
{
	roi(src, temp1, roi_i, roi_j, roi_i_size, roi_j_size);
	temp1.save(TEMP_PPM.c_str());
	edgeSharp(TEMP_PPM, temp2, T);
	std::remove(TEMP_PPM.c_str());
	mergeRoi(src, temp2, tgt, roi_i, roi_j, roi_i_size, roi_j_size);
}


// ec fncs
void utility::bandStop(string src, image &tgt, int r1, int r2) 
{
    cv::Mat img, complexImg, high, low, filter, filterOutput, imgOutput, planes[2];

    img = imread(src, 0);

    complexImg = computeDFT(img);
    low = complexImg.clone();
	high = complexImg.clone();

    lowpassFilter(high, r1);
    highpassFilter(low, r2);
	subtract(high, low, filter);

    fftShift(complexImg); // rearrage quadrants
    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
    fftShift(complexImg); // rearrage quadrants

    // compute inverse
    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, NORM_MINMAX);

    split(filter, planes);
    normalize(planes[1], filterOutput, 0, 1, NORM_MINMAX);

    imwrite("Filter.pgm", filterOutput*255);			// Debug
    // imwrite("Low_pass_filter.pgm", imgOutput*255);	// Debug
    imwrite(TEMP_PGM, imgOutput*255);
	save_to_tgt(TEMP_PGM, tgt);
	std::remove(TEMP_PGM.c_str());
}



    int radius = 30;
    cv::Mat img, complexImg, filter, filterOutput, imgOutput, planes[2];

    img = imread("truck.jpg", 1); // Load color image in BGR format (use 1 instead of 0)
    if (img.empty()) {
        return -1;
    }

    // Convert the color image to HSV
    cv::Mat hsvImage;
    cvtColor(img, hsvImage, CV_BGR2HSV);

    // Process the desired HSV channel (e.g., S channel)
    int channelToProcess = 1; // Change this to process H, S, or V channel
    cv::Mat channel = hsvImage.clone();
    extractChannel(hsvImage, channelToProcess, channel);

    complexImg = computeDFT(channel);
    filter = complexImg.clone();

    highpassFilter(filter, radius); // create an ideal high pass filter

    fftShift(complexImg); // rearrange quadrants
    mulSpectrums(complexImg, filter, complexImg, 0); // multiply 2 spectrums
    fftShift(complexImg); // rearrange quadrants

    // Compute inverse
    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

    split(filter, planes);
    normalize(planes[1], filterOutput, 0, 1, CV_MINMAX);

    imshow("Input image", img);