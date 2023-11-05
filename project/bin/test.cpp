#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void saveFourierTransform(const cv::Mat& inputImage, const std::string& outputFilename) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(inputImage.rows);
    int n = cv::getOptimalDFTSize(inputImage.cols);
    cv::copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage);

    // Compute the magnitude of the Fourier transform
    cv::split(complexImage, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magnitudeImage = planes[0];

    // Normalize the magnitude image for saving
    cv::normalize(magnitudeImage, magnitudeImage, 0, 255, cv::NORM_MINMAX);

    // Save the magnitude image
    cv::imwrite(outputFilename, magnitudeImage);
}

void saveInverseFourierTransform(const cv::Mat& inputImage, const std::string& outputFilename) {
    cv::Mat inverseTransform;
    cv::dft(inputImage, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, inverseTransform, 0, 255, cv::NORM_MINMAX);

    // Save the inverse Fourier transform
    cv::imwrite(outputFilename, inverseTransform);
}

int main() {
    // Load the input PGM image
    cv::Mat inputImage = cv::imread("portrait.pgm", cv::IMREAD_GRAYSCALE);

    // Compute the Fourier transform and save it
    saveFourierTransform(inputImage, "fourier_transform.png");

    // Compute the inverse Fourier transform and save it
    saveInverseFourierTransform(inputImage, "inverse_fourier_transform.png");

    return 0;
}
