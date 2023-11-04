#include "../iptools/core.h"
#include <string.h>

using namespace std;

// :^) Please let us use python, trying to modify this code base is hell! 
#define ROI_INPUTS	pch = strtok(NULL, " ");\
		int roi_i = atoi(pch); pch = strtok(NULL, " ");\
		int roi_j = atoi(pch);pch = strtok(NULL, " ");\
		int roi_i_size = atoi(pch);pch = strtok(NULL, " ");\
		int roi_j_size = atoi(pch);pch = strtok(NULL, " ");
#define XD pch = strtok(NULL, " ");	// :^))))))))
#define ROI_FNC_INPUTS temp, tgt, atoi(pch), roi_i, roi_j, roi_i_size, roi_j_size
#define MAXLEN 256


int main (int argc, char** argv)
{
	image src, tgt, temp;
	
	FILE *fp;
	char str[MAXLEN];
	char srcfile[MAXLEN];
	char outfile[MAXLEN];
	char *pch;
	if ((fp = fopen(argv[1],"r")) == NULL) {
		fprintf(stderr, "Can't open file: %s\n", argv[1]);
		exit(1);
	}

	while(fgets(str,MAXLEN,fp) != NULL) 
	{
		pch = strtok(str, " ");
		strcpy(srcfile, pch);
		src.read(pch);
		temp.copyImage(src);
		
		XD
		strcpy(outfile, pch);

		XD // XDDDDD

		bool flag = false;
		if (strcmp(pch,"binarize")==0)
		{
			XD
			flag = true;
			utility::binarize(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"add")==0)
		{
			XD
			flag = true;
			utility::addGrey(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"scale")==0)
		{
			XD
			flag = true;
			utility::scale(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"rotate")==0)
		{
			XD
			flag = true;
			utility::rotate(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"addColor")==0)
		{
			XD
			flag = true;
			utility::addColor(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"addBrightness")==0)
		{
			XD
			flag = true;
			utility::addColorBrightness(src, tgt, atoi(pch));
		}
		else if (strcmp(pch,"stretch")==0)
		{
			XD
			int A = atoi(pch);
			XD
			int B = atoi(pch);
			flag = true;
			utility::histogramStretching(src, tgt, A, B);
		}
		//project 2 fncs
		else if(strcmp(pch,"equalizeGrey")==0)
		{
			flag = true;
			utility::equalizeGrey(srcfile, tgt);
		}
		else if(strcmp(pch,"equalizeColor")==0)
		{
			XD
			flag = true;
			utility::equalizeColor(srcfile, tgt, atoi(pch));
		}
		else if(strcmp(pch,"equalizeHSV")==0)
		{
			XD
			flag = true;
			utility::equalizeHSV(srcfile, tgt, atoi(pch));
		}
		else if(strcmp(pch,"equalizeT")==0)
		{
			XD
			flag = true;
			utility::equalizeT(src, srcfile, tgt, atoi(pch));
		}


		if(flag)
		{
			tgt.save(outfile);
			continue;
		}

		int num_of_rois = atoi(pch);

		for(int i = 0; i<num_of_rois; i++)
		{
			ROI_INPUTS
			if (strcmp(pch,"binarize")==0)
			{
				XD
				utility::binarizeWrapper(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"add")==0)
			{
				XD
				utility::addGreyWrapper(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"scale")==0)
			{
				XD
				utility::scaleWrapper(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"rotate")==0)
			{
				XD
				utility::rotateWrapper(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"addColor")==0)
			{	
				XD
				utility::addColorWrapper(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"addBrightness")==0)
			{	
				XD
				utility::addColorBrightnessWrapper(ROI_FNC_INPUTS);
			}
			// Just ROI Fncs
			else if (strcmp(pch,"binarizeROI")==0)
			{
				XD
				utility::binarizeROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"addROI")==0)
			{
				XD
				utility::addGreyROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"scaleROI")==0)
			{
				XD
				utility::scaleROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"rotateROI")==0)
			{
				XD
				utility::rotateROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"addColorROI")==0)
			{	
				XD
				utility::addColorROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"addBrightnessROI")==0)
			{	
				XD
				utility::addColorBrightnessROI(ROI_FNC_INPUTS);
			}
			else if (strcmp(pch,"stretchROI")==0)
			{
				XD
				int A = atoi(pch);
				XD
				int B = atoi(pch);
				utility::histogramStretchingROI(temp, tgt, A, B, roi_i, roi_j, roi_i_size, roi_j_size);
			}
			else if (strcmp(pch,"equalizeGrey")==0)	// only call as last parameter
			{
				utility::equalizeGreyWrapper(temp, tgt, outfile, roi_i, roi_j, roi_i_size, roi_j_size);
			}
			else if (strcmp(pch,"equalizeGreyROI")==0)	// only call as last parameter
			{
				utility::equalizeGreyROI(temp, tgt, outfile, roi_i, roi_j, roi_i_size, roi_j_size);
			}
			else if (strcmp(pch,"equalizeColor")==0)	// only call as last parameter
			{
				XD
				utility::equalizeColorWrapper(temp, tgt, outfile, atoi(pch), roi_i, roi_j, roi_i_size, roi_j_size);
			}
			else if (strcmp(pch,"equalizeColorROI")==0)	// only call as last parameter
			{
				XD
				utility::equalizeColorROI(temp, tgt, outfile, atoi(pch), roi_i, roi_j, roi_i_size, roi_j_size);
			}

			else if (strcmp(pch,"equalizeHSV")==0)	// only call as last parameter
			{
				XD
				utility::equalizeHSVWrapper(temp, tgt, outfile, atoi(pch), roi_i, roi_j, roi_i_size, roi_j_size);
			}
			else if (strcmp(pch,"equalizeHSVROI")==0)	// only call as last parameter
			{
				XD
				utility::equalizeHSVROI(temp, tgt, outfile, atoi(pch), roi_i, roi_j, roi_i_size, roi_j_size);
			}

			temp.copyImage(tgt);
		}
		
		tgt.save(outfile);
	}
	fclose(fp);
	return 0;
}

// 3.
// open CV
// HSI --> HSV
// Follow the right steps, convert to grey levels