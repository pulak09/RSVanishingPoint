/*
 * Project:  vanishingPoint
 *
 * File:     main.cpp
 *
 * Contents: Creation, initialisation and usage of MSAC object
 *           for vanishing point estimation in images or videos
 *
 * Author:   Pulak Purkaitg <pulak.isi@gmail.com>
 *
 * Github:   Rolling shutter correction in manhattan world
 * 
 * Acknowledgement: Marcos Nieto <marcos.nieto.doncel@gmail.com> for the backbone code (www.marcosnieto.net/vanishingPoint) 
 */


#ifdef WIN32
#include <windows.h>
#endif
#include <iostream>
#ifdef linux
	#include <stdio.h>
#endif

#define USE_PPHT
#define MAX_NUM_LINES	1500

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/line_descriptor.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include <ctime> 
#include "MSAC.h"
#include "lsd.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cv.h>

using namespace cv::line_descriptor;
using namespace std;
using namespace cv;

void help()
{
	 cout << "/*\n"
         << " **************************************************************************************************\n"
		 << " * Vanishing point detection using Hough and MSAC \n"
         << " * ----------------------------------------------------\n"		 
		 << " * \n"
		 << " * Author: Pulak Purkait\n"
		 << " * pulak.isi@gmail.com\n"
		 << " * \n"
		 << " **************************************************************************************************\n"
		 << " * \n"
		 << " * Usage: \n"		 		 
		 << " *		-video		# Specifies video file as input (if not specified, camera is used) \n"
		 << " *		-image		# Specifies image file as input (if not specified, camera is used) \n"	 
		 << " * \n"
		 << " * Keys:\n"
		 << " *		Esc: Quit\n"
         << " * /\n" << endl;
}


// This function contains the actions performed for each image
void processImage(MSAC &msac, int numVps, cv::Mat &vp_pre, cv::Mat &imgGRAY, cv::Mat &outputImg)
{

	// Hough
	vector<vector<cv::Point> > lineSegments;
	vector<cv::Point> aux;

    int cols  = imgGRAY.cols;
    int rows = imgGRAY.rows;
	
  cv::Mat mask = Mat::ones( imgGRAY.size(), CV_8UC1 ); 
  // create a pointer to an LSDDetector object
  Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
  // compute lines
  std::vector<KeyLine> lines;
  lsd->detect( imgGRAY, lines, 2, 1, mask );
    cv::Point pt1, pt2;
    for (int j = 0; j != lines.size() ; ++j)

    {
	KeyLine kl = lines[j]; 
        pt1.x = kl.startPointX;
        pt1.y = kl.startPointY;
        pt2.x = kl.endPointX;
        pt2.y = kl.endPointY;

        int width = kl.lineLength;
	if (width < 15)
		continue; 
	//line(outputImg, pt1, pt2, CV_RGB(0, 255, 0), 1, 8);
	aux.clear();
	aux.push_back(pt1);
	aux.push_back(pt2);
	lineSegments.push_back(aux);
    }

 /*   cv::Mat src_gray;

    imgGRAY.convertTo(src_gray, CV_64FC1);

    image_double image = new_image_double(cols, rows);
    image->data = src_gray.ptr<double>(0);

    ntuple_list ntl = lsd(image);

    cv::Mat lsd = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Point pt1, pt2;
    for (int j = 0; j != ntl->size ; ++j)
    {
        pt1.x = int(ntl->values[0 + j * ntl->dim]);
        pt1.y = int(ntl->values[1 + j * ntl->dim]);
        pt2.x = int(ntl->values[2 + j * ntl->dim]);
        pt2.y = int(ntl->values[3 + j * ntl->dim]);
        int width = int(ntl->values[4 + j * ntl->dim]);
	//line(outputImg, pt1, pt2, CV_RGB(0, 255, 0), 1, 8);
	aux.clear();
	aux.push_back(pt1);
	aux.push_back(pt2);
	lineSegments.push_back(aux);

    }
    free_ntuple_list(ntl); */


	// Multiple vanishing points
	cv::Mat vps;			// vector of vps: vps[vpNum], with vpNum=0...numDetectedVps
	std::vector<std::vector<int> > CS;	// index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
	std::vector<int> numInliers;

	std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

	// Call msac function for multiple vanishing point estimation
	msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vp_pre, numVps); 
	/*for(unsigned int v=0; v<3; v++)
	{	
		printf("VP %d (%.3f, %.3f, %.3f)", v, vps.at<float>(0,v), vps.at<float>(1,v), vps.at<float>(2,v));
		fflush(stdout);
		// double vpNorm = cv::norm(vps[v]);
		/* if(fabs(vpNorm - 1) < 0.001)
		{
			printf("(INFINITE)");
			fflush(stdout);
		}*
		printf("\n"); 
	}	*/

	// Draw line segments according to their cluster
	printf("Angular velocity : %.3fdegree/s\n", atan(vp_pre.at<float>(2,3))*360/3.14); // tan(atof(argv[++i])*3.14/360);
	msac.drawCS(outputImg, lineSegmentsClusters, vp_pre);
	if (abs(vp_pre.at<float>(2,3)) < 0.5)
		msac.RotwarpImage(outputImg, vp_pre); 

	// vp_pre = vps; 

	// Project the image into the image plane 
	// 

}

// Main function */
int main(int argc, char** argv)
{	
	// Images
	cv::Mat inputImg, imgGRAY;	
	cv::Mat outputImg;

	// Other variables
	char *videoFileName = 0;
	char *imageFileName = 0;
	char *outputvideoFileName = 0;

	cv::VideoCapture video;
	cv::VideoWriter videoout;
	bool useCamera = true;
	int mode = MODE_LS;
	int numVps = 1;
	bool playMode = true;
	bool stillImage = false;
	bool verbose = false;

	int procWidth = -1;
	int procHeight = -1;
	cv::Size procSize;
	
	cv::Mat vp_pre = Mat::eye(3, 3, CV_64F); 
	// Start showing help
	// help();

	// Parse arguments
	if(argc < 2)
		return -1;	
	for(int i=1; i<argc; i++)
	{
		const char* s = argv[i];

		if(strcmp(s, "-video" ) == 0)
		{
			// Input video is a video file
			videoFileName = argv[++i];
			//outputvideoFileName = "RollingShutter.avi"; 
			if (videoFileName != '\0')
				useCamera = false;
		}
		else if(strcmp(s,"-image") == 0)
		{
			// Input is a image file
			imageFileName = argv[++i];
			stillImage = true;
			useCamera = false;
		}
		else if(strcmp(s, "-resizedWidth") == 0)
		{
			procWidth = atoi(argv[++i]);
		}
		else if(strcmp(s, "-verbose" ) == 0)
		{
			const char* ss = argv[++i];
			if(strcmp(ss, "ON") == 0 || strcmp(ss, "on") == 0 
				|| strcmp(ss, "TRUE") == 0 || strcmp(ss, "true") == 0 
				|| strcmp(ss, "YES") == 0 || strcmp(ss, "yes") == 0 )
				verbose = true;			
		}
		else if(strcmp(s, "-play" ) == 0)
		{
			const char* ss = argv[++i];
			if(strcmp(ss, "OFF") == 0 || strcmp(ss, "off") == 0 
				|| strcmp(ss, "FALSE") == 0 || strcmp(ss, "false") == 0 
				|| strcmp(ss, "NO") == 0 || strcmp(ss, "no") == 0 
				|| strcmp(ss, "STEP") == 0 || strcmp(ss, "step") == 0)
				playMode = false;			
		}
		else if(strcmp(s, "-mode" ) == 0)
		{
			const char* ss = argv[++i];
			if(strcmp(ss, "LS") == 0)
				mode = MODE_LS;
			else if(strcmp(ss, "NIETO") == 0)
				mode = MODE_NIETO;
			else
			{
				perror("ERROR: Only LS or NIETO modes are supported\n");
			}
		}
		else if(strcmp(s,"-numVps") == 0)
		{
			numVps = atoi(argv[++i]);
		}
	}

	// Open video input
	if( useCamera )
		video.open(0);
	else
	{
		if(!stillImage) {
			video.open(videoFileName);
			//videoout.open(outputvideoFileName);
		}
	}

	// Check video input
	int width = 0, height = 0, fps = 0, fourcc = 0;
	if(!stillImage)
	{
		if( !video.isOpened() )
		{
			printf("ERROR: can not open camera or video file\n");
			return -1;
		}
		else
		{
			// Show video information

			video.set(CV_CAP_PROP_FPS, 10); 
			//video.set(CV_CAP_PROP_GAIN, 0.5);
			width = (int) video.get(CV_CAP_PROP_FRAME_WIDTH);
			height = (int) video.get(CV_CAP_PROP_FRAME_HEIGHT);
			fps = (int) video.get(CV_CAP_PROP_FPS);
			fourcc = (int) video.get(CV_CAP_PROP_FOURCC);
			//video.set(15, -8.0); 
			//video.set(16, 0.2); 


			if(!useCamera)
				printf("Input video: (%d x %d) at %d fps, fourcc = %d\n", width, height, fps, fourcc);
			else
				printf("Input camera: (%d x %d) at %d fps\n", width, height, fps);
		}
	}
	else
	{
		inputImg = cv::imread(imageFileName);
		if(inputImg.empty())
			return -1;

		width = inputImg.cols;
		height = inputImg.rows;

		printf("Input image: (%d x %d)\n", width, height);

		playMode = false;
	}

	

	// Resize	
	if(procWidth != -1)
	{
	
		procHeight = height*((double)procWidth/width);
		procSize = cv::Size(procWidth, procHeight);

		printf("Resize to: (%d x %d)\n", procWidth, procHeight);	
	}
	else
		procSize = cv::Size(width, height);

	// Create and init MSAC
	MSAC msac;
	msac.init(mode, procSize, verbose);
	
	//videoout.open(outputvideoFileName, fourcc, fps, procSize, true);
	int frameNum=0;
	for( ;; )
	{
		if(!stillImage)
		{
			printf("\n-------------------------\nFRAME #%6d\n", frameNum);
			frameNum++;

			// Get current image		
			video >> inputImg;
		}	

		if( inputImg.empty() )
			break;
		
		// Resize to processing size
		clock_t launch = clock();
 		
		//cv::resize(inputImg, inputImg, procSize);		

		// Color Conversion
		if(inputImg.channels() == 3)
		{
			cv::cvtColor(inputImg, imgGRAY, CV_BGR2GRAY);	
			inputImg.copyTo(outputImg);
		}
		else
		{
			inputImg.copyTo(imgGRAY);
			cv::cvtColor(inputImg, outputImg, CV_GRAY2BGR);
		}

		// ++++++++++++++++++++++++++++++++++++++++
		// Process		
		// ++++++++++++++++++++++++++++++++++++++++
		 processImage(msac, numVps, vp_pre, imgGRAY, outputImg);
		
		// cout << vp_pre << endl; 

		clock_t done = clock();
		//cout << (done - launch) / CLOCKS_PER_SEC << endl;
		// View
		imshow("input", inputImg); 
		moveWindow("input", 300, 200); 
		imshow("Output", outputImg);
		moveWindow("Output", 1000, 200); 
		//videoout << outputImg;

		if(playMode)
			cv::waitKey(1);
		else
			cv::waitKey(100000);

		char q = (char)waitKey(1);

		if( q == 27 )
		{
			printf("\nStopped by user request\n");
			break;
		}	

		if(stillImage)
			break;
	}

	if(!stillImage) {
		video.release(); 
		videoout.release(); 
	}
	
	return 0;	
	
}
