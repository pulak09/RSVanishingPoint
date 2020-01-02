#include <Eigen/Eigenvalues>
#include "MSAC.h"
//#include "errorNIETO.h"
//#include "lmmin.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#ifdef DEBUG_MAP	// if defined, a 2D map will be created (which slows down the process)
#include<iostream>
using namespace std;
using namespace cv;
   using namespace Eigen;

MSAC::MSAC(void)
{
	// Auxiliar variables
	__a = Mat(3,1,CV_32F);
	__an = Mat(3,1,CV_32F);
	__b = Mat(3,1,CV_32F);
	__bn = Mat(3,1,CV_32F);
	__li = Mat(3,1,CV_32F);
	__c = Mat(3,1,CV_32F);	

	__vp = cv::Mat(3,4,CV_32F);
	__vpAux = cv::Mat(3,4,CV_32F);
}

MSAC::~MSAC(void)
{
}
void MSAC::init(int mode, cv::Size imSize, bool verbose)
{
	// Arguments
	__verbose = verbose;
	__width = imSize.width;
	__height = imSize.height;
	__verbose = verbose;
	__mode = mode;
	
	// MSAC parameters
	__epsilon = (float)1e-6;
	__P_inlier = (float)0.95;	
	__T_noise_squared = (float)0.005;
	__min_iters = 50;
	__max_iters = 500;//INT_MAX;
	__update_T_iter = false;

	// Parameters
	__minimal_sample_set_dimension = 4;

	// Minimal Sample Set
	for(int i=0; i<__minimal_sample_set_dimension; ++i) __MSS.push_back(0);
	
	// (Default) Calibration	
	__K = Mat(3,3,CV_32F);
	__K.setTo(0);
	__K.at<float>(0,0) = (float)(__width+__height)*0.85;
	__K.at<float>(0,2) = (float)__width/2;
	__K.at<float>(1,1) = (float)(__width+__height)*0.85;
	__K.at<float>(1,2) = (float)__height/2;
	__K.at<float>(2,2) = (float)1;	
//	cout << __K << endl; 
	// printf("w - h %d %d\n", __width, __height); 
	// Webcam 1 calibration 
	__K = Mat(3,3,CV_32F);
	__K.setTo(0);
	__K.at<float>(0,0) = (float) 528.4168;
	__K.at<float>(1,1) = (float) 522.0538; // Principal point x 
	__K.at<float>(0,2) = (float) 306.7807; 
	__K.at<float>(1,2) = (float) 226.60; // Principal point y 
	__K.at<float>(2,2) = (float)1;	

	// Webcam 2 calibration 
	__K = Mat(3,3,CV_32F);
	__K.setTo(0);
	__K.at<float>(0,0) = (float) 629.4738;
	__K.at<float>(1,1) = (float) 630.1313; 
	__K.at<float>(0,2) = (float) 307.5436; // Principal point x  
	__K.at<float>(1,2) = (float) 244.2221; // Principal point y 
	__K.at<float>(2,2) = (float)1;	

/*
// Webcam 2 calibration 
	__K = Mat(3,3,CV_32F);
	__K.setTo(0);
	__K.at<float>(0,0) = (float) 674.9;
	__K.at<float>(1,1) = (float) 674.9; // Principal point x 
	__K.at<float>(0,2) = (float) 307.5436; 
	__K.at<float>(1,2) = (float) 251.45; // Principal point y 
	__K.at<float>(2,2) = (float)1;	

	// Laptop Webcam calibration 
	/*__K = Mat(3,3,CV_32F);
	__K.setTo(0);
	__K.at<float>(0,0) = (float) 686.4168;
	__K.at<float>(1,1) = (float) 685.0538; // Principal point x 
	__K.at<float>(0,2) = (float) 328.0807; 
	__K.at<float>(1,2) = (float) 263.60; // Principal point y 
	__K.at<float>(2,2) = (float)1;	*/
	cout << __K << endl; 
}

// COMPUTE VANISHING POINTS
void MSAC::fillDataContainers(std::vector<std::vector<cv::Point> > &lineSegments)
{
	int numLines = lineSegments.size();
	if(__verbose)
		printf("Line segments: %d\n", numLines);

	// Transform all line segments
	// __Li = [l_00 l_01 l_02; l_10 l_11 l_12; l_20 l_21 l_22; ...]; where li=[l_i0;l_i1;l_i2]^T is li=an x bn; 
	__Li = Mat(numLines, 6, CV_32F);
	__Mi = Mat(numLines, 3, CV_32F);
	__Lengths = Mat(numLines, numLines, CV_32F);
	__Lengths.setTo(0);

	// Fill data containers (__Li, __Mi, __Lenghts)
	double sum_lengths = 0;
	for (int i=0; i<numLines; i++)
	{
		// Extract the end-points					
		Point p1 = lineSegments[i][0];
		Point p2 = lineSegments[i][1];
		__a.at<float>(0,0) = (float)p1.x;
		__a.at<float>(1,0) = (float)p1.y;
		__a.at<float>(2,0) = 1;
		__b.at<float>(0,0) = (float)p2.x;
		__b.at<float>(1,0) = (float)p2.y;
		__b.at<float>(2,0) = 1;

		if(__mode == MODE_NIETO)
			__c = 0.5*(__a+__b);
		
		double length = sqrt((__b.at<float>(0,0)-__a.at<float>(0,0))*(__b.at<float>(0,0)-__a.at<float>(0,0))
			+ (__b.at<float>(1,0)-__a.at<float>(1,0))*(__b.at<float>(1,0)-__a.at<float>(1,0)));
		sum_lengths += length;
		__Lengths.at<float>(i,i) = (float)length;
				
		if(__mode == MODE_LS)
		{
			// Normalize into the sphere
			__an = __K.inv()*__a;
			__bn = __K.inv()*__b;					
		}
		else // __mode == MODE_NIETO requires not to calibrate into the sphere
		{
			__an = __a;
			__bn = __b;
		}

		// Compute the general form of the line
		__li = __an.cross(__bn);		
		cv::normalize(__li,__li);

		// Insert line into appended array
		/*__Li.at<float>(i,0) = __li.at<float>(0,0);
		__Li.at<float>(i,1) = __li.at<float>(1,0);
		__Li.at<float>(i,2) = __li.at<float>(2,0); */

		__Li.at<float>(i,0) = __an.at<float>(0,0);
		__Li.at<float>(i,1) = __an.at<float>(1,0);
		__Li.at<float>(i,2) = __an.at<float>(2,0);

		__Li.at<float>(i,3) = __bn.at<float>(0,0);
		__Li.at<float>(i,4) = __bn.at<float>(1,0);
		__Li.at<float>(i,5) = __bn.at<float>(2,0);

		if( __mode == MODE_NIETO )
		{
			// Store mid-Point too
			__Mi.at<float>(i,0) = __c.at<float>(0,0);
			__Mi.at<float>(i,1) = __c.at<float>(1,0);
			__Mi.at<float>(i,2) = __c.at<float>(2,0);
		}
	}	
	__Lengths = __Lengths*((double)1/sum_lengths);
}

void shiftRow(cv::Mat & mat) {
	mat = mat.t(); 
     int n = 0; 
    	cv::Mat vdz = Mat(3,1,CV_32F);
	vdz.setTo(0);
	vdz.at<float>(2,0) = 1.0;
	vdz = mat*vdz; 
	
	float vl_tmp = 0;  
	for (unsigned int t=0; t<3; t++)
		if  (vl_tmp < abs(vdz.at<float>(t, 0))) {
			n = t; vl_tmp = abs(vdz.at<float>(t, 0)); 
			}
	// n = n + 1;
	//printf("%d\n", n); 

	cv::Mat temp;

	mat.row(n).copyTo(temp);
	mat.row(2).copyTo(mat.row(n)); 
	temp.copyTo(mat.row(2)); 

	vdz = Mat(3,1,CV_32F);
	vdz.setTo(0);
	vdz.at<float>(1,0) = 1.0;
	vdz = mat*vdz; 
	
	vl_tmp = 0;  
	for (unsigned int t=0; t<3; t++)
		if  (vl_tmp < abs(vdz.at<float>(t, 0))) {
			n = t; vl_tmp = abs(vdz.at<float>(t, 0)); 
			}
	// n = n + 1;
	//printf("%d\n", n); 

	mat.row(n).copyTo(temp);
	mat.row(1).copyTo(mat.row(n)); 
	temp.copyTo(mat.row(1)); 
	
	for (unsigned int t=0; t<3; t++) 
		if  (mat.at<float>(t, t) < 0) {
			mat.row(t).copyTo(temp); 
			temp = -1*temp; 
			temp.copyTo(mat.row(t)); 
			} 
	mat = mat.t(); 
}

void MSAC::multipleVPEstimation(std::vector<std::vector<cv::Point> > &lineSegments, std::vector<std::vector<std::vector<cv::Point> > > &lineSegmentsClusters, std::vector<int> &numInliers, cv::Mat &vps, int numVps)
{	
	// Make a copy of lineSegments because it is modified in the code (it will be restored at the end of this function)
	 std::vector<std::vector<cv::Point> > lineSegmentsCopy = lineSegments;
	 
	//  __vpN = cv::Mat(3,3,CV_32F);
	// Loop over maximum number of vanishing points	
	int number_of_inliers = 0;
	int vpNum=0;

	// cout << vps << endl; 

	// Fill data structures
	fillDataContainers(lineSegments);
	int numLines = lineSegments.size();

	if(__verbose)
		printf("VP %d-----\n", vpNum);		

	// Break if the number of elements is lower than minimal sample set		
	if(numLines < 3 || numLines < __minimal_sample_set_dimension)
	{
		if(__verbose)
			printf("Not enough line segments to compute vanishing point\n");
		return;
	}	

	// Vector containing indexes for current vp
	std::vector<int> ind_CS;
	
	__N_I_best = __minimal_sample_set_dimension;
	__J_best = FLT_MAX;			

	int iter = 0;
	int T_iter = INT_MAX;
	int no_updates = 0;
	int max_no_updates = INT_MAX;		

	// Define containers of CS (Consensus set): __CS_best to store the best one, and __CS_idx to evaluate a new candidate
	__CS_best = vector<int>(numLines, 0);
	__CS_idx = vector<int>(numLines, 0);

	// Allocate Error matrix
	vector<float> E = vector<float>(numLines, 0);		

	//* ---
	for(unsigned int i=0; i<__CS_idx.size(); i++)
		__CS_idx[i] = -1;
	
	float J2 = 0; int Nt; //int numLines = Li.size(); int count; 
	J2 = errorLS(vpNum, __Li, vps, E, &Nt);
	
	
	int inlr_0 = 0; int inlr_1 = 0; int inlr_2 = 0; 
	for(unsigned int i=0; i<__CS_idx.size(); i++) {
		// printf("%d \n", __CS_idx[i]);
		if (__CS_idx[i] == 0) 
			inlr_0 = inlr_0 + 1; 
		 else if (__CS_idx[i] == 1) 
			inlr_1 = inlr_1 + 1; 
		 else if (__CS_idx[i] == 2) 
			inlr_2 = inlr_2 + 1; 
		 
	}	
	__CS_idx1 = vector<int>(inlr_0, 0); __CS_idx2 = vector<int>(inlr_1, 0); __CS_idx3 = vector<int>(inlr_2, 0); inlr_0 = 0; inlr_1 = 0; inlr_2 = 0; 
	for(unsigned int i=0; i<__CS_idx.size(); i++) {
		// printf("%d \n", __CS_idx[i]);
		if (__CS_idx[i] == 0) {
			
			__CS_idx1[inlr_0] = i; 
			inlr_0 = inlr_0 + 1; 
		 } else if (__CS_idx[i] == 1) { 
			
			__CS_idx2[inlr_1] = i; 
			inlr_1 = inlr_1 + 1; 
		 } else if (__CS_idx[i] == 2) {
			
			__CS_idx3[inlr_2] = i; 	
			inlr_2 = inlr_2 + 1; 
		 }	 
	}
	
	
	// MSAC
	/*if(__verbose)
	{
		if(__mode == MODE_LS)
			printf("Method: Calibrated Least Squares\n");			
		if(__mode == MODE_NIETO)
			printf("Method: Nieto\n");

		printf("Start MSAC\n");
	} */

	// RANSAC loop
	while ( (iter <= __min_iters) || ((iter<=T_iter) && (iter <=__max_iters) && (no_updates <= max_no_updates)) )
	{					
		iter++;
		

		if(iter >= __max_iters)
			break;

		// Hypothesize ------------------------
		// Select MSS				
		if(__Li.rows < (int)__MSS.size())
			break;
		std::vector<cv::Mat> __vpN; 
		GetMinimalSampleSet(__Li, __Lengths, __Mi, __MSS, __vpN, iter);		// output __vpAux is calibrated						
		// printf("%d \n", __vpN.size()); 
		// Test --------------------------------
		// Find the consensus set and cost	
		for(unsigned int vpNum2=0; vpNum2<__vpN.size(); vpNum2++)
		{

			__vpAux = __vpN[vpNum2]; 

			int N_I = 0;
			float J = GetConsensusSet(vpNum, __Li, __Lengths, __Mi, __vpAux, E, &N_I);		// the CS is indexed in CS_idx		


			// Update ------------------------------
			// If the new cost is better than the best one, update
			if (N_I >= __minimal_sample_set_dimension && (J<__J_best) || ((J == __J_best) && (N_I > __N_I_best)))
			{
				__notify = true;

				__J_best = J;
				__CS_best = __CS_idx; 

				__vp = __vpAux;			// Store into __vp (current best hypothesis): __vp is therefore calibrated								
				
									
				if (N_I > __N_I_best)			
					__update_T_iter = true;					

				__N_I_best = N_I;

				if (__update_T_iter)
				{
					// Update number of iterations
					double q = 0;
					if (__minimal_sample_set_dimension > __N_I_best)
					{
						// Error!
						perror("The number of inliers must be higher than minimal sample set");							
					}
					if(numLines == __N_I_best)
					{
						q = 1;
					}
					else
					{
						q = 1;
						for (int j=0; j<__minimal_sample_set_dimension; j++)
							q *= (double)(__N_I_best - j)/(double)(numLines - j);
					}
					// Estimate the number of iterations for RANSAC
					if ((1-q) > 1e-12)
						T_iter = (int)ceil( log((double)__epsilon) / log((double)(1-q)));
					else
						T_iter = 0;
				}
			}
			else
				__notify = false;

			// Verbose
			/* if (__verbose && __notify)
			{
				int aux = max(T_iter, __min_iters);
				printf("Iteration = %5d/%9d. ", iter, aux);
				printf("Inliers = %6d/%6d (cost is J = %8.4f)\n", __N_I_best, numLines, __J_best);

				if(__verbose)
					printf("MSS Cal.VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));				
			}

			// Check CS length (for the case all line segments are in the CS)
			if (__N_I_best == numLines)
			{
				if(__verbose)
					printf("All line segments are inliers. End MSAC at iteration %d.\n", iter);
				break;				
			} */
		}
	}
	
	// Reestimate ------------------------------
	//cout << "M = "<< endl << " "  << __vp << endl << endl;
	shiftRow(__vp); 
	//cout << "M = "<< endl << " "  << __vp << endl << endl;
	
	__N_I_best = 0; 
	 float J = GetConsensusSet(vpNum, __Li, __Lengths, __Mi, __vp, E, &__N_I_best);
	 __CS_best = __CS_idx; 
	
	if(__verbose)
	{
		printf("Number of iterations: %d\n", iter);
		printf("Final number of inliers = %d/%d\n", __N_I_best, numLines); 			
	}			

	// Fill ind_CS with __CS_best
	

	for(int k=-1; k<3;k++) {
		std::vector<std::vector<cv::Point> > lineSegmentsCurrent;
		for(int i=0; i<numLines; i++)
		{
			if(__CS_best[i] == k)
			{
				int a = i;
				ind_CS.push_back(a);
				lineSegmentsCurrent.push_back(lineSegments[i]);
			}
		}
		lineSegmentsClusters.push_back(lineSegmentsCurrent); 
	}
	vps = __vp;//(Range(0, __vp.rows), Range(0, __vp.cols - 1));  	
	// cout << "Rotation = " << 2*atan(__vp.at<float>(2,3))*180/3.14 << endl; 
	//vps = __vp; 	
}


// RANSAC
void MSAC::GetMinimalSampleSet(cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, std::vector<int> &MSS, std::vector<cv::Mat> &vp, int iter)
{	
	int N = Li.rows;	
	int N1; 
	int N2;
	int N3;

	if (iter % 5 == 0) {
		N1 = 0; 
		N2 = 0;
		N3 = 0; 
	} else {
		N1 = __CS_idx1.size(); 
		N2 = __CS_idx2.size(); 
		N3 = __CS_idx3.size(); 
		//printf("iter = %d \n", iter); 
	}
	//printf("%d %d %d\n", N1, N2, N3); 
	// Generate a pair of samples	
	if (N1 > 5) {
		while (N1 <= (MSS[0] = rand() / (RAND_MAX/(N1-1)))); 
		MSS[0] = __CS_idx1[MSS[0]]; 
		}
	else
		while (N <= (MSS[0] = rand() / (RAND_MAX/(N-1)))); 
	if (N2 > 5){
		while (N2 <= (MSS[1] = rand() / (RAND_MAX/(N2-1)))); 
		MSS[1] = __CS_idx2[MSS[1]]; 
		}
	else
		while (N <= (MSS[1] = rand() / (RAND_MAX/(N-1))));
	if (N3 > 5){
		while (N3 <= (MSS[2] = rand() / (RAND_MAX/(N3-1)))); 
		MSS[2] = __CS_idx3[MSS[2]]; 

		while (N3 <= (MSS[3] = rand() / (RAND_MAX/(N3-1)))); 
		MSS[3] = __CS_idx3[MSS[3]]; 
		}
	else {
		while (N <= (MSS[2] = rand() / (RAND_MAX/(N-1))));
		while (N <= (MSS[3] = rand() / (RAND_MAX/(N-1))));
	}

	

	//printf("%d %d %d\n", MSS[0], MSS[1], MSS[2]); 

	//while (N <= (MSS[1] = rand() / (RAND_MAX/(N-1))));
	//while (N <= (MSS[2] = rand() / (RAND_MAX/(N-1))));
	//while (N <= (MSS[3] = rand() / (RAND_MAX/(N-1))));
	// Estimate the vanishing point and the residual error
	if(__mode == MODE_LS)
		//  estimateLS(Li,Lengths, MSS, 4, vp);
		 estimateRS(Li,Lengths, MSS, 4, vp);		
	
	//else if(__mode == MODE_NIETO)
	//estimateNIETO(Li, Mi, Lengths, MSS, 2, vp);
	else
		perror("ERROR: mode not supported. Please use {LS, LIEB, NIETO}\n");
}


float MSAC::GetConsensusSet(int vpNum, cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, cv::Mat &vp, std::vector<float> &E, int *CS_counter)
{
	// Compute the error of each line segment of LSS with respect to v_est
	// If it is less than the threshold, add to the CS	
	for(unsigned int i=0; i<__CS_idx.size(); i++)
		__CS_idx[i] = -1;
	
	float J = 0;

	if(__mode == MODE_LS)
		J = errorLS(vpNum, Li, vp, E, CS_counter);	
	//else if(__mode == MODE_NIETO)
		//J = errorNIETO(vpNum, Li, Lengths, Mi, vp, E, CS_counter);
	else
		perror("ERROR: mode not supported, please use {LS, LIEB, NIETO}\n");

	return J;
}


// Estimation functions // GS vanishing directions 
void MSAC::estimateLS(cv::Mat &Li, cv::Mat &Lengths, std::vector<int> &set, int set_length, std::vector<cv::Mat> &vp)
{
	if (set_length == __minimal_sample_set_dimension)
	{	
		// Just the cross product
		// DATA IS CALIBRATED in MODE_LS
		cv::Mat lsx0 = Mat(3,1,CV_32F);
		cv::Mat lsx1 = Mat(3,1,CV_32F);
		cv::Mat lsx2 = Mat(3,1,CV_32F);

		cv::Mat lsy0 = Mat(3,1,CV_32F);
		cv::Mat lsy1 = Mat(3,1,CV_32F);
		cv::Mat lsy2 = Mat(3,1,CV_32F);

		lsx0.at<float>(0,0) = Li.at<float>(set[0],0);
		lsx0.at<float>(1,0) = Li.at<float>(set[0],1);
		lsx0.at<float>(2,0) = Li.at<float>(set[0],2);

		lsx1.at<float>(0,0) = Li.at<float>(set[1],0);
		lsx1.at<float>(1,0) = Li.at<float>(set[1],1);
		lsx1.at<float>(2,0) = Li.at<float>(set[1],2);
		
		lsx2.at<float>(0,0) = Li.at<float>(set[2],0);
		lsx2.at<float>(1,0) = Li.at<float>(set[2],1);
		lsx2.at<float>(2,0) = Li.at<float>(set[2],2);
		

		lsy0.at<float>(0,0) = Li.at<float>(set[0],3);
		lsy0.at<float>(1,0) = Li.at<float>(set[0],4);
		lsy0.at<float>(2,0) = Li.at<float>(set[0],5);

		lsy1.at<float>(0,0) = Li.at<float>(set[1],3);
		lsy1.at<float>(1,0) = Li.at<float>(set[1],4);
		lsy1.at<float>(2,0) = Li.at<float>(set[1],5);
		
		lsy2.at<float>(0,0) = Li.at<float>(set[2],3);
		lsy2.at<float>(1,0) = Li.at<float>(set[2],4);
		lsy2.at<float>(2,0) = Li.at<float>(set[2],5);
		

		double a0 = lsx0.dot(lsx1);
		double b0 = lsx0.dot(lsx2);		
		double c0 = lsx1.dot(lsx2);


		double a1 = - lsx1.dot(lsy0);
		double b1 = - lsx2.dot(lsy0);		
		double c1 = - lsx2.dot(lsy1);


		double a2 = - lsx0.dot(lsy1);
		double b2 = - lsx0.dot(lsy2);		
		double c2 = - lsx1.dot(lsy2);


		double a3 = lsy0.dot(lsy1);
		double b3 = lsy0.dot(lsy2);		
		double c3 = lsy1.dot(lsy2);


		
		double	w0 = a0*b0*c3 - a2*b0*c2 - a0*b2*c1 + a2*b2*c0;
		double	w1 = a1*b0*c3 - a3*b0*c2 + a0*b1*c3 - a2*b1*c2 - a0*b3*c1 + a2*b3*c0 - a1*b2*c1 + a3*b2*c0;
		double  w2 = a1*b1*c3 - a3*b1*c2 - a1*b3*c1 + a3*b3*c0;

		if (w1*w1 > 4*w0*w2) {
			int ii; 
			 for (ii=0; ii < 2; ii++) {
			double lambda = (- w1 + 2*(ii - 0.5)*sqrt(w1*w1 - 4*w0*w2))/(2*w2); 
			cv::Mat __vp2 = Mat(3,4,CV_32F); 

			cv::Mat ls1 = Mat(3,1,CV_32F); 
			ls1.at<float>(0,0) = lsx0.at<float>(0, 0) - lambda*lsy0.at<float>(0, 0);
			ls1.at<float>(1,0) = lsx0.at<float>(1, 0) - lambda*lsy0.at<float>(1, 0);
			ls1.at<float>(2,0) = lsx0.at<float>(2, 0) - lambda*lsy0.at<float>(2, 0);

			cv::normalize(ls1, ls1);
			__vp2.at<float>(0,0) = ls1.at<float>(0, 0); 
			__vp2.at<float>(1,0) = ls1.at<float>(1, 0); 
			__vp2.at<float>(2,0) = ls1.at<float>(2, 0); 

			lambda = lsx1.dot(ls1) / lsy1.dot(ls1); 
			cv::Mat ls2 = Mat(3,1,CV_32F); 
			ls2.at<float>(0,0) = lsx1.at<float>(0, 0) - lambda*lsy1.at<float>(0, 0);
			ls2.at<float>(1,0) = lsx1.at<float>(1, 0) - lambda*lsy1.at<float>(1, 0);
			ls2.at<float>(2,0) = lsx1.at<float>(2, 0) - lambda*lsy1.at<float>(2, 0);

			cv::normalize(ls2, ls2);
			__vp2.at<float>(0,1) = ls2.at<float>(0, 0); 
			__vp2.at<float>(1,1) = ls2.at<float>(1, 0); 
			__vp2.at<float>(2,1) = ls2.at<float>(2, 0); 
			
			lambda = lsx2.dot(ls2) / lsy2.dot(ls2); 
			cv::Mat ls3 = Mat(3,1,CV_32F); 
			ls3.at<float>(0,0) = lsx2.at<float>(0, 0) - lambda*lsy2.at<float>(0, 0);
			ls3.at<float>(1,0) = lsx2.at<float>(1, 0) - lambda*lsy2.at<float>(1, 0);
			ls3.at<float>(2,0) = lsx2.at<float>(2, 0) - lambda*lsy2.at<float>(2, 0);

			cv::normalize(ls3, ls3);
			__vp2.at<float>(0,2) = ls3.at<float>(0, 0); 
			__vp2.at<float>(1,2) = ls3.at<float>(1, 0); 
			__vp2.at<float>(2,2) = ls3.at<float>(2, 0); 
			
			__vp2.at<float>(0,3) = 0; 
			__vp2.at<float>(1,3) = 0; 
			__vp2.at<float>(2,3) = 0; 

			vp.push_back(__vp2);
			//cout << __vp.t()*__vp << endl << endl; 
			} 

		}; 
		//else {
		//	vp.push_back(Mat::eye(3, 3, CV_32F));
		//}
		
		return;
	}	
	else if (set_length<__minimal_sample_set_dimension)
	{
		perror("Error: at least 2 line-segments are required\n");
		return;
	}
	
	// Extract the line segments corresponding to the indexes contained in the set

	cv::Mat li_set = Mat(3, set_length, CV_32F);
	cv::Mat Lengths_set = Mat(set_length, set_length, CV_32F);
	Lengths_set.setTo(0);
		
	// Fill line segments info
	for (int i=0; i<set_length; i++)
	{
		li_set.at<float>(0,i) = Li.at<float>(set[i], 0);
		li_set.at<float>(1,i) = Li.at<float>(set[i], 1);
		li_set.at<float>(2,i) = Li.at<float>(set[i], 2);
		
		Lengths_set.at<float>(i,i) = Lengths.at<float>(set[i],set[i]);		
	}		

/*
	// Least squares solution
	// Generate the matrix ATA (a partir de LSS_set=A)
	cv::Mat L = li_set.t();
	cv::Mat Tau = Lengths_set;
	cv::Mat ATA = Mat(3,3,CV_32F);
	ATA = L.t()*Tau.t()*Tau*L;
	
	// Obtain eigendecomposition
	cv::Mat w, v, vt;
	cv::SVD::compute(ATA, w, v, vt);	
	
	// Check eigenvecs after SVDecomp
	if(v.rows < 3)
		return;

	// print v, w, vt...
	//std::cout << "w=" << w << endl;
	//std::cout << "v=" << v << endl;
	//std::cout << "vt" << vt << endl;

	// Assign the result (the last column of v, corresponding to the eigenvector with lowest eigenvalue)
	vp.at<float>(0,0) = v.at<float>(0,2);
	vp.at<float>(1,0) = v.at<float>(1,2);
	vp.at<float>(2,0) = v.at<float>(2,2);	
	
	cv::normalize(vp,vp);	*/ 
	return;
}


float MSAC::errorLS(int vpNum, cv::Mat &Li, cv::Mat &vp, std::vector<float> &E, int *CS_counter) // count inliers 
{
	cv::Mat vn = vp;
	double vn_norm = cv::norm(vn);
	
	cv::Mat li;
	cv::Mat l1i = Mat(3,1,CV_32F);	cv::Mat l2i = Mat(3,1,CV_32F); cv::Mat lni = Mat(3,1,CV_32F); cv::Mat vpi = Mat(3,1,CV_32F); 
	double li_norm = 0;
	float di = 0, di_tmp = 0; int id = 0; 
	
	float J = 0;

	 cv::Mat __wx = Mat::zeros(3, 3, CV_32F); 
	double c = vp.at<float>(2,3); 
	//__wx.at<float>(0, 1) = -c;
	//__wx.at<float>(1, 0) = c; 

	/*Eigen::Matrix3d __wx; 
	__wx << 0, -vp.at<float>(2,3), 0,
		vp.at<float>(2,3), 0, 0,
		0, 0, 0;*/
	//cout << __wx; 

	for(int i=0; i<Li.rows; i++)

	{

		l1i.at<float>(0,0) = Li.at<float>(i,0) - Li.at<float>(i,1)*c*Li.at<float>(i,1);
		l1i.at<float>(1,0) = Li.at<float>(i,1) + Li.at<float>(i,1)*c*Li.at<float>(i,0);
		l1i.at<float>(2,0) = Li.at<float>(i,2);

		l2i.at<float>(0,0) = Li.at<float>(i,3) - Li.at<float>(i,4)*c*Li.at<float>(i,4);
		l2i.at<float>(1,0) = Li.at<float>(i,4) + Li.at<float>(i,4)*c*Li.at<float>(i,3);
		l2i.at<float>(2,0) = Li.at<float>(i,5);

		//__wx.at<float>(0, 1) = -l1i.at<float>(1,0)*vp.at<float>(2,3);
		//__wx.at<float>(1, 0) = l1i.at<float>(1,0)*vp.at<float>(2,3); 

		//l1i = l1i + __wx*l1i;
		//l2i = l2i + __wx*l2i; 

		lni = l1i.cross(l2i); 
		cv::normalize(lni, lni);

		/*Eigen::Vector3d l1i(Li.at<float>(i,0), Li.at<float>(i,1), Li.at<float>(i,2)); 
		Eigen::Vector3d l2i(Li.at<float>(i,3), Li.at<float>(i,4), Li.at<float>(i,5)); 

		Eigen::Vector3d lni = l1i.cross(l2i); 
		lni = lni.normalized(); 
		*/

		di = 100000000000000; id = 0; 
		for (int j=0; j < 3; j++) {
			vpi.at<float>(0, 0) = vp.at<float>(0,j) ; 
			vpi.at<float>(1, 0) = vp.at<float>(1,j) ; 
			vpi.at<float>(2, 0) = vp.at<float>(2,j) ; 

			// Eigen::Vector3d vpi(vp.at<float>(0,j), vp.at<float>(1,j), vp.at<float>(2,j)); 
			di_tmp = (float)  abs(vpi.dot(lni));	
			if (di > di_tmp) {
				di = di_tmp; 
				id = j; 
			}	
		}

		E[i] = di*di;

	//	printf("Converged Cal.VP %f \n", di); 
	/* Add to CS if error is less than expected noise */

		if (E[i] <= __T_noise_squared)
		{
			__CS_idx[i] = id;		// set index to 1
			(*CS_counter)++;	// COnsensus set size 
			
			// Torr method
			J += E[i];
		}
		else
		{
			J += __T_noise_squared;
		}
	}	

	J /= (*CS_counter);
	//cout << (*CS_counter) << endl; 
	if ((float)(*CS_counter)/Li.rows < 0.75)
		vp.at<float>(2,3) = 0; 
	return J;
}

void MSAC::drawCS(cv::Mat &im, std::vector<std::vector<std::vector<cv::Point> > > &lineSegmentsClusters, cv::Mat &vps_pre) // Draws lines and circles 
{
	vector<cv::Scalar> colors;
	colors.push_back(cv::Scalar(255, 255,0)); // First is BLACK	
	colors.push_back(cv::Scalar(0,0,255)); // First is RED
	colors.push_back(cv::Scalar(0,255,0)); // Second is GREEN 
	colors.push_back(cv::Scalar(255,0,0)); // Third is BLUE

	cv::Mat vps = __K*Mat::eye(3, 3, CV_32F); 
 	vps = __K*vps_pre(Range(0, vps_pre.rows), Range(0, vps_pre.cols - 1)); 
	//cout << vps << endl; 
	// Paint vps
	Point2f vp, vp1, vp2; 
	for(unsigned int vpNum=0; vpNum < 3; vpNum++)
	{
		if(vps.at<float>(2,vpNum) != 0)
		{
			vp.x = vps.at<float>(0,vpNum)/vps.at<float>(2,vpNum);
			vp.y = vps.at<float>(1,vpNum)/vps.at<float>(2,vpNum);

			// Paint vp if inside the image
			if(vp.x >=0 && vp.x < im.cols && vp.y >=0 && vp.y <im.rows)
			{
				circle(im, vp, 10, colors[vpNum+1], 4);	
				//cvCircle(im, vp, 3, CV_RGB(0,0,0), 1);				
			}
		}
		if (vpNum == 0)
			vp1 = vp; 
		else if (vpNum == 2)
			vp2 = vp; 
	}
	
	if (vp1.x != vp2.x) {
		if (vp1.x > 0) {
			vp2.y = vp1.y - vp1.x*(vp1.y - vp2.y)/(vp1.x - vp2.x); 
			vp2.x = 0; 
		} else {
			vp2.y = vp1.y + (im.size().width - vp1.x)*(vp1.y - vp2.y)/(vp1.x - vp2.x); 
			vp2.x = im.size().width; 
		}
		line(im, vp1, vp2, cv::Scalar(0,255,255), 2); 
	} else
		line(im, vp1, vp2, cv::Scalar(0,255,255), 2); 
	

	// Paint line segments 
	for(unsigned int c=0; c<lineSegmentsClusters.size(); c++)
	{
		for(unsigned int i=0; i<lineSegmentsClusters[c].size(); i++)
		{
			Point pt1 = lineSegmentsClusters[c][i][0];
			Point pt2 = lineSegmentsClusters[c][i][1];

			line(im, pt1, pt2, colors[c], 2);
		}
	}	
}


void MSAC::RotwarpImage(cv::Mat &im, cv::Mat &vps)
{
	// Rotate the image 
	//cv::Mat Trans =  vps * __K.inv();//Mat::eye(3, 3, CV_32F);//__K*vps; 
	// warpPerspective(im, im, Trans, im.size(), INTER_LANCZOS4); 
	//vps.at<float>(2,3); 
	//l1i.at<float>(0,0) = Li.at<float>(i,0) - Li.at<float>(i,1)*c*Li.at<float>(i,1);
	//l1i.at<float>(1,0) = Li.at<float>(i,1) + Li.at<float>(i,1)*c*Li.at<float>(i,0);

	//im.copyTo(oim);
	cv::Mat invK = __K.inv(); 
	cv::Point2f final_point; 
	float tt = 0; 
	cv::Mat Temp; 
	im.copyTo(Temp); 
	// cout << im.size().height <<  im.size().width << endl; 

	for(int y = 0; y < im.size().height; y++)
	{
		//cout << vps << endl; 
		tt = ((float) y*invK.at<float>(0, 0) + invK.at<float>(0, 2))*vps.at<float>(2,3); 
		// cout << tt<< endl; 
		for(int x = 0; x < im.size().width; x++)
		{
			im.at<Vec3b>(y,x) = Vec3b(0,0,0); 
			cv::Point2f current_pos(x,y);
			// im.at<uchar>(current_pos.y, current_pos.x) = Temp.at<uchar>(current_pos.y, current_pos.x); 
			// cout << im.at<uchar>(y, x) << endl; 
			final_point.y = (float) y -  tt*__K.at<float>(0, 0)*((float)x*invK.at<float>(1, 1) + invK.at<float>(1, 2)); 
			final_point.x = (float) x +  tt*__K.at<float>(1, 1)*((float)y*invK.at<float>(0, 0) + invK.at<float>(0, 2)); 

			cv::Point2f top_left((int)final_point.x, (int)final_point.y); //top left because of integer rounding

			// cout << top_left.x << "\t" << (float) x << "\t" << top_left.y << "\t" << (float) y << endl; 
			//make sure the point is actually inside the original image
			if(top_left.x < (int) 0 ||
			   top_left.x > im.size().width-2 ||
			   top_left.y < (int) 0 ||
			   top_left.y > im.size().height-2)
			{
			    continue;
			}

			//bilinear interpolation
			float dx = final_point.x-top_left.x;
			float dy = final_point.y-top_left.y;
			//cout << top_left.x << "\t" << (float) x << "\t" << top_left.y << "\t" << (float) y << endl;
			float weight_tl = (1.0 - dx) * (1.0 - dy);
			float weight_tr = (dx)       * (1.0 - dy);
			float weight_bl = (1.0 - dx) * (dy);
			float weight_br = (dx)       * (dy);

			Vec3b value =   weight_tl * Temp.at<Vec3b>(top_left.y,top_left.x) +
			weight_tr * Temp.at<Vec3b>(top_left.y,top_left.x+1) +
			weight_bl * Temp.at<Vec3b>(top_left.y+1,top_left.x) +
			weight_br * Temp.at<Vec3b>(top_left.y+1,top_left.x+1); 

			im.at<Vec3b>(y,x) = value;//Temp.at<Vec3b>(top_left.y, top_left.x);
		}
	}

}


void rref(cv::Mat &A) {
double TOL = 6.7987e-14; 
int i = 0;
int j = 0;
int M = A.size[0];
int N = A.size[1]; 
// cout << "M: " << M << "N: " << N << endl; 
	while ( (i<M) && (j<N) )
	{
		//	Find value and index of largest element in the remainder of column j

		double p = 0.0;
		int k = 0;
		for (unsigned int loop=i; loop<M; loop++)
		{
			double a = abs(A.at<double>(loop, j));
			if (loop == i)
			{
				p = a;
				k = loop;
			}
			else
			{
				if (a > p)
				{
					p = a;
					k = loop;
				}
			}
		}

		if (p <= TOL)
		{
			//	The column is negligible, zero it out
			for (unsigned int loop=i; loop<M; loop++)
				A.at<double>(loop, j) = 0.0;
			j++;
		}

		else
		{
			//	Remember column index (one-based for matlab return value)
			// jb[i] = j + 1;

			//	Swap i-th and k-th rows
			for (unsigned int loop=j; loop<N; loop++)
			{
				double temp = A.at<double>(k, loop);
				A.at<double>(k, loop) = A.at<double>(i, loop);
				A.at<double>(i, loop) = temp;
			}

			//	Divide the pivot row by the pivot element
			double div = A.at<double>(i, j);
			// cout << "Div" << div << "i: " << i << " j: " << j << endl; 
			for (unsigned int loop=j; loop<N; loop++)
			{
				A.at<double>(i, loop) /= div;
			}

			//	Subtract multiples of the pivot row from all the other rows
			for (unsigned int k=0; k<M; k++)
			{
				if (k == i)
					continue;

				double fac = A.at<double>(k, j);
				for (unsigned int loop=j; loop<N; loop++)
					A.at<double>(k, loop) -= fac * A.at<double>(i, loop);
			}

			//	advance
			i++;
			j++;
		}
	}

}

// Estimation functions
void MSAC::estimateRS(cv::Mat &Li, cv::Mat &Lengths, std::vector<int> &set, int set_length, std::vector<cv::Mat> &vp)
{
		
double *E; 
int  numEl = 4; 

E = (double *)calloc((unsigned int)4*numEl, sizeof(double));

memset(&E[0], 0, 16U * sizeof(double));


for (unsigned int ii = 0; ii < 4; ii++) {
	for (unsigned int jj = 0; jj < 4; jj++) 
		if (jj < 2) {
			
			E[4*ii + jj] = Li.at<float>(set[ii], jj); // cout << " P1: "  << E[4*ii + jj] << endl << endl;
		} else {
			E[4*ii + jj] = Li.at<float>(set[ii], jj+1); // cout << " P2: "  << E[4*ii + jj] << endl << endl;
	}
	// cout << "\n"<< endl; 
}

  int partialTrueCount;
//  double *E111;
  static const signed char iv0[12] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };

 // double *E211;
//  double *E311;
  int ii;
  int loop_ub;
  static const signed char idx[12] = { 0, 0, 1, 1, 2, 2, 2, 1, 0, 3, 3, 3 };//= { 1, 1, 2, 2, 3, 3, 3, 2, 1, 4, 4, 4 };

  double c_idx_0,  c_idx_30,  c_idx_31,  c_idx_1,  c_idx_2,  c_idx_3,  c_idx_4,  c_idx_5,  c_idx_6,  c_idx_7,  c_idx_8,  c_idx_9,  c_idx_10,  c_idx_11,  c_idx_12,  c_idx_13,  c_idx_14,  c_idx_15,  c_idx_16,  c_idx_17,  c_idx_18,  c_idx_19,  c_idx_20,  c_idx_21,  c_idx_22,  c_idx_23,  c_idx_24,  c_idx_25,  c_idx_26,  c_idx_27,  c_idx_28,  c_idx_29,  c_idx_32,  c_idx_33,  c_idx_34,  c_idx_35,  c_idx_36,  c_idx_37,  c_idx_38,  c_idx_39,  c_idx_40,  c_idx_41,  c_idx_42,  c_idx_43,  c_idx_44,  c_idx_45,  c_idx_46,  c_idx_47;
  static const short iv1[8] = { 64, 136, 208, 280, 547, 765, 1129, 2265 };

  static const short iv2[8] = { 356, 428, 500, 572, 1715, 1933, 2297, 3725 };

  static const short iv3[8] = { 648, 866, 1011, 1229, 2372, 2590, 2881, 4309 };

  static const short iv4[8] = { 721, 939, 1084, 1302, 2445, 2663, 2954, 4382 };

  static const short iv5[8] = { 1524, 1596, 1668, 1740, 3175, 3393, 3757, 4674 };

  static const short iv6[8] = { 1816, 2034, 2179, 2397, 3832, 4050, 4341, 5258 };

  static const short iv7[8] = { 1889, 2107, 2252, 2470, 3905, 4123, 4414, 5331 };

  static const short iv8[8] = { 2546, 2764, 2836, 3054, 4489, 4561, 4633, 5550 };

  static const short iv9[8] = { 3276, 3494, 3639, 3857, 4781, 4999, 5290, 5623 };

  static const short iv10[8] = { 3349, 3567, 3712, 3930, 4854, 5072, 5363, 5696
  };

  static const short iv11[8] = { 4006, 4224, 4296, 4514, 5438, 5510, 5582, 5915
  };

  static const short iv12[8] = { 4955, 5173, 5245, 5463, 5803, 5875, 5947, 5988
  };

  static const short iv13[8] = { 67, 139, 284, 551, 623, 841, 1205, 2339 };

  static const short iv14[8] = { 359, 431, 576, 1719, 1791, 2009, 2373, 3799 };

  static const short iv15[8] = { 578, 796, 1160, 2303, 2375, 2593, 2884, 4310 };

  static const short iv16[8] = { 724, 942, 1306, 2449, 2521, 2739, 3030, 4456 };

  static const short iv17[8] = { 1527, 1599, 1744, 3179, 3251, 3469, 3833, 4748
  };

  static const short iv18[8] = { 1746, 1964, 2328, 3763, 3835, 4053, 4344, 5259
  };

  static const short iv19[8] = { 1892, 2110, 2474, 3909, 3981, 4199, 4490, 5405
  };

  static const short iv20[8] = { 2476, 2694, 2985, 4420, 4492, 4564, 4636, 5551
  };

  static const short iv21[8] = { 3206, 3424, 3788, 4712, 4784, 5002, 5293, 5624
  };

  static const short iv22[8] = { 3352, 3570, 3934, 4858, 4930, 5148, 5439, 5770
  };

  static const short iv23[8] = { 3936, 4154, 4445, 5369, 5441, 5513, 5585, 5916
  };

  static const short iv24[8] = { 4885, 5103, 5394, 5734, 5806, 5878, 5950, 5989
  };

  static const short iv25[8] = { 70, 142, 287, 555, 700, 918, 1282, 2413 };

  static const short iv26[8] = { 362, 434, 579, 1723, 1868, 2086, 2450, 3873 };

  static const short iv27[8] = { 581, 799, 1163, 2307, 2452, 2670, 2961, 4384 };

  static const short iv28[8] = { 654, 872, 1236, 2380, 2525, 2743, 3034, 4457 };

  static const short iv29[8] = { 1530, 1602, 1747, 3183, 3328, 3546, 3910, 4822
  };

  static const short iv30[8] = { 1749, 1967, 2331, 3767, 3912, 4130, 4421, 5333
  };

  static const short iv31[8] = { 1822, 2040, 2404, 3840, 3985, 4203, 4494, 5406
  };

  static const short iv32[8] = { 2406, 2624, 2915, 4351, 4496, 4568, 4640, 5552
  };

  static const short iv33[8] = { 3209, 3427, 3791, 4716, 4861, 5079, 5370, 5698
  };

  static const short iv34[8] = { 3282, 3500, 3864, 4789, 4934, 5152, 5443, 5771
  };

  static const short iv35[8] = { 3866, 4084, 4375, 5300, 5445, 5517, 5589, 5917
  };

  static const short iv36[8] = { 4815, 5033, 5324, 5665, 5810, 5882, 5954, 5990
  };

  static const short iv37[27] = { 72, 144, 345, 417, 562, 634, 706, 851, 923,
    1214, 1286, 1703, 1775, 1847, 1992, 2064, 2355, 2427, 2499, 2717, 3008, 3805,
    3877, 3949, 4167, 4458, 5402 };

  static const short iv38[27] = { 145, 217, 418, 490, 781, 853, 925, 997, 1069,
    1360, 1432, 1922, 1994, 2066, 2138, 2210, 2574, 2646, 2718, 2790, 3081, 4024,
    4096, 4168, 4240, 4531, 5475 };

  static const short iv39[27] = { 364, 436, 1513, 1585, 1730, 1802, 1874, 2019,
    2091, 2382, 2454, 3163, 3235, 3307, 3452, 3524, 3815, 3887, 3959, 4177, 4468,
    4754, 4826, 4898, 5116, 5407, 5767 };

  static const short iv40[27] = { 437, 509, 1586, 1658, 1949, 2021, 2093, 2165,
    2237, 2601, 2673, 3382, 3454, 3526, 3598, 3670, 4034, 4106, 4178, 4250, 4541,
    4973, 5045, 5117, 5189, 5480, 5840 };

  static const short iv41[27] = { 583, 801, 1732, 1950, 2314, 2386, 2458, 2603,
    2675, 2893, 2965, 3747, 3819, 3891, 4036, 4108, 4326, 4398, 4470, 4542, 4614,
    5265, 5337, 5409, 5481, 5553, 5913 };

  static const short iv42[27] = { 1751, 1969, 3192, 3410, 3774, 3846, 3918, 4063,
    4135, 4353, 4425, 4696, 4768, 4840, 4985, 5057, 5275, 5347, 5419, 5491, 5563,
    5630, 5702, 5774, 5846, 5918, 5986 };

  static const short iv43[22] = { 352, 424, 569, 860, 932, 1223, 1295, 1712,
    1784, 1856, 2001, 2073, 2364, 2436, 2727, 3018, 3810, 3882, 3954, 4172, 4463,
    5403 };

  static const short iv44[22] = { 425, 497, 788, 1006, 1078, 1369, 1441, 1931,
    2003, 2075, 2147, 2219, 2583, 2655, 2800, 3091, 4029, 4101, 4173, 4245, 4536,
    5476 };

  static const short iv45[22] = { 1520, 1592, 1737, 2028, 2100, 2391, 2463, 3172,
    3244, 3316, 3461, 3533, 3824, 3896, 4187, 4478, 4759, 4831, 4903, 5121, 5412,
    5768 };

  static const short iv46[22] = { 1593, 1665, 1956, 2174, 2246, 2610, 2682, 3391,
    3463, 3535, 3607, 3679, 4043, 4115, 4260, 4551, 4978, 5050, 5122, 5194, 5485,
    5841 };

  static const short iv47[22] = { 1739, 1957, 2321, 2612, 2684, 2902, 2974, 3756,
    3828, 3900, 4045, 4117, 4335, 4407, 4552, 4624, 5270, 5342, 5414, 5486, 5558,
    5914 };

  static const short iv48[22] = { 3199, 3417, 3781, 4072, 4144, 4362, 4434, 4705,
    4777, 4849, 4994, 5066, 5284, 5356, 5501, 5573, 5635, 5707, 5779, 5851, 5923,
    5987 };

  double A[100], M[6059];
 // double fcnOutput[100];
  //double auto_gen_tmp_1[100];
  static const signed char iv49[10] = { 82, 81, 80, 79, 78, 77, 76, 73, 72, 71 };

  static const signed char iv50[2] = { 1, 10 };

  static const signed char iv51[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };


cv::Mat __vp2 = Mat(3,4,CV_32F); 

  for (unsigned int i1 = 0; i1 < 3; i1++) 
	for (unsigned int i2 = 0; i2 < 4; i2++) 
		if (i1 == i2)	
			__vp2.at<float>(i1, i2) = 1; 
		else
			__vp2.at<float>(i1, i2) = 0; 
  
	// vp.push_back(__vp2);

  	for (ii = 0; ii < 1; ii++) {

		//  precalculate polynomial equations coefficients
		c_idx_0 = ((((((E[1 + numEl * (idx[ii] )] * E[1 +
			    numEl * (idx[ii] )] * (E[1 + numEl *
		(idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]) -
			    E[1 + numEl * (idx[ii] )] * E[1 +
			    numEl * (idx[ii] )] * (E[3 + numEl *
		(idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )])) -
			   E[3 + numEl * (idx[ii] )] * E[3 + numEl * (idx[ii] )] * (E[1 + numEl * (idx[3 +
		ii] )] * E[1 + numEl * (idx[3 + ii] )])) + E[3 +
			  numEl * (idx[ii] )] * E[3 + numEl * (idx[ii])] * (E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )])) + E[numEl * (idx[ii])] * E[1 + numEl * (idx[ii] )] * E[numEl * (idx
		[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]) - E
			[numEl * (idx[ii] )] * E[1 + numEl * (idx[ii])] * E[2 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )]) - E[2 + numEl * (idx[ii])] * E[3 + numEl * (idx[ii] )] * E[numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii])]) + E[2 + numEl * (idx[ii] )] * E[3 +
		numEl * (idx[ii] )] * E[2 + numEl * (idx[3 + ii] )] *
		E[3 + numEl * (idx[3 + ii] )];
		
		c_idx_30 = E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )];
		c_idx_31 = E[1 + numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )];

		c_idx_1 = ((((((((((((((E[0 + numEl * (idx[ii] )] * c_idx_30 - c_idx_31 * E[0 + numEl * (idx[3 + ii] )])
		- E[numEl * (idx[ii] )] * (E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])) +
				  E[2 + numEl * (idx[ii] )] * (E[1 +
		numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])) + E[1 + numEl * (idx[ii] )] * E[1 + numEl *
				 (idx[ii] )] * E[2 + numEl * (idx[3 + ii]
		)]) + E[3 + numEl * (idx[ii] )] * E[3 + numEl *
				(idx[ii] )] * E[numEl * (idx[3 + ii] )])
			       - E[2 + numEl * (idx[ii] )] * (E[3 +
		numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )])) - E[3 + numEl * (idx[ii] )] * E[3 + numEl *
			      (idx[ii] )] * E[2 + numEl * (idx[3 + ii] )])
			     + E[numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[1 + numEl * (idx[3
		+ ii] )]) - E[numEl * (idx[ii] )] * E[1 + numEl
			    * (idx[ii] )] * E[3 + numEl * (idx[3 + ii] )])
			   - E[2 + numEl * (idx[ii] )] * E[3 +
			   numEl * (idx[ii] )] * E[1 + numEl * (idx[3
		+ ii] )]) + E[2 + numEl * (idx[ii] )] * E[3 +
			  numEl * (idx[ii] )] * E[3 + numEl * (idx[3
		+ ii] )]) + E[1 + numEl * (idx[ii] )] * E[numEl
			 * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]) - E[3 + numEl * (idx[ii] )] * E[numEl *
			(idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])
		       - E[1 + numEl * (idx[ii] )] * E[2 + numEl
		       * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )])
		+ E[3 + numEl * (idx[ii] )] * E[2 + numEl * (idx
		[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )];
		c_idx_2 = ((E[3 + numEl * (idx[ii] )] * E[3 + numEl
			* (idx[ii] )] * (E[1 + numEl * (idx[3 + ii] )] *
		E[1 + numEl * (idx[3 + ii] )]) - E[1 + numEl *
			(idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			(E[1 + numEl * (idx[3 + ii] )] * E[1 +
			 numEl * (idx[3 + ii] )])) - E[numEl *
		       (idx[ii] )] * E[1 + numEl * (idx[ii] )] *
		       E[numEl * (idx[3 + ii] )] * E[1 + numEl *
		       (idx[3 + ii] )]) + E[2 + numEl * (idx[ii] )] *
		E[3 + numEl * (idx[ii] )] * E[numEl * (idx[3 + ii]
		)] * E[1 + numEl * (idx[3 + ii] )];
		c_idx_3 = ((E[1 + numEl * (idx[ii] )] * E[1 + numEl
			* (idx[ii] )] * (E[3 + numEl * (idx[3 + ii] )] *
		E[3 + numEl * (idx[3 + ii] )]) - E[1 + numEl *
			(idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			(E[1 + numEl * (idx[3 + ii] )] * E[1 +
			 numEl * (idx[3 + ii] )])) - E[numEl *
		       (idx[ii] )] * E[1 + numEl * (idx[ii] )] *
		       E[numEl * (idx[3 + ii] )] * E[1 + numEl *
		       (idx[3 + ii] )]) + E[numEl * (idx[ii] )] *
		E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[3
		+ ii] )] * E[3 + numEl * (idx[3 + ii] )];
		c_idx_4 = ((((((E[numEl * (idx[ii] )] * E[numEl *
			    (idx[3 + ii] )] - E[numEl * (idx[ii] )] *
			    E[2 + numEl * (idx[3 + ii] )]) + E[1 +
			   numEl * (idx[ii] )] * E[1 + numEl * (idx[3
		+ ii] )]) - E[2 + numEl * (idx[ii] )] * E[numEl
			  * (idx[3 + ii] )]) - E[1 + numEl * (idx[ii] )]
			 * E[3 + numEl * (idx[3 + ii] )]) + E[2 +
			numEl * (idx[ii] )] * E[2 + numEl * (idx[3 +
		ii] )]) - E[3 + numEl * (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )]) + E[3 + numEl * (idx[ii]
		)] * E[3 + numEl * (idx[3 + ii] )];
		c_idx_5 = ((((((E[numEl * (idx[ii] )] * (E[1 + numEl
		* (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]) +
			    E[1 + numEl * (idx[ii] )] * E[1 +
			    numEl * (idx[ii] )] * E[numEl * (idx[3 +
		ii] )]) - E[2 + numEl * (idx[ii] )] * (E[1 +
		numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])) - E[3 + numEl * (idx[ii] )] * E[3 + numEl *
			  (idx[ii] )] * E[numEl * (idx[3 + ii] )]) -
			 E[numEl * (idx[ii] )] * E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )]) +
			E[2 + numEl * (idx[ii] )] * E[3 + numEl
			* (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )])
		       - E[1 + numEl * (idx[ii] )] * E[numEl *
		       (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])
		+ E[3 + numEl * (idx[ii] )] * E[numEl * (idx[3 +
		ii] )] * E[1 + numEl * (idx[3 + ii] )];
		c_idx_6 = ((((((E[numEl * (idx[ii] )] * (E[1 + numEl
		* (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]) +
			    E[1 + numEl * (idx[ii] )] * E[1 +
			    numEl * (idx[ii] )] * E[numEl * (idx[3 +
		ii] )]) - E[numEl * (idx[ii] )] * (E[3 + numEl
		* (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )])) -
			  E[1 + numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[3 + ii] )]) - E[numEl * (idx[ii] )] * E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )]) +
			E[numEl * (idx[ii] )] * E[1 + numEl *
			(idx[ii] )] * E[3 + numEl * (idx[3 + ii] )]) -
		       E[1 + numEl * (idx[ii] )] * E[numEl *
		       (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])
		+ E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx
		[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )];
		c_idx_7 = E[1 + numEl * (idx[ii] )] * E[1 + numEl *
		(idx[ii] )] * (E[1 + numEl * (idx[3 + ii] )] * E[1
				+ numEl * (idx[3 + ii] )]) + E[numEl *
		(idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )];
		c_idx_8 = ((E[2 + numEl * (idx[ii] )] * E[numEl *
			(idx[3 + ii] )] - E[1 + numEl * (idx[ii] )] *
			E[1 + numEl * (idx[3 + ii] )]) - E[numEl
		       * (idx[ii] )] * E[numEl * (idx[3 + ii] )]) +
		E[3 + numEl * (idx[ii] )] * E[1 + numEl * (idx[3
		+ ii] )];
		c_idx_9 = ((E[numEl * (idx[ii] )] * E[2 + numEl *
			(idx[3 + ii] )] - E[numEl * (idx[ii] )] *
			E[numEl * (idx[3 + ii] )]) - E[1 + numEl
		       * (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )]) +
		E[1 + numEl * (idx[ii] )] * E[3 + numEl * (idx[3
		+ ii] )];
		c_idx_10 = ((E[numEl * (idx[ii] )] * E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[3 + ii] )] -
			 E[1 + numEl * (idx[ii] )] * E[1 + numEl
			 * (idx[ii] )] * E[numEl * (idx[3 + ii] )]) -
			E[numEl * (idx[ii] )] * (E[1 + numEl *
		(idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )])) +
		E[1 + numEl * (idx[ii] )] * E[numEl * (idx[3 + ii]
		)] * E[1 + numEl * (idx[3 + ii] )];
		c_idx_11 = (E[numEl * (idx[ii] )] * E[numEl * (idx
		[3 + ii] )] + E[1 + numEl * (idx[ii] )] * E[1 +
			numEl * (idx[3 + ii] )]) + 1.0F;
		c_idx_12 = ((((((E[1 + numEl * (idx[ii] )] * E[1 +
			     numEl * (idx[ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) -
			     E[1 + numEl * (idx[ii] )] * E[1 +
			     numEl * (idx[ii] )] * (E[3 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) -
			    E[3 + numEl * (idx[ii] )] * E[3 +
			    numEl * (idx[ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) +
			   E[3 + numEl * (idx[ii] )] * E[3 + numEl * (idx[ii] )] * (E[3 + numEl * (idx[6 +
		ii] )] * E[3 + numEl * (idx[6 + ii] )])) + E
			  [numEl * (idx[ii] )] * E[1 + numEl *
			  (idx[ii] )] * E[numEl * (idx[6 + ii] )] *
			  E[1 + numEl * (idx[6 + ii] )]) - E[numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl
			 * (idx[6 + ii] )]) - E[2 + numEl * (idx[ii] )]
			* E[3 + numEl * (idx[ii] )] * E[numEl *
			(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])
		+ E[2 + numEl * (idx[ii] )] * E[3 + numEl *
		(idx[ii] )] * E[2 + numEl * (idx[6 + ii] )] * E[3 +
		numEl * (idx[6 + ii] )];
		c_idx_30 = E[3 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_31 = E[1 + numEl * (idx[ii] )] * E[1 + numEl *
		(idx[ii] )];
		c_idx_13 = ((((((((((((((E[0 + numEl * (idx[ii] )] * c_idx_30 - c_idx_31 * E[0 + numEl * (idx[6 + ii] )])
		- E[numEl * (idx[ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) +
		E[2 + numEl * (idx[ii] )] * (E[1 + numEl * (idx[6
		+ ii] )] * E[1 + numEl * (idx[6 + ii] )])) + E[1 +
				  numEl * (idx[ii] )] * E[1 + numEl *
				  (idx[ii] )] * E[2 + numEl * (idx[6 + ii]
		)]) + E[3 + numEl * (idx[ii] )] * E[3 + numEl *
				 (idx[ii] )] * E[numEl * (idx[6 + ii] )])
				- E[2 + numEl * (idx[ii] )] * (E[3 +
		numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) - E[3 + numEl * (idx[ii] )] * E[3 + numEl *
			       (idx[ii] )] * E[2 + numEl * (idx[6 + ii] )]) + E[numEl * (idx[ii] )] * E[1 + numEl *
			      (idx[ii] )] * E[1 + numEl * (idx[6 + ii] )])
			     - E[numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[3 + numEl * (idx[6
		+ ii] )]) - E[2 + numEl * (idx[ii] )] * E[3 +
			    numEl * (idx[ii] )] * E[1 + numEl * (idx
		[6 + ii] )]) + E[2 + numEl * (idx[ii] )] * E[3 +
			   numEl * (idx[ii] )] * E[3 + numEl * (idx[6
		+ ii] )]) + E[1 + numEl * (idx[ii] )] * E[numEl
			  * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) - E[3 + numEl * (idx[ii] )] * E[numEl *
			 (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])
			- E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii]
		)]) + E[3 + numEl * (idx[ii] )] * E[2 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_14 = ((E[3 + numEl * (idx[ii] )] * E[3 + numEl
			 * (idx[ii] )] * (E[1 + numEl * (idx[6 + ii] )]
		* E[1 + numEl * (idx[6 + ii] )]) - E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			 (E[1 + numEl * (idx[6 + ii] )] * E[1 +
			  numEl * (idx[6 + ii] )])) - E[numEl *
			(idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			E[numEl * (idx[6 + ii] )] * E[1 + numEl
			* (idx[6 + ii] )]) + E[2 + numEl * (idx[ii] )] *
		E[3 + numEl * (idx[ii] )] * E[numEl * (idx[6 + ii]
		)] * E[1 + numEl * (idx[6 + ii] )];
		c_idx_15 = ((E[1 + numEl * (idx[ii] )] * E[1 + numEl
			 * (idx[ii] )] * (E[3 + numEl * (idx[6 + ii] )]
		* E[3 + numEl * (idx[6 + ii] )]) - E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			 (E[1 + numEl * (idx[6 + ii] )] * E[1 +
			  numEl * (idx[6 + ii] )])) - E[numEl *
			(idx[ii] )] * E[1 + numEl * (idx[ii] )] *
			E[numEl * (idx[6 + ii] )] * E[1 + numEl
			* (idx[6 + ii] )]) + E[numEl * (idx[ii] )] *
		E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[6
		+ ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_16 = ((((((E[numEl * (idx[ii] )] * E[numEl *
			     (idx[6 + ii] )] - E[numEl * (idx[ii] )] *
			     E[2 + numEl * (idx[6 + ii] )]) + E[1 +
			    numEl * (idx[ii] )] * E[1 + numEl * (idx
		[6 + ii] )]) - E[2 + numEl * (idx[ii] )] * E
			   [numEl * (idx[6 + ii] )]) - E[1 + numEl *
			  (idx[ii] )] * E[3 + numEl * (idx[6 + ii] )])
			 + E[2 + numEl * (idx[ii] )] * E[2 + numEl * (idx[6 + ii] )]) - E[3 + numEl *
			(idx[ii] )] * E[1 + numEl * (idx[6 + ii] )]) +
		E[3 + numEl * (idx[ii] )] * E[3 + numEl * (idx[6
		+ ii] )];
		c_idx_17 = ((((((E[numEl * (idx[ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) +
			     E[1 + numEl * (idx[ii] )] * E[1 +
			     numEl * (idx[ii] )] * E[numEl * (idx[6 +
		ii] )]) - E[2 + numEl * (idx[ii] )] * (E[1 +
		numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) - E[3 + numEl * (idx[ii] )] * E[3 + numEl *
			   (idx[ii] )] * E[numEl * (idx[6 + ii] )]) -
			  E[numEl * (idx[ii] )] * E[1 + numEl *
			  (idx[ii] )] * E[1 + numEl * (idx[6 + ii] )])
			 + E[2 + numEl * (idx[ii] )] * E[3 + numEl * (idx[ii] )] * E[1 + numEl * (idx[6 + ii]
		)]) - E[1 + numEl * (idx[ii] )] * E[numEl *
			(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])
		+ E[3 + numEl * (idx[ii] )] * E[numEl * (idx[6 +
		ii] )] * E[1 + numEl * (idx[6 + ii] )];
		c_idx_18 = ((((((E[numEl * (idx[ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) +
			     E[1 + numEl * (idx[ii] )] * E[1 +
			     numEl * (idx[ii] )] * E[numEl * (idx[6 +
		ii] )]) - E[numEl * (idx[ii] )] * (E[3 + numEl
		* (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) -
			   E[1 + numEl * (idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx[6 +
		ii] )]) - E[numEl * (idx[ii] )] * E[1 + numEl *
			  (idx[ii] )] * E[1 + numEl * (idx[6 + ii] )])
			 + E[numEl * (idx[ii] )] * E[1 + numEl *
			 (idx[ii] )] * E[3 + numEl * (idx[6 + ii] )]) -
			E[1 + numEl * (idx[ii] )] * E[numEl *
			(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])
		+ E[1 + numEl * (idx[ii] )] * E[2 + numEl * (idx
		[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_19 = E[1 + numEl * (idx[ii] )] * E[1 + numEl *
		(idx[ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1
				+ numEl * (idx[6 + ii] )]) + E[numEl *
		(idx[ii] )] * E[1 + numEl * (idx[ii] )] * E[numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )];
		c_idx_20 = ((E[2 + numEl * (idx[ii] )] * E[numEl *
			 (idx[6 + ii] )] - E[1 + numEl * (idx[ii] )] *
			 E[1 + numEl * (idx[6 + ii] )]) - E[numEl * (idx[ii] )] * E[numEl * (idx[6 + ii] )])
		+ E[3 + numEl * (idx[ii] )] * E[1 + numEl * (idx
		[6 + ii] )];
		c_idx_21 = ((E[numEl * (idx[ii] )] * E[2 + numEl *
			 (idx[6 + ii] )] - E[numEl * (idx[ii] )] *
			 E[numEl * (idx[6 + ii] )]) - E[1 + numEl * (idx[ii] )] * E[1 + numEl * (idx[6 + ii] )])
		+ E[1 + numEl * (idx[ii] )] * E[3 + numEl * (idx
		[6 + ii] )];
		c_idx_22 = ((E[numEl * (idx[ii] )] * E[1 + numEl *
			 (idx[ii] )] * E[1 + numEl * (idx[6 + ii] )] -
			 E[1 + numEl * (idx[ii] )] * E[1 + numEl
			 * (idx[ii] )] * E[numEl * (idx[6 + ii] )]) -
			E[numEl * (idx[ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) +
		E[1 + numEl * (idx[ii] )] * E[numEl * (idx[6 + ii]
		)] * E[1 + numEl * (idx[6 + ii] )];
		c_idx_23 = (E[numEl * (idx[ii] )] * E[numEl * (idx
		[6 + ii] )] + E[1 + numEl * (idx[ii] )] * E[1 +
			numEl * (idx[6 + ii] )]) + 1.0F;
		c_idx_24 = ((((((E[1 + numEl * (idx[3 + ii] )] * E[1 +
			     numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) -
			     E[1 + numEl * (idx[3 + ii] )] * E[1 +
			     numEl * (idx[3 + ii] )] * (E[3 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) -
			    E[3 + numEl * (idx[3 + ii] )] * E[3 +
			    numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) +
			   E[3 + numEl * (idx[3 + ii] )] * E[3 +
			   numEl * (idx[3 + ii] )] * (E[3 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) +
			  E[numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) - E[numEl *
			 (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )]
			 * E[2 + numEl * (idx[6 + ii] )] * E[3 +
			 numEl * (idx[6 + ii] )]) - E[2 + numEl *
			(idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )]
			* E[numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) + E[2 + numEl * (idx[3 + ii]
		)] * E[3 + numEl * (idx[3 + ii] )] * E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_30 = E[3 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_31 = E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )];
		c_idx_25 = ((((((((((((((E[0 + numEl * (idx[3 + ii] )] * c_idx_30 - c_idx_31 * E[0 + numEl * (idx[6 + ii] )])
		- E[numEl * (idx[3 + ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]))
		+ E[2 + numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) +
				  E[1 + numEl * (idx[3 + ii] )] * E
				  [1 + numEl * (idx[3 + ii] )] * E[2 +
				  numEl * (idx[6 + ii] )]) + E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl *
				 (idx[3 + ii] )] * E[numEl * (idx[6 + ii]
		)]) - E[2 + numEl * (idx[3 + ii] )] * (E[3 +
		numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii])])) - E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )] * E[2 + numEl * (idx
		[6 + ii] )]) + E[numEl * (idx[3 + ii] )] * E[1 +
			      numEl * (idx[3 + ii] )] * E[1 + numEl *
			      (idx[6 + ii] )]) - E[numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] * E[3 + numEl
			     * (idx[6 + ii] )]) - E[2 + numEl * (idx[3 +
		ii] )] * E[3 + numEl * (idx[3 + ii] )] * E[1 +
			    numEl * (idx[6 + ii] )]) + E[2 + numEl *
			   (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[6 + ii] )]) + E[1 + numEl
			  * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )]
			  * E[1 + numEl * (idx[6 + ii] )]) - E[3 +
			 numEl * (idx[3 + ii] )] * E[numEl * (idx[6 +
		ii] )] * E[1 + numEl * (idx[6 + ii] )]) - E[1 +
			numEl * (idx[3 + ii] )] * E[2 + numEl * (idx
		[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )]) + E[3
		+ numEl * (idx[3 + ii] )] * E[2 + numEl * (idx[6 + ii]
		)] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_26 = ((E[3 + numEl * (idx[3 + ii] )] * E[3 +
			 numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) -
			 E[1 + numEl * (idx[3 + ii] )] * E[1 +
			 numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) -
			E[numEl * (idx[3 + ii] )] * E[1 + numEl
			* (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] *
			E[1 + numEl * (idx[6 + ii] )]) + E[2 +
		numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii])] * E[numEl * (idx[6 + ii] )] * E[1 + numEl *
		(idx[6 + ii] )];
		c_idx_27 = ((E[1 + numEl * (idx[3 + ii] )] * E[1 +
			 numEl * (idx[3 + ii] )] * (E[3 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )]) -
			 E[1 + numEl * (idx[3 + ii] )] * E[1 +
			 numEl * (idx[3 + ii] )] * (E[1 + numEl *
		(idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) -
			E[numEl * (idx[3 + ii] )] * E[1 + numEl
			* (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] *
			E[1 + numEl * (idx[6 + ii] )]) + E[numEl
		* (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] *
		E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl *
		(idx[6 + ii] )];
		c_idx_28 = ((((((E[numEl * (idx[3 + ii] )] * E[numEl
			     * (idx[6 + ii] )] - E[numEl * (idx[3 + ii] )] * E[2 + numEl * (idx[6 + ii] )]) + E[1 + numEl
			    * (idx[3 + ii] )] * E[1 + numEl * (idx[6 + ii]
		)]) - E[2 + numEl * (idx[3 + ii] )] * E[numEl *
			   (idx[6 + ii] )]) - E[1 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[6 + ii] )]) + E[2 + numEl
			 * (idx[3 + ii] )] * E[2 + numEl * (idx[6 + ii] )]) - E[3 + numEl * (idx[3 + ii] )] * E[1 + numEl
			* (idx[6 + ii] )]) + E[3 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_29 = ((((((E[numEl * (idx[3 + ii] )] * (E[1 +
		numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) + E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl
			     * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )]) - E[2 + numEl * (idx[3 + ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )])) -
			   E[3 + numEl * (idx[3 + ii] )] * E[3 +
			   numEl * (idx[3 + ii] )] * E[numEl * (idx[6
		+ ii] )]) - E[numEl * (idx[3 + ii] )] * E[1 +
			  numEl * (idx[3 + ii] )] * E[1 + numEl *
			  (idx[6 + ii] )]) + E[2 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[3 + ii] )] * E[1 + numEl
			 * (idx[6 + ii] )]) - E[1 + numEl * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] * E[1 + numEl *
			(idx[6 + ii] )]) + E[3 + numEl * (idx[3 + ii] )]
		* E[numEl * (idx[6 + ii] )] * E[1 + numEl * (idx
		[6 + ii] )];
		c_idx_30 = ((((((E[numEl * (idx[3 + ii] )] * (E[1 +
		numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) + E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl
			     * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )]) - E[numEl * (idx[3 + ii] )] * (E[3 + numEl *
		(idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )])) -
			   E[1 + numEl * (idx[3 + ii] )] * E[1 +
			   numEl * (idx[3 + ii] )] * E[2 + numEl *
			   (idx[6 + ii] )]) - E[numEl * (idx[3 + ii] )]
			  * E[1 + numEl * (idx[3 + ii] )] * E[1 +
			  numEl * (idx[6 + ii] )]) + E[numEl * (idx[3
		+ ii] )] * E[1 + numEl * (idx[3 + ii] )] * E[3 +
			 numEl * (idx[6 + ii] )]) - E[1 + numEl *
			(idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] *
			E[1 + numEl * (idx[6 + ii] )]) + E[1 +
		numEl * (idx[3 + ii] )] * E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_31 = E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] * (E[1 + numEl * (idx[6 + ii] )]
		* E[1 + numEl * (idx[6 + ii] )]) + E[numEl *
		(idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 +
		ii] )];
		c_idx_32 = ((E[2 + numEl * (idx[3 + ii] )] * E[numEl
			 * (idx[6 + ii] )] - E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[6 + ii] )]) - E[numEl *
			(idx[3 + ii] )] * E[numEl * (idx[6 + ii] )]) +
		E[3 + numEl * (idx[3 + ii] )] * E[1 + numEl *
		(idx[6 + ii] )];
		c_idx_33 = ((E[numEl * (idx[3 + ii] )] * E[2 + numEl
			 * (idx[6 + ii] )] - E[numEl * (idx[3 + ii] )] *
			 E[numEl * (idx[6 + ii] )]) - E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[6 + ii]
		)]) + E[1 + numEl * (idx[3 + ii] )] * E[3 + numEl * (idx[6 + ii] )];
		c_idx_34 = ((E[numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl * (idx[6 + ii] )] - E[1 + numEl * (idx[3 + ii] )] * E[1 + numEl
			 * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )])
			- E[numEl * (idx[3 + ii] )] * (E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )]))
		+ E[1 + numEl * (idx[3 + ii] )] * E[numEl * (idx
		[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )];
		c_idx_35 = (E[numEl * (idx[3 + ii] )] * E[numEl * (idx[6 + ii] )] + E[1 + numEl * (idx[3 + ii] )]
			* E[1 + numEl * (idx[6 + ii] )]) + 1.0F;
		c_idx_36 = E[2 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )] - E[numEl * (idx[6 + ii] )] *
		E[1 + numEl * (idx[6 + ii] )];
		c_idx_37 = E[numEl * (idx[9 + ii] )] * E[1 + numEl *
		(idx[9 + ii] )] - E[2 + numEl * (idx[9 + ii] )] * E[3 + numEl * (idx[9 + ii] )];
		c_idx_38 = E[3 + numEl * (idx[6 + ii] )] - E[1 + numEl * (idx[6 + ii] )];
		c_idx_39 = E[1 + numEl * (idx[9 + ii] )] - E[3 + numEl * (idx[9 + ii] )];
		c_idx_40 = E[numEl * (idx[6 + ii] )] * E[1 + numEl *
		(idx[6 + ii] )] - E[numEl * (idx[9 + ii] )] * E[1 +
		numEl * (idx[9 + ii] )];
		c_idx_41 = E[1 + numEl * (idx[6 + ii] )] - E[1 + numEl * (idx[9 + ii] )];
		c_idx_42 = E[3 + numEl * (idx[6 + ii] )] * E[3 + numEl * (idx[6 + ii] )] - E[1 + numEl * (idx[6 + ii] )]
		* E[1 + numEl * (idx[6 + ii] )];
		c_idx_43 = E[1 + numEl * (idx[9 + ii] )] * E[1 + numEl * (idx[9 + ii] )] - E[3 + numEl * (idx[9 + ii] )]
		* E[3 + numEl * (idx[9 + ii] )];
		c_idx_44 = E[numEl * (idx[6 + ii] )] - E[2 + numEl *
		(idx[6 + ii] )];
		c_idx_45 = E[2 + numEl * (idx[9 + ii] )] - E[numEl *
		(idx[9 + ii] )];
		c_idx_46 = E[1 + numEl * (idx[6 + ii] )] * E[1 + numEl * (idx[6 + ii] )] - E[1 + numEl * (idx[9 + ii] )] * E[1 + numEl * (idx[9 + ii] )];
		c_idx_47 = E[numEl * (idx[9 + ii] )] - E[numEl *(idx[6 + ii] )];

		memset(&M[0], 0, 6059U * sizeof(double));
		
		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv1[partialTrueCount]] = c_idx_0;
		}
		
		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv2[partialTrueCount]] = c_idx_1;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv3[partialTrueCount]] = c_idx_2;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv4[partialTrueCount]] = c_idx_3;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv5[partialTrueCount]] = c_idx_4;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv6[partialTrueCount]] = c_idx_5;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv7[partialTrueCount]] = c_idx_6;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv8[partialTrueCount]] = c_idx_7;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv9[partialTrueCount]] = c_idx_8;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv10[partialTrueCount]] = c_idx_9;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv11[partialTrueCount]] = c_idx_10;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv12[partialTrueCount]] = c_idx_11;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv13[partialTrueCount]] = c_idx_12;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv14[partialTrueCount]] = c_idx_13;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv15[partialTrueCount]] = c_idx_14;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv16[partialTrueCount]] = c_idx_15;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv17[partialTrueCount]] = c_idx_16;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv18[partialTrueCount]] = c_idx_17;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv19[partialTrueCount]] = c_idx_18;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv20[partialTrueCount]] = c_idx_19;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv21[partialTrueCount]] = c_idx_20;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv22[partialTrueCount]] = c_idx_21;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv23[partialTrueCount]] = c_idx_22;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv24[partialTrueCount]] = c_idx_23;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv25[partialTrueCount]] = c_idx_24;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv26[partialTrueCount]] = c_idx_25;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv27[partialTrueCount]] = c_idx_26;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv28[partialTrueCount]] = c_idx_27;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv29[partialTrueCount]] = c_idx_28;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv30[partialTrueCount]] = c_idx_29;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv31[partialTrueCount]] = c_idx_30;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv32[partialTrueCount]] = c_idx_31;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv33[partialTrueCount]] = c_idx_32;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv34[partialTrueCount]] = c_idx_33;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv35[partialTrueCount]] = c_idx_34;
		}

		for (partialTrueCount = 0; partialTrueCount < 8; partialTrueCount++) {
		M[iv36[partialTrueCount]] = c_idx_35;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv37[partialTrueCount]] = c_idx_36;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv38[partialTrueCount]] = c_idx_37;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv39[partialTrueCount]] = c_idx_38;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv40[partialTrueCount]] = c_idx_39;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv41[partialTrueCount]] = c_idx_40;
		}

		for (partialTrueCount = 0; partialTrueCount < 27; partialTrueCount++) {
		M[iv42[partialTrueCount]] = c_idx_41;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv43[partialTrueCount]] = c_idx_42;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv44[partialTrueCount]] = c_idx_43;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv45[partialTrueCount]] = c_idx_44;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv46[partialTrueCount]] = c_idx_45;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv47[partialTrueCount]] = c_idx_46;
		}

		for (partialTrueCount = 0; partialTrueCount < 22; partialTrueCount++) {
		M[iv48[partialTrueCount]] = c_idx_47;
		}

		// rref(M);  % replace me with a MEX
		cv::Mat Mr = Mat(73, 83,CV_64F); 
		for (unsigned int ii1=0; ii1<73; ii1++)
			for (unsigned int ii2=0; ii2<83; ii2++)
				Mr.at<double>(ii1, ii2) = M[ii1+73*ii2]; 
		// cout << "M = "<< endl << " "  << Mr << endl << endl; 
		 rref(Mr); 
		// cout << "M = "<< endl << " "  << Mr << endl << endl; 
		 memset(&A[0], 0, 100U * sizeof(double));

		A[10] = 1.0F;
		A[61] = 1.0F;
		A[74] = 1.0F;
		A[85] = 1.0F;
		for (partialTrueCount = 0; partialTrueCount < 10; partialTrueCount++) {
			A[2 + 10 * partialTrueCount] = -Mr.at<double>(72, iv49[partialTrueCount]);	//[72 + 73 * iv49[partialTrueCount]];
			A[3 + 10 * partialTrueCount] = -Mr.at<double>(71, iv49[partialTrueCount]);	//[71 + 73 * iv49[partialTrueCount]];
			A[6 + 10 * partialTrueCount] = -Mr.at<double>(63, iv49[partialTrueCount]);	//[63 + 73 * iv49[partialTrueCount]];
			A[7 + 10 * partialTrueCount] = -Mr.at<double>(60, iv49[partialTrueCount]);	//[60 + 73 * iv49[partialTrueCount]];
			A[8 + 10 * partialTrueCount] = -Mr.at<double>(59, iv49[partialTrueCount]);	//[59 + 73 * iv49[partialTrueCount]];
			A[9 + 10 * partialTrueCount] = -Mr.at<double>(58, iv49[partialTrueCount]);	//[58 + 73 * iv49[partialTrueCount]];
		}
		
		//  eig(A, fcnOutput, auto_gen_tmp_1);

		/*
		cv::Mat ATA = Mat(10, 10, CV_32F);
		for (unsigned int i1=0; i1<10; i1++)
			for (unsigned int i2=0; i2<10; i2++)
				ATA.at<float>(i1, i2) = A[i1+10*i2];//(i1+1)*(i2+1); 
		// Obtain eigendecomposition
		cv::Mat u, v, vt;
		// cv::SVD::compute(ATA, w, v, vt); 
		cv::SVD::compute(ATA, u, v, vt); // Find sutable Eigen decomposition. 
		// cout << "Eigen = "<< endl << " "  << ATA - v*Mat::diag(u)*vt.t()  << endl << endl;
  		// cout << "Eigen = "<< endl << " "  << ATA  << endl << endl;  
		*/

		MatrixXcd eigenA(10, 10);
		
		for (unsigned int i1=0; i1<10; i1++)
			for (unsigned int i2=0; i2<10; i2++) {
				eigenA.real()(i1, i2) = A[i1+10*i2];//(i1+1)*(i2+1);
				eigenA.imag()(i1, i2) = 0; 
				}
	  	//cout << "Eigen = "<< endl << " "  << eigenA.real()  << endl << endl;
		// cv2eigen(ATA,eigenA); //convert OpenCV to Eigen 
		Eigen::ComplexEigenSolver<MatrixXcd> ces;
		ces.compute(eigenA);
	 	//cout << "The eigenvalues of A are:\n" <<ces.eigenvalues() << endl;
		//cout << "The matrix of eigenvectors, V, is:\n" << ces.eigenvectors() << endl;

		// pair<Matrix4cd, Matrix4cd> result;
		// cout << "Eigen = "<< endl << " "  << ATA - v*Mat::diag(u)*vt.t()  << endl << endl;  
		// for(unsigned int k=0; k < 16; k++) 
		// cout << "vt = "<< endl << " "  << u  << endl << endl;  

		double x, y, z, c; 
		cv::Mat lsx0 = Mat(3, 1, CV_64F);
		cv::Mat lsx1 = Mat(3, 1, CV_64F); 
		cv::Mat lsx2 = Mat(3, 1, CV_64F); 
		cv::Mat lsy0 = Mat(3, 1, CV_64F); 
		cv::Mat lsy1 = Mat(3, 1, CV_64F); 		
		cv::Mat lsy2 = Mat(3, 1, CV_64F);  
		cv::Mat __wx = Mat::zeros(3, 3, CV_64F); 

		for (unsigned int ij1=0; ij1<10; ij1++) {

			if (abs(ces.eigenvalues().imag()(ij1, 0)) < 0.000000001) {
				if (abs(ces.eigenvectors().real()(1, ij1)/ces.eigenvectors().real()(0, ij1)) < 2) {
				// cout << "The best\n" << ces.eigenvectors().real()(9, ij1) << endl; 
				 x = ces.eigenvectors().real()(5, ij1)/ces.eigenvectors().real()(0, ij1);
 				 y = ces.eigenvectors().real()(4, ij1)/ces.eigenvectors().real()(0, ij1);
				 z = ces.eigenvectors().real()(3, ij1)/ces.eigenvectors().real()(0, ij1); 
			//	 s = ces.eigenvectors().real()(2, ij1)/ces.eigenvectors().real()(0, ij1); 
				 c = ces.eigenvectors().real()(1, ij1)/ces.eigenvectors().real()(0, ij1);  

			//	cout << "x = "<< endl << " "  << x << endl << endl; 
			//	cout << "y = "<< endl << " "  << y << endl << endl; 
			//	cout << "z = "<< endl << " "  << z << endl << endl; 
			//	cout << "c = "<< endl << " "  << c << endl << endl; 

				 


				lsx0.at<double>(0,0) = Li.at<float>(set[0],0) - Li.at<float>(set[0],1)*c*Li.at<float>(set[0],1);
				lsx0.at<double>(1,0) = Li.at<float>(set[0],1) + Li.at<float>(set[0],1)*c*Li.at<float>(set[0],0);
				lsx0.at<double>(2,0) = Li.at<float>(set[0],2);

				lsx1.at<double>(0,0) = Li.at<float>(set[1],0) - Li.at<float>(set[1],1)*c*Li.at<float>(set[1],1);
				lsx1.at<double>(1,0) = Li.at<float>(set[1],1) + Li.at<float>(set[1],1)*c*Li.at<float>(set[1],0);
				lsx1.at<double>(2,0) = Li.at<float>(set[1],2);

				
				lsx2.at<double>(0,0) = Li.at<float>(set[2],0) - Li.at<float>(set[2],1)*c*Li.at<float>(set[2],1);
				lsx2.at<double>(1,0) = Li.at<float>(set[2],1) + Li.at<float>(set[2],1)*c*Li.at<float>(set[2],0);
				lsx2.at<double>(2,0) = Li.at<float>(set[2],2);

				
				lsy0.at<double>(0,0) = Li.at<float>(set[0],3) - Li.at<float>(set[0],4)*c*Li.at<float>(set[0],4);
				lsy0.at<double>(1,0) = Li.at<float>(set[0],4) + Li.at<float>(set[0],4)*c*Li.at<float>(set[0],3);
				lsy0.at<double>(2,0) = Li.at<float>(set[0],5);

				
				lsy1.at<double>(0,0) = Li.at<float>(set[1],3) - Li.at<float>(set[1],4)*c*Li.at<float>(set[1],4);
				lsy1.at<double>(1,0) = Li.at<float>(set[1],4) + Li.at<float>(set[1],4)*c*Li.at<float>(set[1],3);
				lsy1.at<double>(2,0) = Li.at<float>(set[1],5);

				
				lsy2.at<double>(0,0) = Li.at<float>(set[2],3) - Li.at<float>(set[2],4)*c*Li.at<float>(set[2],4);
				lsy2.at<double>(1,0) = Li.at<float>(set[2],4) + Li.at<float>(set[2],4)*c*Li.at<float>(set[2],3);
				lsy2.at<double>(2,0) = Li.at<float>(set[2],5);

				
				/*__wx.at<double>(0, 1) = -c;
				__wx.at<double>(1, 0) = c; 
				
				//cout << "Lines = "<< lsy0 <<endl << lsy1 <<endl << lsy2 <<endl; 

				lsx0 = lsx0 + lsx0.at<double>(1,0)*__wx*lsx0;
				lsx1 = lsx1 + lsx1.at<double>(1,0)*__wx*lsx1; 
				lsx2 = lsx2 + lsx2.at<double>(1,0)*__wx*lsx2; 

				lsy0 = lsy0 + lsy0.at<double>(1,0)*__wx*lsy0;  
				lsy1 = lsy1 + lsy1.at<double>(1,0)*__wx*lsy1; 
				lsy2 = lsy2 + lsy2.at<double>(1,0)*__wx*lsy2; */
				
				cv::Mat v1 = (1 - x)*lsx0 + x*lsy0; 
				cv::Mat v2 = (1 - y)*lsx1 + y*lsy1; 
				cv::Mat v3 = (1 - z)*lsx2 + z*lsy2; 

				//cout << "V1 = "<< v1 <<endl << v2 <<endl << v3 <<endl; 

				cv::normalize(v1, v1); 
				cv::normalize(v2, v2); 
				cv::normalize(v3, v3); 

								

				cv::Mat __vp2 = Mat::zeros(3, 4, CV_32F); 
				__vp2.at<float>(0,0) = v1.at<double>(0, 0); 
				__vp2.at<float>(1,0) = v1.at<double>(1, 0); 
				__vp2.at<float>(2,0) = v1.at<double>(2, 0); 
			
				__vp2.at<float>(0,1) = v2.at<double>(0, 0); 
				__vp2.at<float>(1,1) = v2.at<double>(1, 0); 
				__vp2.at<float>(2,1) = v2.at<double>(2, 0); 
			
				__vp2.at<float>(0,2) = v3.at<double>(0, 0); 
				__vp2.at<float>(1,2) = v3.at<double>(1, 0); 
				__vp2.at<float>(2,2) = v3.at<double>(2, 0); 
			
				//__vp2.at<float>(0,3) = 0; 
				//__vp2.at<float>(1,3) = 0; 
				__vp2.at<float>(2,3) = c; 

				//cout << "vt = "<< c << endl << " "  << __vp2.t()*__vp2 << endl << endl; 
				if (abs(c) < 1)
					vp.push_back(__vp2); 
				}

			} else {
				continue; 
			}
		}
		
	// emxFree_real32_T(&E311);
	// emxFree_real32_T(&E211);
	//  emxFree_real32_T(&E111);
	}
	return; 
}

