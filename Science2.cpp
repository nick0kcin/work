#define _USE_MATH_DEFINES
#define DISABLE_OPENCV_24_COMPATIBILITY
#include "stdafx.h"
#include "opencv\cv.h"
#include "opencv\cv.hpp"
#include "opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2\stereo\stereo.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\ximgproc.hpp"
#include "opencv2\stitching.hpp"
#include "opencv2\stitching\detail\autocalib.hpp"
#include "opencv2\stitching\detail\motion_estimators.hpp"
#include "iostream"
#include "opencv2/optflow.hpp"
#include <cmath>
#include "genhough.h"
#include <queue>
#include <map>
#include "CalibratorCipolla.h"
#pragma comment(linker, "/STACK:16777216")

using namespace cv;
const int imcap=2;
double timestep=900.0;

int main() // калибровка
{
	VideoCapture cap("v.mov");
	namedWindow("ff");
	Mat frame[imcap];
	Mat inp[imcap];
	cap.set(CAP_PROP_POS_MSEC,99000);

	std::vector<cv::KeyPoint> points[imcap];
	UMat descriptors[imcap];
	std::vector<cv::DMatch> matches[imcap];
	std::vector<char> maskmatches[imcap];
	int matchchain[imcap];
	std::vector<DMatch> goodmatches;
	std::vector<std::vector<Point2f> > pnts(imcap,std::vector<Point2f>());
	for(int i=0;i<imcap;i++)
		pnts[i].reserve(1000);
	std::vector<Mat> Fund;
	std::vector<double> trust;


	BFMatcher matcher;
	Ptr<Feature2D> detector = xfeatures2d::SURF::create();
	Ptr<Feature2D> descriptor = xfeatures2d::SURF::create();

	double width=cap.get(CAP_PROP_FRAME_WIDTH);
	double height=cap.get(CAP_PROP_FRAME_HEIGHT);
	Mat K=Mat(Matx33d(1,0,width/4.0,0,1,height/4.0,0,0,1));
	int n=0;
	bool hasRound=0;
	while(1)
	{
		char c= waitKey(500);
		if(c==' ')
		{
			std::cout<<"]=>\n";
			cap>>frame[n];
		cap.set(CAP_PROP_POS_MSEC,cap.get(CAP_PROP_POS_MSEC)+timestep);
			cv::resize(frame[n],frame[n],cv::Size(0,0),0.5,0.5);
		cv::cvtColor(frame[n],inp[n],CV_RGB2GRAY);
		GaussianBlur(inp[n],inp[n],Size(0,0),2);
		detector->detect(inp[n],points[n]);
		detector->compute(inp[n],points[n],descriptors[n]);
		if(hasRound||n+1==imcap)
		{
			for(int i=0;i<imcap;i++)
			{
				matcher.match(descriptors[(n-i+imcap)%imcap],descriptors[(n-i-1+imcap)%imcap],matches[i]);
				pnts[i].clear();
				maskmatches[i].assign(matches[i].size(),0);
			}
			float distmin=1000000;
			for(int i=0;i<matches[0].size();i++)
				distmin=min(distmin,matches[0][i].distance);
			for(int i=0;i<matches[0].size();i++)
			{
				matchchain[0]=matches[0][i].trainIdx;
				float dist=matches[0][i].distance;
				for(int j=1;j<imcap;j++)
				{
					matchchain[j]=matches[j][matchchain[j-1]].trainIdx;
					dist=max(dist,matches[j][matchchain[j-1]].distance);
				}
				if(matchchain[imcap-1]==i&&dist<4*distmin)
				{
					for(int j=0;j<imcap;j++)
					{
						pnts[j].push_back(points[(n-j+imcap)%imcap][matches[j][matchchain[(j-1+imcap)%imcap]].queryIdx].pt);
						maskmatches[j][matchchain[(j-1+imcap)%imcap]]=1;
					}
				}
			}
			Mat immatch;
			for(int i=0;i<imcap;i++)
			{
				drawMatches(frame[(n-i+imcap)%imcap],points[(n-i+imcap)%imcap],frame[(n-i-1+imcap)%imcap],points[(n-i-1+imcap)%imcap],matches[i],
					immatch,cv::Scalar_<double>::all(-1),cv::Scalar_<double>::all(-1),maskmatches[i]);
				std::string name(i+1,'0');
				imshow(name,immatch);
			}
			for(int i=0;i<imcap-1;i++)
				for(int j=i+1;j<imcap;j++)
				{
					Mat mask;
					 Mat F=findFundamentalMat(pnts[i],pnts[j],FM_RANSAC,0.3,0.99,mask);
					//std::cout<<F[i][j]<<"\n";
					if(!F.empty())
					{
					 double maxdist=0;
					for(int k=0;k<pnts[i].size();k++)
					{
						Mat p1=Mat(Matx31d(pnts[i][k].x,pnts[i][k].y,1));
						Mat p2=Mat(Matx31d(pnts[j][k].x,pnts[j][k].y,1));
						Mat ll=(p2.t()*F);
						maxdist=max(maxdist,ll.dot(p1.t())*mask.at<bool>(k));
					}
					trust.push_back(1.0/maxdist);
					Fund.push_back(F);
					}
					//std::cout<<mask<<"\n";
					//std::cin.get();
				}
			double val=0;
			calibrateCipolla(trust,Fund,val,K,CalibrationType::NO_SKEW);
			std::cout<<K<<"\n"<<val<<"\n";
			for(int i=0;i<imcap;i++)
				perspectiveTransform(pnts[i],pnts[i],K);
			for(int i=0;i<imcap-1;i++)
			{
				Mat E=findEssentialMat(pnts[i],pnts[i+1]);
				Mat R,t;
				recoverPose(E,pnts[i],pnts[i+1],R,t);
				std::cout<<R<<"\n"<<t<<"\n";
			}
				/*for(int j=0;j<Fund.size();j++)
					if(!Fund[j].empty())
				{
					Mat E=K.t()*Fund[j]*K;
					std::cout<<E<<"\n"<<SVD(E).w<<"\n";
				}*/
		}
		n++;
		hasRound=hasRound||(n==imcap);
		n-=((n==imcap)?n:0);
		}
	}
	return 0;
}