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
#include <queue>
#include <map>
#include "CalibratorCipolla.h"
#include "BundleAdjuster.h"
#pragma comment(linker, "/STACK:16777216")

//#define CALIBRATION
#define BUNDLE_ADJ 



using namespace cv;
const int imcap=2;
double timestep=1500.0;

double computeFMError( Point2f m1,Point2f m2, Mat Fund )
{
            double a, b, c, d1, d2, s1, s2;

			double* F=Fund.ptr<double>();

            a = F[0]*m1.x + F[1]*m1.y + F[2];
            b = F[3]*m1.x + F[4]*m1.y + F[5];
            c = F[6]*m1.x + F[7]*m1.y + F[8];

            s2 = 1./(a*a + b*b);
            d2 = m2.x*a + m2.y*b + c;

            a = F[0]*m2.x + F[3]*m2.y + F[6];
            b = F[1]*m2.x + F[4]*m2.y + F[7];
            c = F[2]*m2.x + F[5]*m2.y + F[8];

            s1 = 1./(a*a + b*b);
            d1 = m1.x*a + m1.y*b + c;

            return std::max(d1*d1*s1, d2*d2*s2);
        }



int main() // калибровка
{

	VideoCapture cap("v.mov");
	namedWindow("ff");
	Mat frame[imcap];
	Mat inp[imcap];
	cap.set(CAP_PROP_POS_MSEC,110000);

	std::vector<cv::KeyPoint> points[imcap];
	UMat descriptors[imcap];
	std::vector<cv::DMatch> matches[imcap];
	std::vector<char> maskmatches[imcap];
	int matchchain[imcap];
	std::vector<DMatch> goodmatches;
	std::vector<std::vector<Point2f> > pnts(imcap,std::vector<Point2f>());
	for(int i=0;i<imcap;i++)
		pnts[i].reserve(1000);
#ifdef CALIBRATION
	std::vector<Mat> Fund;
	std::vector<double> trust;
	Mat F,FF;
#endif
#ifdef BUNDLE_ADJ
	std::vector<std::vector<std::pair<Point2f,int> > > for_ba_points;  //[world points][projection on each image
	std::vector<Mat> for_ba_cams; // camera matrixs
	std::vector<int> keys[imcap]; // i-th keypoint in j image is projection keys[i][j]-th world point
	std::vector<Mat> for_ba_world; //world points
	std::vector<std::vector<std::pair<int,int> > > for_ba_ind;
	std::vector<std::vector<Point2f> > normpnts(imcap,std::vector<Point2f>());
	Mat R,t,E;
		Mat GlobalR=Mat::eye(3,3,CV_64F);
	Mat Globalt=Mat::zeros(3,1,CV_64F);
	std::vector<short> newp;
#endif

	BFMatcher matcher;
	Ptr<Feature2D> detector = xfeatures2d::SURF::create();
	Ptr<Feature2D> descriptor = xfeatures2d::SURF::create();

	double width=cap.get(CAP_PROP_FRAME_WIDTH);
	double height=cap.get(CAP_PROP_FRAME_HEIGHT);
#ifdef CALIBRATION
	Mat K=Mat(Matx33d(1,0,width/4.0,0,1,height/4.0,0,0,1));
#else
	Mat K=Mat(Matx33d(1211.07665496295, 0, 495.9584508898477,0, 1217.085992209764, 225.3544367583887,0, 0, 1));
#endif

	int n=0;
	bool hasRound=0;
	while(1)
	{
		char c= waitKey(500);
		//if(c=='b')
		{
		}
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
#ifdef BUNDLE_ADJ
		keys[n].assign(points[n].size(),-1);
		int newpoints=0;
		newp.clear();
		int worldsize=for_ba_world.size();
		for_ba_points.push_back(std::vector<std::pair<Point2f,int> >());
		#endif
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
			for(int i=0;i<matches[1].size();i++)
				distmin=min(distmin,matches[1][i].distance);
			std::cout<<distmin<<"\n";
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
#ifdef BUNDLE_ADJ
						if(j==0) //prepare points  for bundle adjustment
						{
							int prevPointInd=matchchain[0];
							int curPointInd=i;
							keys[n][curPointInd]=keys[(n-1+imcap)%imcap][prevPointInd]; 
							if(keys[(n-1+imcap)%imcap][prevPointInd]==-1)// if we meet point firstly
						{
							for_ba_points[for_ba_points.size()-2].push_back(std::make_pair(points[(n-1+imcap)%imcap][prevPointInd].pt,worldsize));
							for_ba_points.back().push_back(std::make_pair(points[n][curPointInd].pt,worldsize));
							keys[n][curPointInd]=worldsize;
							newpoints++;
							newp.push_back(pnts[0].size()-1);
							worldsize++;
							for_ba_ind.push_back(std::vector<std::pair<int,int> >());
							for_ba_ind.back().push_back(std::make_pair(for_ba_points.size()-2,for_ba_points[for_ba_points.size()-2].size()-1));
							for_ba_ind.back().push_back(std::make_pair(for_ba_points.size()-1,for_ba_points.back().size()-1));
						}
						else
						{
							for_ba_points.back().push_back(std::make_pair(points[n][curPointInd].pt,keys[(n-1+imcap)%imcap][prevPointInd]));
							for_ba_ind[keys[n][curPointInd]].push_back(std::make_pair(for_ba_points.size()-1,for_ba_points.back().size()-1));
							}
						}
#endif
						maskmatches[j][matchchain[(j-1+imcap)%imcap]]=1;
					}
				}
			}


			/*Mat immatch;
			for(int i=0;i<imcap;i++)
			{
				drawMatches(frame[(n-i+imcap)%imcap],points[(n-i+imcap)%imcap],frame[(n-i-1+imcap)%imcap],points[(n-i-1+imcap)%imcap],matches[i],
					immatch,cv::Scalar_<double>::all(-1),cv::Scalar_<double>::all(-1),maskmatches[i]);
				std::string name(i+1,'0');
				imshow(name,immatch);
			}*/


#ifdef CALIBRATION
			for(int i=0;i<imcap-1;i++)
				//for(int j=i+1;j<imcap;j++)
				{
					Mat mask;
					 F=findFundamentalMat(pnts[i],pnts[i+1],FM_RANSAC,0.65,0.99,mask);
					std::vector<Point2f> pp1,pp2;
					//std::cout<<F[i][j]<<"\n";
					if(!F.empty())
					{
						for(int k=0;k<pnts[i].size();k++)
						if(mask.at<bool>(k))
						{
							pp1.push_back(pnts[i][k]);
							pp2.push_back(pnts[i+1][k]);
						}
						 FF=findFundamentalMat(pp1,pp2,FM_8POINT);
					 double maxdist=0;
					for(int k=0;k<pnts[i].size();k++)
					{
						double err=computeFMError(pnts[i][k],pnts[i+1][k],FF);
						maxdist=max(maxdist,err*mask.at<bool>(k));
					}
					//std::cout<<SVD(F).vt.row(2)<<"\n";
					//std::cout<<SVD(FF).vt.row(2)<<"\n";
					trust.push_back(1.0/maxdist);
					Fund.push_back(FF);
					}
					//std::cout<<mask<<"\n";
					//std::cin.get();
				}
			double val=0;
			Mat Klast=K.clone();
			calibrateCipolla(trust,Fund,val,K,CalibrationType::NO_SKEW);
			std::cout<<K<<"\n"<<norm(K-Klast,NormTypes::NORM_L2)<<"\n"<<val<<"\n";
#endif
#ifdef BUNDLE_ADJ
			//if(!hasRound)
			//{
				for(int i=0;i<imcap;i++)
					perspectiveTransform(pnts[i],normpnts[i],K);
				for(int i=0;i<imcap-1;i++)
				{
					E=findEssentialMat(normpnts[i],normpnts[i+1]);
					recoverPose(E,normpnts[i],normpnts[i+1],R,t);
					GlobalR=GlobalR*R;
					Globalt=(Globalt+GlobalR*t);
					Mat P=GlobalR.t();
					P.push_back(Globalt.t());
					for_ba_cams.push_back(K*P.t());
					std::cout<<R<<"\n";
				}
			//}
			//else
			//{
			//	Mat A(2*(for_ba_points.back().size()-newpoints),12,CV_64F);
			//	A.setTo(0);
			//	double* pt=A.ptr<double>();
			//	int w=0;
			//	for(int i=0;i<for_ba_points.back().size();i++)
			//		if(for_ba_points.back()[i].second<for_ba_world.size())
			//		{
			//			w++;
			//			double* ptx=for_ba_world[for_ba_points.back()[i].second].ptr<double>();
			//			for(int j=0;j<4;j++)
			//			{
			//				*(pt+4+j)=-*(ptx+j);
			//				*(pt+8+j)=for_ba_points.back()[i].first.y**(ptx+j);
			//				*(pt+12+j)=*(ptx+j);
			//				*(pt+20+j)=-for_ba_points.back()[i].first.x**(ptx+j);
			//			}
			//			pt+=24;
			//		}
			//		//std::cout<<A<<"\n";
			//		Mat vt=SVD(A).vt;
			//		Mat pvec=vt.row(vt.rows-1);
			//		//std::cout<<A*pvec.t()<<"\n";
			//		Mat P=Mat(Size(4,3),CV_64F,pvec.ptr<double>());
			//		for_ba_cams.push_back(P.clone());
			//		std::cout<<K.inv()*P<<"\n";
			//}
			Mat P1=for_ba_cams[for_ba_cams.size()-2],P2=for_ba_cams.back();
			//std::cout<<P1<<' '<<P2<<"\n";
			for(int i=0;i<newp.size();i++)
			{
				Mat Ap=pnts[1][newp[i]].x*P1.row(2)-P1.row(0);
				Ap.push_back(pnts[1][newp[i]].y*P1.row(2)-P1.row(1));
				Ap.push_back(pnts[0][newp[i]].x*P2.row(2)-P2.row(0));
				Ap.push_back(pnts[0][newp[i]].y*P2.row(2)-P2.row(1));
				//std::cout<<Ap<<"\n";
				Mat X=SVD(Ap).vt.row(3);
				//std::cout<<X<<"\n";
				for_ba_world.push_back(X.t());
			}
			std::cout<<for_ba_points.size()<<" "<<for_ba_world.size()<<"\n";
#endif
		}
		else
		{
			#ifdef BUNDLE_ADJ
			for_ba_cams.push_back(K*Mat::eye(3,4,CV_64F));
           #endif
			}
		n++;
		hasRound=hasRound||(n==imcap);
		n-=((n==imcap)?n:0);
		if(for_ba_cams.size()==6)
		{
			BundleAdjuster bund(&for_ba_world,&for_ba_cams,&for_ba_points,&for_ba_ind);
			std::cout<<bund.adjust()<<"\n";
			for(int i=0;i<for_ba_cams.size();i++)
			std::cout<<for_ba_cams[i]<<"\n";
		}
		}
	}
	return 0;
}
