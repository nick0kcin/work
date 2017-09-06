#include "stdafx.h"
#if _FILE==4
#define _USE_MATH_DEFINES
#define DISABLE_OPENCV_24_COMPATIBILITY
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
#include "HypotEstim.h"
#include "buildImage.h"
//#include "ba.h"
#pragma comment(linker, "/STACK:16777216")


#define _OPTFLOW
//#define  _SURF
const int tripcap=5;

double timestep=1500.0;
double startpos=158000.0;
double minstep=1500.0;
double maxstep=1500.0;
double framepos=startpos;

double koeff_resol_x=0.5;
double koeff_resol_y=0.5;

Mat colframe[tripcap];
Mat* frame[tripcap]={0};


int maxcorners=200;

#ifdef _OPTFLOW
std::vector<Point2f> corners[tripcap][3];   //for goodFeaturesToTrack 
double qualitycorners=0.09;
double mindistance=30.0;


Mat matches[tripcap][2];
Size windowsize(30,30);

#endif

#ifdef _SURF
std::vector<KeyPoint> corners[tripcap];
Mat descriptors[tripcap];
Ptr<Feature2D> detector;
BFMatcher matcher;
std::vector<DMatch> goodmatches[tripcap];
std::vector<DMatch> matches[2];
#endif

std::vector<std::vector<Point2f> > triplet[tripcap];

std::vector<Mat> cams[tripcap];
std::vector<Mat> world[tripcap];

std::vector<int> link;
std::vector<int> linkinv;

std::vector<std::vector<std::pair<Point2f,int> > > for_ba_points;  //[world points][projection on each image
	std::vector<Mat> for_ba_cams; // camera matrixs
	std::vector<Mat> for_ba_world; //world points
	std::vector<char> isfixed;
	std::vector<std::vector<std::pair<int,int> > > for_ba_ind;



std::vector<Triangle> triangles[tripcap];
Mat Pano;
Mat GlobalCam;
Mat swap34(Matx44d(1,0,0,0,
	               0,1,0,0,
				   0,0,0,1,
				   0,0,1,0));
int main() 
{
	VideoCapture cap("v.mov");
	namedWindow("ff");
	Pano=Mat(1000,1000,CV_8UC3);
	Pano.setTo(0);
	GlobalCam=Mat(Matx34d(1, 0,0,0,
						  0, 1,0,0,
						  0,   0,  1,5));
	for(int i=0;i<tripcap;i++)
	triplet[i].assign(3,std::vector<Point2f>());
	for_ba_points.assign(4,std::vector<std::pair<Point2f,int> >());
	for_ba_cams.assign(4,Mat());
	cap.set(CAP_PROP_POS_MSEC,startpos);
	for(int i=0;i<tripcap;i++)
	{
		cams[i].assign(3,Mat());
	}
#ifdef _SURF
	detector= xfeatures2d::SURF::create(100,4,2,true,false);

#endif
	for(int iter=0;;iter=(iter+1)%tripcap,framepos+=timestep)
	{
		if(!frame[iter])
			frame[iter]=new Mat();
		Mat* previmage=frame[(iter+tripcap-1)%tripcap];
		Mat* prevprevimage=frame[(iter+tripcap-2)%tripcap];
		Mat* curimage=frame[iter];
		cap.set(CAP_PROP_POS_MSEC,framepos);
		cap>>colframe[iter];
		if(colframe[iter].empty())
			break;
		cv::resize(colframe[iter],colframe[iter],cv::Size(0,0),koeff_resol_x,koeff_resol_y);
		cv::cvtColor(colframe[iter],*curimage,CV_RGB2GRAY);

		#ifdef _OPTFLOW
			goodFeaturesToTrack(*curimage,corners[iter][0],maxcorners,qualitycorners,mindistance);
			cornerSubPix(*curimage,corners[iter][0],Size(5,5),Size(1,1),TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,200,1e-05));
		#endif
		#ifdef _SURF
			GaussianBlur(*curimage,*curimage,Size(0,0),2.0);
			detector->detect(*curimage,corners[iter]);
			//KeyPointsFilter::retainBest(corners[iter],maxcorners);
			detector->compute(*curimage,corners[iter],descriptors[iter]);
		#endif
		if(previmage!=0)
		{
			#ifdef _OPTFLOW
			calcOpticalFlowPyrLK(*previmage,*curimage,corners[(iter+tripcap-1)%tripcap][0],corners[(iter+tripcap-1)%tripcap][1],
				matches[(iter+tripcap-1)%tripcap][0],noArray(),windowsize,4);

			findFundMat(corners[(iter+tripcap-1)%tripcap][0],corners[(iter+tripcap-1)%tripcap][1],matches[(iter+tripcap-1)%tripcap][0],1.0,
				Rect2f(20,20,colframe[iter].cols-20,colframe[iter].rows-20));

				/*std::vector<KeyPoint> p1,p2;
			for(int i=0;i<matches[(iter+tripcap-1)%tripcap][0].rows;i++)
			if(matches[(iter+tripcap-1)%tripcap][0].at<bool>(i))
			{
			p1.push_back(KeyPoint(corners[(iter+tripcap-1)%tripcap][0][i],1.0,1.0));
			p2.push_back(KeyPoint(corners[(iter+tripcap-1)%tripcap][1][i],1.0,1.0));

			}
			std::vector<DMatch> mtc(p1.size(),DMatch());

			for(int i=0;i<mtc.size();i++)
			mtc[i]=DMatch(i,i,1.0);

			Mat drmt;
			drawMatches(*previmage,p1,*curimage,p2,mtc,drmt);
			imshow("mat",drmt);
			waitKey(5000);*/



			int numgood=countNonZero(matches[(iter+tripcap-1)%tripcap][0]);
			int numbad=matches[(iter+tripcap-1)%tripcap][0].rows-numgood;
			double ratio=(double)numbad/matches[(iter+tripcap-1)%tripcap][0].rows;
			double maxnorm=0;
			for(int i=0;i<corners[(iter+tripcap-1)%tripcap][0].size();i++)
				if(matches[(iter+tripcap-1)%tripcap][0].at<bool>(i))
					maxnorm=max(maxnorm,norm(corners[(iter+tripcap-1)%tripcap][0][i]-corners[(iter+tripcap-1)%tripcap][1][i]));
			if(ratio>0.5/*||maxnorm>windowsize.width*/)
			{
				framepos-=timestep;
				timestep/=2;
				timestep=max(timestep,minstep);
				iter=(iter-1)%tripcap;
				continue;
			}
			if(maxnorm<0.6*windowsize.width)
			{
				timestep*=2;
				timestep=min(timestep,maxstep);
			}
			else
				if(maxnorm<windowsize.width)
				{
					timestep*=windowsize.width/maxnorm;
					timestep=min(timestep,maxstep);
				}
				#endif
			#ifdef _SURF
				matcher.match(descriptors[iter],descriptors[(iter+tripcap-1)%tripcap],matches[0]);
				matcher.match(descriptors[(iter+tripcap-1)%tripcap],descriptors[iter],matches[1]);
				filterMatches(corners[iter],corners[(iter+tripcap-1)%tripcap],matches[0],matches[1],goodmatches[(iter+tripcap-1)%tripcap]);
				//Mat immatch;
				//drawMatches(*previmage,corners[iter],*curimage,corners[(iter+tripcap-1)%tripcap],goodmatches[(iter+tripcap-1)%tripcap],immatch);
				//imshow("fdfs",immatch);
				//waitKey(5000);
			#endif
		}
		if(prevprevimage!=0)
		{
			for(int i=0;i<3;i++)
				triplet[(iter+tripcap-2)%tripcap][i].clear();
			#ifdef _OPTFLOW
			calcOpticalFlowPyrLK(*previmage,*curimage,corners[(iter+tripcap-2)%tripcap][1],corners[(iter+tripcap-2)%tripcap][2],
				matches[(iter+tripcap-2)%tripcap][1],noArray(),windowsize,4);

			findFundMat(corners[(iter+tripcap-2)%tripcap][1],corners[(iter+tripcap-2)%tripcap][2],matches[(iter+tripcap-2)%tripcap][1],1.0,
				Rect2f(20,20,colframe[iter].cols-20,colframe[iter].rows-20));

			bool * ptrmatch1=matches[(iter+tripcap-2)%tripcap][0].ptr<bool>();
			bool * ptrmatch2=matches[(iter+tripcap-2)%tripcap][1].ptr<bool>();
			for(int i=0;i<matches[(iter+tripcap-2)%tripcap][0].rows;i++)
			{
				if(*ptrmatch1&&*ptrmatch2)
					for(int j=0;j<3;j++)
						triplet[(iter+tripcap-2)%tripcap][j].push_back(corners[(iter+tripcap-2)%tripcap][j][i]);
				ptrmatch1++,ptrmatch2++;
			}
			#endif
			#ifdef _SURF
			for(int i=0;i<goodmatches[(iter+tripcap-2)%tripcap].size();i++)
			{
				int quind=goodmatches[(iter+tripcap-2)%tripcap][i].queryIdx;
				for(int j=0;j<goodmatches[(iter+tripcap-1)%tripcap].size();j++)
					if(quind==goodmatches[(iter+tripcap-1)%tripcap][j].trainIdx)
					{
						triplet[(iter+tripcap-2)%tripcap][0].push_back(corners[(iter+tripcap-2)%tripcap][goodmatches[(iter+tripcap-2)%tripcap][i].trainIdx].pt);
						triplet[(iter+tripcap-2)%tripcap][1].push_back(corners[(iter+tripcap-1)%tripcap][quind].pt);
						triplet[(iter+tripcap-2)%tripcap][2].push_back(corners[iter][goodmatches[(iter+tripcap-1)%tripcap][j].queryIdx].pt);
					}
			}
			#endif
			world[(iter+tripcap-2)%tripcap].assign(triplet[(iter+tripcap-2)%tripcap][0].size(),Mat());
			factorisation(triplet[(iter+tripcap-2)%tripcap],cams[(iter+tripcap-2)%tripcap],world[(iter+tripcap-2)%tripcap]);
			//afiineFactorisation(triplet[(iter+tripcap-2)%tripcap],cams[(iter+tripcap-2)%tripcap],world[(iter+tripcap-2)%tripcap]);


			Mat X1,X2;
				triangulatePoints(cams[(iter+tripcap-2)%tripcap][0],cams[(iter+tripcap-2)%tripcap][1],
					triplet[(iter+tripcap-2)%tripcap][0],triplet[(iter+tripcap-2)%tripcap][1],X1);
				triangulatePoints(cams[(iter+tripcap-2)%tripcap][1],cams[(iter+tripcap-2)%tripcap][2],
					triplet[(iter+tripcap-2)%tripcap][1],triplet[(iter+tripcap-2)%tripcap][2],X2);

				convertPointsFromHomogeneous(X1.t(),X1);
				convertPointsFromHomogeneous(X2.t(),X2);
				X1=X1.reshape(1,0);
				X2=X2.reshape(1,0);
				for(int i=0;i<world[(iter+tripcap-2)%tripcap].size();i++)
				{
					//Mat XX;
					//convertPointsFromHomogeneous(world[(iter+tripcap-2)%tripcap][i].t(),XX);
					//XX=XX.reshape(1,0);
					//std::cout<<X1.row(i)<<"\n"<<X2.row(i)<<"\n"<<XX<<"\n";
					//std::cout<<"==================================\n";
					for(int j=0;j<3;j++)
					{
						double det=determinant(cams[(iter+tripcap-2)%tripcap][j].colRange(0,3));
						double nr=norm(cams[(iter+tripcap-2)%tripcap][j](Range(2,3),Range(0,3)));
						double w=cams[(iter+tripcap-2)%tripcap][j].row(2).dot(world[(iter+tripcap-2)%tripcap][i].t());
						double depth=det*w*world[(iter+tripcap-2)%tripcap][i].at<double>(3);
						std::cout<<depth<<" ";
						//if(depth<0)
						//	world[(iter+tripcap-2)%tripcap][i].at<double>(3)*=-1;
					}
					std::cout<<"\n";
					std::cin.get();
				}


			/*for(int i=0;i<3;i++)
				std::cout<<cams[(iter+tripcap-2)%tripcap][i]<<"\n";
			std::cout<<"+++++++++++++\n";
			for(int i=0;i<world[(iter+tripcap-2)%tripcap].size();i++)
			std::cout<<world[(iter+tripcap-2)%tripcap][i]<<"\n";*/

			//std::vector<KeyPoint> p1,p2;
			//for(int i=0;i<triplet[(iter+tripcap-2)%tripcap][0].size();i++)
			//{
			//p1.push_back(KeyPoint(triplet[(iter+tripcap-2)%tripcap][0][i],1.0,1.0));
			//p2.push_back(KeyPoint(triplet[(iter+tripcap-2)%tripcap][2][i],1.0,1.0));

			//}
			//std::vector<DMatch> mtc;
			//	for(int j=0;j<world[(iter+tripcap-2)%tripcap].size();j++)
			//	{
			//		double mx=0;
			//		for(int i=0;i<3;i++)
			//{
			//		Mat pn=cams[(iter+tripcap-2)%tripcap][i]*world[(iter+tripcap-2)%tripcap][j];
			//		//std::cout<<pn.at<double>(2)<<"\n";
			//		pn/=pn.at<double>(2);
			//		mx=max(mx,
			//		(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)*(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)
			//		+(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y)*(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y));

			//	}
			//		if(mx>1.0)
			//		{
			//			mtc.push_back(DMatch(j,j,1.0));
			//		}
			//}

			//	Mat drmt;
			//drawMatches(*frame[(iter+tripcap-2)%tripcap],p1,*curimage,p2,mtc,drmt);
			//imshow("mat",drmt);
			//waitKey(1000);






			if(!cams[(iter+tripcap-3)%tripcap][1].empty())
			{
				Mat H1=P2EHomo(cams[(iter+tripcap-3)%tripcap][1]);
				Mat H2=P2EHomo(cams[(iter+tripcap-2)%tripcap][0]);
				std::cout<<determinant(cams[(iter+tripcap-2)%tripcap][0](Range(0,3),Range(0,3)))<<"\n";
				std::cout<<determinant(cams[(iter+tripcap-3)%tripcap][1](Range(0,3),Range(0,3)))<<"\n";
				std::cout<<"detH1="<<determinant(H1)<<"\n";
				std::cout<<"detH2="<<determinant(H2)<<"\n";
				//std::cout<<cams[(iter+tripcap-3)%tripcap][1]*H1<<"\n";
				//std::cout<<cams[(iter+tripcap-2)%tripcap][0]*H2<<"\n";
				Mat P1=cams[(iter+tripcap-3)%tripcap][2]*H1;
				Mat P2=cams[(iter+tripcap-2)%tripcap][1]*H2;
				//std::cout<<P1<<"\n"<<P2<<"\n";
				Mat Hl=computeElation(P1,P2);
				//std::cout<<"detHl="<<determinant(Hl)<<"\n";
				//std::cout<<norm(P1.col(3).cross(P2.col(3)))/norm(P1.col(3))/norm(P2.col(3))<<"\n";
				//std::cout<<norm((P2*Hl).col(3))<<"\n";
				Mat H=H2*Hl*H1.inv();
				//H/=norm(H);
				for(int i=0;i<3;i++)
					cams[(iter+tripcap-2)%tripcap][i]*=H;
				std::cout<<"detH="<<determinant(H)<<"\n";
				Mat Hinv=H.inv(DECOMP_SVD);
				for(int i=0;i<world[(iter+tripcap-2)%tripcap].size();i++)
					world[(iter+tripcap-2)%tripcap][i]=Hinv*world[(iter+tripcap-2)%tripcap][i];


				link.assign(triplet[(iter+tripcap-3)%tripcap][1].size(),-1);
				linkinv.assign(triplet[(iter+tripcap-2)%tripcap][0].size(),-1);
				for(int i=0;i<triplet[(iter+tripcap-3)%tripcap][1].size();i++)
				{
					for(int j=0;j<triplet[(iter+tripcap-2)%tripcap][0].size();j++)
						if(norm(triplet[(iter+tripcap-2)%tripcap][0][j]-triplet[(iter+tripcap-3)%tripcap][1][i])<=1&&
							norm(triplet[(iter+tripcap-2)%tripcap][1][j]-triplet[(iter+tripcap-3)%tripcap][2][i])<=1)
						{
							link[i]=j;
							linkinv[j]=i;
							auto p1=(triplet[(iter+tripcap-2)%tripcap][0][j]+triplet[(iter+tripcap-3)%tripcap][1][i])/2;
							auto p2=(triplet[(iter+tripcap-2)%tripcap][1][j]+triplet[(iter+tripcap-3)%tripcap][2][i])/2;
							triplet[(iter+tripcap-2)%tripcap][0][j]=p1;
							triplet[(iter+tripcap-3)%tripcap][1][i]=p1;
							triplet[(iter+tripcap-2)%tripcap][1][j]=p2;
							triplet[(iter+tripcap-3)%tripcap][2][i]=p2;
						}
				}



				std::cout<<"cam1="<<norm(cams[(iter+tripcap-3)%tripcap][1]-cams[(iter+tripcap-2)%tripcap][0])<<"\n";
				//std::cout<<"cam1="<<cams[(iter+tripcap-2)%tripcap][0]<<"\n";
				std::cout<<"cam2="<<norm(cams[(iter+tripcap-3)%tripcap][2]-cams[(iter+tripcap-2)%tripcap][1])<<"\n";
				std::cout<<"cam2="<<norm(cams[(iter+tripcap-2)%tripcap][1])<<"\n";


				for_ba_cams[0]=(cams[(iter+tripcap-3)%tripcap][0].clone());
						for_ba_cams[1]=((cams[(iter+tripcap-3)%tripcap][1]+cams[(iter+tripcap-2)%tripcap][0])/2);
						for_ba_cams[2]=((cams[(iter+tripcap-3)%tripcap][2]+cams[(iter+tripcap-2)%tripcap][1])/2);
						for_ba_cams[3]=(cams[(iter+tripcap-2)%tripcap][2].clone());
						for(int i=0;i<4;i++)
						{
							Mat K,R,t;
							decomposeProjectionMatrix(for_ba_cams[i],K,R,t);
							std::cout<<K<<"\n";
							Mat cam(10,1,CV_64F);
							Rodrigues(R,cam.rowRange(0,3));
							for(int j=0;j<3;j++)
								cam.at<double>(3+j)=t.at<double>(j);
							cam.at<double>(6)=K.at<double>(0,0);
							cam.at<double>(7)=K.at<double>(1,1);
							cam.at<double>(8)=K.at<double>(0,2);
							cam.at<double>(9)=K.at<double>(1,2);
							for_ba_cams[i]=cam.clone();
							for_ba_points[i].clear();
						}


						for_ba_world.clear();
						for_ba_ind.clear();

				for(int i=0;i<link.size();i++)
					if(link[i]>=0)
					{
						Mat w1,w2;
						w1=(world[(iter+tripcap-3)%tripcap][i]+world[(iter+tripcap-2)%tripcap][link[i]])/2;
						//w2=world[(iter+tripcap-2)%tripcap][link[i]];
						convertPointsFromHomogeneous(w1.t(),w1);
						//convertPointsFromHomogeneous(w2.t(),w2);
						w1=w1.reshape(1,0);
						//w2=w2.reshape(1,0);
						//std::cout<<w1<<"\n"<<w2<<"\n";
						//std::cout<<norm(world[(iter+tripcap-3)%tripcap][i].t())<<"\n"<<norm(world[(iter+tripcap-2)%tripcap][link[i]].t())<<"\n";
						for_ba_world.push_back(w1.clone());
						for_ba_points[0].push_back(std::make_pair(triplet[(iter+tripcap-3)%tripcap][0][i],for_ba_world.size()-1));
						for_ba_points[1].push_back(std::make_pair(
							(triplet[(iter+tripcap-3)%tripcap][1][i]+triplet[(iter+tripcap-2)%tripcap][0][link[i]])/2
							,for_ba_world.size()-1));
						for_ba_points[2].push_back(std::make_pair(
							(triplet[(iter+tripcap-3)%tripcap][2][i]+triplet[(iter+tripcap-2)%tripcap][1][link[i]])/2
							,for_ba_world.size()-1));
						for_ba_points[3].push_back(
							std::make_pair(triplet[(iter+tripcap-2)%tripcap][2][link[i]],for_ba_world.size()-1));
						for_ba_ind.push_back(std::vector<std::pair<int,int> >(4,std::pair<int,int>()));
						for(int j=0;j<4;j++)
						for_ba_ind.back()[j]=std::make_pair(j,for_ba_points[j].size()-1);
					}
					else
					{
						Mat w1=world[(iter+tripcap-3)%tripcap][i];
						convertPointsFromHomogeneous(w1.t(),w1);
						w1=w1.reshape(1,0);
						for_ba_world.push_back(w1.clone());
						for(int j=0;j<3;j++)
							for_ba_points[j].push_back(std::make_pair(triplet[(iter+tripcap-3)%tripcap][j][i],for_ba_world.size()-1));
						for_ba_ind.push_back(std::vector<std::pair<int,int> >(3,std::pair<int,int>()));
						for(int j=0;j<3;j++)
						for_ba_ind.back()[j]=std::make_pair(j,for_ba_points[j].size()-1);
					}
					for(int i=0;i<linkinv.size();i++)
						if(linkinv[i]==-1)
						{
							Mat w1=world[(iter+tripcap-2)%tripcap][i];
							convertPointsFromHomogeneous(w1.t(),w1);
							w1=w1.reshape(1,0);
							for_ba_world.push_back(w1.clone());
						for(int j=1;j<4;j++)
							for_ba_points[j].push_back(std::make_pair(triplet[(iter+tripcap-2)%tripcap][j-1][i],for_ba_world.size()-1));
						for_ba_ind.push_back(std::vector<std::pair<int,int> >(3,std::pair<int,int>()));
						for(int j=1;j<4;j++)
						for_ba_ind.back()[j-1]=std::make_pair(j,for_ba_points[j].size()-1);
						}
						//for(int i=0;i<4;i++)
						//	for_ba_cams[i]=for_ba_cams[i].reshape(0,12);
						

						BundleAdjuster adjuster(&for_ba_world,&for_ba_cams,&for_ba_points,&for_ba_ind,0,30);
						Mat res=adjuster.adjust();
						for(int i=0;i<3;i++)
						{
							cams[(iter+tripcap-3)%tripcap][i]=composeProjectionMatrix(for_ba_cams[i]);
							cams[(iter+tripcap-2)%tripcap][i]=composeProjectionMatrix(for_ba_cams[i+1]);
						cams[(iter+tripcap-3)%tripcap][i]/=norm(cams[(iter+tripcap-3)%tripcap][i].row(2));
						cams[(iter+tripcap-2)%tripcap][i]/=norm(cams[(iter+tripcap-2)%tripcap][i].row(2));
						}
						for(int i=0;i<for_ba_points[0].size();i++)
						{
							Mat w;
							convertPointsToHomogeneous(for_ba_world[i],w);
							world[(iter+tripcap-3)%tripcap][i]=w.reshape(1,4);
						}
						int wpos=for_ba_points[0].size();
						int k=0;
						for(int i=0;i<for_ba_points[3].size();i++)
							if(linkinv[i]>=0)
						{
							Mat w;
							convertPointsToHomogeneous(for_ba_world[linkinv[i]],w);
							world[(iter+tripcap-2)%tripcap][i]=w.reshape(1,4);
						}
						for(int i=0;i<for_ba_points[3].size();i++)
							if(linkinv[i]==-1)
							{
								Mat w;
								convertPointsToHomogeneous(for_ba_world[wpos++],w);
								world[(iter+tripcap-2)%tripcap][i]=w.reshape(1,4);
							}
					//	std::cout<<norm(res)<<"\n"<<res<<"\n";



				std::cout<<"cam="<<cams[(iter+tripcap-2)%tripcap][2]<<"\n";
				triangles[(iter+tripcap-3)%tripcap].clear();
				buildMesh(triplet[(iter+tripcap-3)%tripcap][2],triangles[(iter+tripcap-3)%tripcap]);
				projectImage(Pano,colframe[(iter+tripcap-1)%tripcap],triplet[(iter+tripcap-3)%tripcap][2],triangles[(iter+tripcap-3)%tripcap],world[(iter+tripcap-3)%tripcap],
					/*Mat(Matx33d(1,0,0,0,1,0,0,0,5))*cams[(iter+tripcap-2)%tripcap][2]*/GlobalCam);
			}
			else
			{
				Mat H=P2EHomo(cams[(iter+tripcap-2)%tripcap][1]);
				Mat Hl=Mat::eye(4,4,CV_64F);
				for(int i=0;i<3;i++)
					cams[(iter+tripcap-2)%tripcap][i]*=H;
				Hl.at<double>(3,3)/=norm(cams[(iter+tripcap-2)%tripcap][2].col(3));
				for(int i=0;i<3;i++)
					cams[(iter+tripcap-2)%tripcap][i]*=Hl;
				H*=Hl;
				Mat Hinv=H.inv(DECOMP_SVD);
				for(int i=0;i<world[(iter+tripcap-2)%tripcap].size();i++)
					world[(iter+tripcap-2)%tripcap][i]=Hinv*world[(iter+tripcap-2)%tripcap][i];


				buildMesh(triplet[(iter+tripcap-2)%tripcap][1],triangles[0]);
				projectImage(Pano,colframe[1],triplet[(iter+tripcap-2)%tripcap][1],triangles[0],world[(iter+tripcap-2)%tripcap],GlobalCam);

			}





			double normerr=0;
			double maxerr=0;
			for(int i=0;i<3;i++)
			{
				for(int j=0;j<world[(iter+tripcap-2)%tripcap].size();j++)
				{
					Mat pn=cams[(iter+tripcap-2)%tripcap][i]*world[(iter+tripcap-2)%tripcap][j];
					//std::cout<<pn.at<double>(2)<<"\n";
					pn/=pn.at<double>(2);
					normerr+=(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)*(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)+
						(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y)*(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y);
					maxerr=max(maxerr,
					(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)*(pn.at<double>(0)-triplet[(iter+tripcap-2)%tripcap][i][j].x)
					+(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y)*(pn.at<double>(1)-triplet[(iter+tripcap-2)%tripcap][i][j].y));

				}
			}

			std::cout<<"t="<<framepos<<" norm="<<sqrt(normerr)<<" with "<<triplet[(iter+tripcap-2)%tripcap][0].size()<<" points|max="<<maxerr<<"\n";
			imshow("pano",Pano);
			Pano.setTo(0);
			waitKey(1000);
		}
	}
	for(int i=0;i<tripcap;i++)
		if(frame[i])
			delete frame[i];
	return 0;
}
#endif