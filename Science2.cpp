#include "stdafx.h"
#if _FILE==2
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
//#include "ba.h"
#pragma comment(linker, "/STACK:16777216")

//#define CALIBRATION
#define BUNDLE_ADJ 
//#define EXPERIMENT1



using namespace cv;
const int imcap=2;
double timestep=1500.0;
double framepos=158000.0;
double startpos=framepos;



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
Point2f projectFromCam2Cam(Mat P,Mat H,Point2f p)
{
	Mat pp(Matx31d(p.x,p.y,1));
	Mat r=H*pp;
	r.push_back(1.0);
	r=P*r;
	double* r_=r.ptr<double>();
	int x=r_[0]/r_[2],y=r_[1]/r_[2];
	return Point2f(x,y);
}
Mat crossMatrix(Mat t)
{
	double x=t.at<double>(0),y=t.at<double>(1),z=t.at<double>(2);
	return Mat(Matx33d( 0,-z,y,
		z,0,-x,
		-y,x,0));
}



struct  Line
{
	Point2f p1,p2;
	Point2f del;
	double len;
	int w1,w2;
	Line(Point2f st,Point2f end,int world1,int world2)
	{
		p1=st;
		p2=end;
		del=p2-p1;
		len=sqrt(del.ddot(del));
		w1=world1;
		w2=world2;
	};
	double dist(Point2f p)
	{
		double cr=del.cross(p-p2)/len;
		double d1=sqrt((p-p1).ddot(p-p1));
		double d2=sqrt((p-p2).ddot(p-p2));
		if(d1>d2)
			swap(d1,d2);
		int sg=((cr<0)?-1:1);
		//if((d1<=abs(cr)&&abs(cr)<=d2))
		//	return cr;
		//else
		return d1;
	};
};

struct Triangle
{
	Point2f p1,p2,p3;
	int wp1,wp2,wp3;
	int image;
	Triangle(Line l,Point p,int w,int im)
	{
		p1=l.p1;
		p2=l.p2;
		p3=p;
		wp1=l.w1;
		wp2=l.w2;
		wp3=w;
		image=im;
	};
	Triangle(Point2f p1_,Point2f p2_,Point2f p3_,int wp1_,int wp2_,int wp3_,int im)
	{
		p1=p1_;
		p2=p2_;
		p3=p3_;
		wp1=wp1_;
		wp2=wp2_;
		wp3=wp3_;
		image=im;
	};
	void estimHomo(Mat& h)
	{
	};
};

std::vector<std::pair<Point2f,int> > qLine;


Mat frame[imcap];
Mat inp[imcap];

std::vector<cv::KeyPoint> points[imcap];
UMat descriptors[imcap];
std::vector<cv::DMatch> matches[imcap];
std::vector<char> maskmatches[imcap];
int matchchain[imcap];
std::vector<DMatch> goodmatches;
std::vector<std::vector<Point2f> > pnts(imcap,std::vector<Point2f>());
std::vector<int> indpnts[imcap];
#ifdef CALIBRATION
std::vector<Mat> Fund;
std::vector<double> trust;
Mat F,FF;
#endif
#ifdef BUNDLE_ADJ
std::vector<std::vector<std::pair<Point2f,int> > > for_ba_points[2];  //[world points][projection on each image
std::vector<Mat> for_ba_cams; // camera matrixs
std::vector<int> keys[imcap]; // i-th keypoint in j image is projection keys[i][j]-th world point
std::vector<Mat> for_ba_world[2]; //world points
std::vector<char> isfixed;
std::vector<std::vector<std::pair<int,int> > > for_ba_ind;
std::vector<std::vector<Point2f> > normpnts(imcap,std::vector<Point2f>());
std::vector<Triangle> triangles;
std::vector<Mat>  images;
Mat R,t,E;
Mat GlobalR=Mat::zeros(3,1,CV_64F);
Mat Globalt=Mat::zeros(3,1,CV_64F);
std::vector<short> newp;
int buff=0;
int npan=4;
bool newframe=0;
Mat pano[31];
Mat Pano(1000,1000,CV_8UC3);
/*	Pano.setTo(0);
for(int i=0;i<npan;i++)
pano[i]=Mat(1000,1000,CV_8UC3);*/
#endif

BFMatcher matcher;
Ptr<Feature2D> detector = xfeatures2d::SURF::create(100,4,2,true);
Ptr<Feature2D> descriptor = xfeatures2d::SURF::create();

#ifdef BUNDLE_ADJ
/*Mat K=Mat(Matx33d(1200, 0, 495,0, 1200, 225,0, 0, 1));
Mat Kvec=Mat(Matx41d(1200,1200,495,225));*/
Mat K=Mat(Matx33d(1200, 0, 500,0, 1200, 250,0, 0, 1));
Mat Kvec=Mat(Matx41d(1200,1200, 500 ,250));
#endif

int n=0;
bool hasRound=0;




#ifdef BUNDLE_ADJ
void buildMesh()
{
	float mx=10000,my=10000;
	int ind=-1;
	for(int i=0;i<keys[1-n].size();i++)
		if(keys[1-n][i]!=-1)
		{
			if(points[1-n][i].pt.x==mx&&points[1-n][i].pt.y<=my)
			{
				my=points[1-n][i].pt.y;
				ind=i;
			}
			else
				if(points[1-n][i].pt.x<mx)
				{
					mx=points[1-n][i].pt.x;
					my=10000;
					ind =i;
				}
		}
		float mindist=1000000;
		int ind2=-1;
		Point2f pp=points[1-n][ind].pt;
		for(int i=0;i<keys[1-n].size();i++)
			if(i!=ind&&keys[1-n][i]!=-1)
			{
				float dst=(pp-points[1-n][i].pt).ddot(pp-points[1-n][i].pt);
				if(mindist>dst)
				{
					mindist=dst;
					ind2=i;
				}
			}
			qLine.clear();
			qLine.push_back(std::make_pair(points[1-n][ind].pt,keys[1-n][ind]));
			qLine.push_back(std::make_pair(points[1-n][ind2].pt,keys[1-n][ind2]));
			keys[1-n][ind2]=-1;
			keys[1-n][ind]=-1;
			while(1)
				//for(int j=0;j<81;j++)
			{
				/*					if(j==50)
				{
				int stop=0;
				}*/
				double mind=100000;
				int cur=-1;
				for(int i=0;i<points[1-n].size();i++)
					if(keys[1-n][i]!=-1)
					{
						double dst=1000000;
						Point2f pp=points[1-n][i].pt;
						for(auto it=qLine.begin();it!=qLine.end();it++)
							dst=min(dst,(it->first-pp).ddot(it->first-pp));
						if(dst==0)
						{
							keys[1-n][i]=-1;
							break;
						}
						if(dst<mind)
						{
							mind=dst;
							cur=i;
						}
					}	
					if(cur!=-1)
					{
						Point2f p=points[1-n][cur].pt;
						bool c=0;
						auto st=qLine.begin();
						auto en=qLine.begin();
						for(auto it=qLine.begin();it!=qLine.end();)
						{
							auto it2=it;
							it2++;
							if(it2==qLine.end())
								it2=qLine.begin();
							auto np=it2->first-it->first;
							auto n2=p-it2->first;
							double cr=np.cross(n2);
							double dot=np.ddot(n2);
							if(cr>1e-06||(abs(cr)<1e-06&&dot>0))
							{
								if(!c)
								{
									st=it;
									//qLine.insert(st,std::make_pair(points[1-n][cur].pt,keys[1-n][cur]));
									c=1;
								}
								en=it;
								triangles.push_back(Triangle(it->first,it2->first,p,it->second,it2->second,keys[1-n][cur],images.size()-2));
								it++;
							}
							else
								it++;
						}
						if(c)
						{
							auto rem=st;
							st++;
							en++;
							qLine.erase(st,en);
							qLine.insert(st,std::make_pair(points[1-n][cur].pt,keys[1-n][cur]));
						}
						keys[1-n][cur]=-1;
					}
					else
						break;
			}

			/*Mat iml=images.back().clone();

			for(int i=0;i<triangles.size();i++)
			if(triangles[i].image==images.size()-2)
			{
			line(iml,triangles[i].p1,triangles[i].p2,CV_RGB(255,0,0));
			line(iml,triangles[i].p2,triangles[i].p3,CV_RGB(255,0,0));
			line(iml,triangles[i].p1,triangles[i].p3,CV_RGB(255,0,0));
			circle(iml,triangles[i].p1,4,CV_RGB(0,255,0));
			circle(iml,triangles[i].p2,4,CV_RGB(0,255,0));
			circle(iml,triangles[i].p3,4,CV_RGB(0,255,0));
			}

			imshow("omnomnom",iml);*/
}
#endif

int main() // калибровка
{

	VideoCapture cap("v.mov");
	namedWindow("ff");
	cap.set(CAP_PROP_POS_MSEC,startpos);

	for(int i=0;i<imcap;i++)
		pnts[i].reserve(1000);
#ifdef BUNDLE_ADJ
	Mat GlobalH=Mat::eye(4,4,CV_64F);
	Pano.setTo(0);
	Mat KK,RR,TT;
	RR=Mat::zeros(1,3,CV_64F);
	TT=Mat::zeros(1,3,CV_64F);
	KK=K.clone();
	KK.at<double>(0,0)/=4;
	KK.at<double>(1,1)/=4;
#endif
	double width=cap.get(CAP_PROP_FRAME_WIDTH);
	double height=cap.get(CAP_PROP_FRAME_HEIGHT);

#ifdef CALIBRATION
	Mat K=Mat(Matx33d(1,0,width/4.0,0,1,height/4.0,0,0,1));
#endif

#ifdef BUNDLE_ADJ
	/*Mat K=Mat(Matx33d(1200, 0, 495,0, 1200, 225,0, 0, 1));
	Mat Kvec=Mat(Matx41d(1200,1200,495,225));*/
	//K=Mat(Matx33d(1200, 0, width/4,0, 1200, height/4,0, 0, 1));
	//Kvec=Mat(Matx41d(1200,1200, width/4 ,height/4));
#endif


	while(1)
	{
		char c= waitKey(500);
		//	if(c==' ')
		{
			std::cout<<"]=>\n";
			cap.set(CAP_PROP_POS_MSEC,framepos);
			cap>>frame[n];
			framepos+=timestep;
			cv::resize(frame[n],frame[n],cv::Size(0,0),0.5,0.5);
#ifdef BUNDLE_ADJ
			images.push_back(frame[n].clone());
#endif
			cv::cvtColor(frame[n],inp[n],CV_RGB2GRAY);
			//GaussianBlur(inp[n],inp[n],Size(0,0),2);
			detector->detect(inp[n],points[n]);
			KeyPointsFilter::removeDuplicated(points[n]);
			KeyPointsFilter::runByImageBorder(points[n],Size(frame[n].cols,frame[n].rows),10);
			KeyPointsFilter::retainBest(points[n],max(min(int(points[n].size()*0.5),500),250));
			detector->compute(inp[n],points[n],descriptors[n]);
#ifdef BUNDLE_ADJ
			keys[n].assign(points[n].size(),-1);
			int newpoints=0;
			newp.clear();
			int worldsize=for_ba_world[buff].size();
			for_ba_points[buff].push_back(std::vector<std::pair<Point2f,int> >());
#endif
			if(hasRound||n+1==imcap)
			{
				for(int i=0;i<imcap;i++)
				{
					matcher.match(descriptors[(n-i+imcap)%imcap],descriptors[(n-i-1+imcap)%imcap],matches[i]);
					pnts[i].clear();
					indpnts[i].clear();
					maskmatches[i].assign(matches[i].size(),0);
				}
				float distmin=1000000;
				for(int i=0;i<matches[0].size();i++)
					distmin=min(distmin,matches[0][i].distance);
				for(int i=0;i<matches[1].size();i++)
					distmin=min(distmin,matches[1][i].distance);
				//std::cout<<distmin<<"\n";
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
							Point2f pt=points[(n-j+imcap)%imcap][matches[j][matchchain[(j-1+imcap)%imcap]].queryIdx].pt;
							pnts[j].push_back(pt);
							indpnts[j].push_back(matches[j][matchchain[(j-1+imcap)%imcap]].queryIdx);
							maskmatches[j][matchchain[(j-1+imcap)%imcap]]=1;
						}
					}
				}

				Mat immatch;
				for(int i=0;i<imcap-1;i++)
				{
					drawMatches(frame[(n-i+imcap)%imcap],points[(n-i+imcap)%imcap],frame[(n-i-1+imcap)%imcap],points[(n-i-1+imcap)%imcap],matches[i],
						immatch,cv::Scalar_<double>::all(-1),cv::Scalar_<double>::all(-1),maskmatches[i]);
					std::string name(i+1,'0');
					imshow(name,immatch);
				}
				//	waitKey(100000);


#ifdef CALIBRATION
				for(int i=0;i<imcap-1;i++)
					//for(int j=i+1;j<imcap;j++)
				{
					Mat mask;
					F=findFundamentalMat(pnts[i],pnts[i+1],FM_RANSAC,3,0.99,mask);
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
							std::cout<<"F="<<F<<"\n";
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
				for(int i=0;i<Fund.size();i++)
				{
					Mat E=K.t()*Fund[i]*K;
					E/=norm(E);
					double* pE=(double*)E.data;
					for(int j=0;j<9;j++)
					{
						if(abs(*pE)<1e-04)
							*pE=0;
						pE++;
					}
					Mat t=SVD(E).u.col(2);
					std::cout<<t<<"\n";
				}
				std::cout<<"++++++++++\n";
				//std::cin.get();
				std::cout<<K<<"\n"<<norm(K-Klast,NormTypes::NORM_L2)<<"\n"<<val<<"\n";
#endif
#ifdef BUNDLE_ADJ
				Mat mask;
				PureMotionFundRansac ransac1(3.0,0.98,4000,&mask);
				//EuclidRansac ransac2(0.5,0.9,2000,&mask,images[0].cols/2,images[0].rows/2);
				planarRotationRansac ransac2(4.0,0.98,1800,&mask);
#ifdef EXPERIMENT1
				for(int i=0;i<pnts[1].size();i++)
				{
					pnts[1][i].x=pnts[0][i].x;
					pnts[1][i].y=pnts[0][i].y+15+7*randu<float>();
				}
				Mat E1=findFundamentalMat(pnts[0],pnts[1],FM_RANSAC,2,0.99,mask);
				E1/=norm(E1);
				E=ransac1.doJob(pnts[0],pnts[1]);
				E/=norm(E);
				std::cout<<E<<"\n";
				std::cout<<SVD(E).u.col(2).t()<<"\n";
				std::cout<<SVD(E).vt.row(2).t()<<"\n";

				std::cout<<"+++++++++++++++++\n";

				std::cout<<E1<<"\n";
				std::cout<<SVD(E1).u.col(2).t()<<"\n";
				std::cout<<SVD(E1).vt.row(2).t()<<"\n";
				for(int i=0;i<pnts[1].size();i++)
				{
					pnts[1][i].x+=-1+2*randu<float>();
					pnts[1][i].y+=-1+2*randu<float>();
				}
				E1=findFundamentalMat(pnts[0],pnts[1],FM_RANSAC,2,0.99,mask);
				E1/=norm(E1);
				E=ransac1.doJob(pnts[0],pnts[1]);
				E/=norm(E);
				std::cout<<E<<"\n";
				std::cout<<SVD(E).u.col(2).t()<<"\n";
				std::cout<<SVD(E).vt.row(2).t()<<"\n";

				std::cout<<"+++++++++++++++++\n";

				std::cout<<E1<<"\n";
				std::cout<<SVD(E1).u.col(2).t()<<"\n";
				std::cout<<SVD(E1).vt.row(2).t()<<"\n";
#endif
				std::vector<std::vector<Point2f> > goodpnts(2,std::vector<Point2f>());
				std::vector<double> ang;
				Mat world;
				std::vector<Mat> World;
				std::vector<Mat> Cam;
				std::vector<std::vector<std::pair<int,int> > > ind_;
				std::vector<std::vector<std::pair<Point2f,int> > > pnts_(2,std::vector<std::pair<Point2f,int> >());
				Mat P1,P2;
				double maxbaseline=0;
				for(int i=0;i<pnts[0].size();i++)
					maxbaseline=max(maxbaseline,sqrt((pnts[0][i]-pnts[1][i]).ddot(pnts[0][i]-pnts[1][i])));
				for(int i=0;i<imcap-1;i++)
				{
					E=ransac1.doJob(pnts[i],pnts[i+1]);
					int cnt=countNonZero(mask);
					double per=(double)cnt/pnts[i].size();
					std::cout<<per*100<<"%\n";
					if(per>0.85)
					{						
						R=Mat::eye(3,3,CV_64F);
						t=Mat(Matx31d(E.at<double>(2,1),E.at<double>(0,2),E.at<double>(1,0)));
					}
					else
					{
						E=ransac2.doJob(pnts[i],pnts[i+1]);
						ransac2.estimateMotion(E,R,t);
					}
					Mat rvec;
					Rodrigues(R,rvec);
					composeRT(GlobalR,Globalt,rvec,t,GlobalR,Globalt);
					Mat RR;
					Rodrigues(GlobalR,RR);

					Mat P=GlobalR.clone();
					P.push_back(Globalt);
					P.push_back(Kvec);
					for_ba_cams.push_back(P.clone());

					/*for(int j=0;j<pnts[i].size();j++)
					if(mask.at<short>(j))
					{
					goodpnts[0].push_back(pnts[i][j]);
					goodpnts[1].push_back(pnts[i+1][j]);
					mask.at<short>(j)=goodpnts[0].size();
					}

					composeProjectionMatrix(GlobalR,Globalt,K,P1);
					composeProjectionMatrix(R,t,K,P2);

					triangulatePoints(P1,P2,goodpnts[0],goodpnts[1],world);
					convertPointsFromHomogeneous(world.t(),world);
					world=world.reshape(1,0);
					for(int j=0;j<world.rows;j++)
					{
					Mat hot;
					world.row(j).convertTo(hot,CV_64F);
					World.push_back(hot);
					}
					Cam.push_back(Mat::zeros(6,1,CV_64F));
					Cam[0].push_back(Kvec);
					Cam.push_back(rvec);
					Cam[1].push_back(t);
					Cam[1].push_back(Kvec);
					ind_.assign(cnt,std::vector<std::pair<int,int> >(2));
					for(int j=0;j<ind_.size();j++)
					{
					ind_[j][0]=std::make_pair(0,j);
					ind_[j][1]=std::make_pair(1,j);
					}
					for(int j=0;j<cnt;j++)
					{
					pnts_[0].push_back(std::make_pair(goodpnts[0][j],j));
					pnts_[1].push_back(std::make_pair(goodpnts[1][j],j));
					}*/
					/*	BundleAdjuster bund(&World,&Cam,&pnts_,&ind_,&std::vector<char>(),12);
					double mres=bund.adjust();
					for_ba_cams.push_back(Cam[1].rowRange(0,6));
					for_ba_cams.back().push_back(Kvec);
					GlobalR=Cam[1].rowRange(0,3);
					Globalt=Cam[1].rowRange(3,6);
					Rodrigues(GlobalR,RR);
					std::cout<<RR<<"\n";
					std::cout<<mres<<"\n";*/
					//	std::cin.get();

				}
				//Mat P1,P2;
				Mat Pv1=for_ba_cams[for_ba_cams.size()-2];
				Mat r2vec=Pv1.rowRange(0,3);
				Mat t2=Pv1.rowRange(3,6);
				composeProjectionMatrix(r2vec,t2,K,P1);
				composeProjectionMatrix(GlobalR,Globalt,K,P2);



				std::cout<<"projective depths:\n";
				for(int i=0;i<pnts[0].size();i++)//prepare points  for bundle adjustment
					if(mask.at<short>(i))
					{
						int prevPointInd=indpnts[1][i];
						int curPointInd=indpnts[0][i];
						keys[n][curPointInd]=keys[(n-1+imcap)%imcap][prevPointInd]; 
						if(keys[(n-1+imcap)%imcap][prevPointInd]==-1)// if we meet point firstly
						{
							for_ba_points[buff][for_ba_points[buff].size()-2].push_back(std::make_pair(points[(n-1+imcap)%imcap][prevPointInd].pt,worldsize));
							for_ba_points[buff].back().push_back(std::make_pair(/*pnts[0][i]*/points[n][curPointInd].pt,worldsize));
							keys[n][curPointInd]=worldsize;
							keys[1-n][prevPointInd]=worldsize;
							newpoints++;


							Mat p1(Matx21d(pnts[1][i].x,pnts[1][i].y));
							Mat p2(Matx21d(pnts[0][i].x,pnts[0][i].y));
							Mat X;
							triangulatePoints(P1,P2,p1,p2,X);


							double det=determinant(P1.colRange(0,3));  //computing depth on 1 and 2 camera
							double w=P1.row(2).dot(X.t());
							double depth=det*w*X.at<double>(3);
							double det1=determinant(P2.colRange(0,3));
							double w1=P2.row(2).dot(X.t());
							double depth1=det1*w1*X.at<double>(3);
							std::cout<<depth<<' '<<depth1<<"\npress enter\n";
							std::cin.get();



							Mat Xx;
							convertPointsFromHomogeneous(X.t(),X);
							X=X.reshape(1,0);
							X.convertTo(Xx,CV_64F);
							for_ba_world[buff].push_back(Xx.clone());

							//for_ba_world[buff].push_back(World[mask.at<unsigned short>(i)-1].clone());

							worldsize++;
							isfixed.push_back(0);
							for_ba_ind.push_back(std::vector<std::pair<int,int> >());
							for_ba_ind.back().push_back(std::make_pair(for_ba_points[buff].size()-2,for_ba_points[buff][for_ba_points[buff].size()-2].size()-1));
							for_ba_ind.back().push_back(std::make_pair(for_ba_points[buff].size()-1,for_ba_points[buff].back().size()-1));
						}
						else
						{
							if(newframe)
							{
								for_ba_points[buff][0].push_back(std::make_pair(points[(n-1+imcap)%imcap][prevPointInd].pt,worldsize));
								for_ba_points[buff].back().push_back(std::make_pair(points[n][curPointInd].pt,worldsize));


								//for_ba_world[buff].push_back(for_ba_world[1-buff][keys[(n-1+imcap)%imcap][prevPointInd]]);

								Mat p1(Matx21d(pnts[1][i].x,pnts[1][i].y));
								Mat p2(Matx21d(pnts[0][i].x,pnts[0][i].y));
								Mat X;
								triangulatePoints(P1,P2,p1,p2,X);

								double det=determinant(P1.colRange(0,3));
								double w=P1.row(2).dot(X.t());
								double depth=det*w*X.at<double>(3);
								double det1=determinant(P2.colRange(0,3));
								double w1=P2.row(2).dot(X.t());
								double depth1=det1*w1*X.at<double>(3);
								std::cout<<depth<<' '<<depth1<<"\n";
								std::cin.get();

								Mat Xx;
								convertPointsFromHomogeneous(X.t(),X);
								X=X.reshape(1,0);
								X.convertTo(Xx,CV_64F);
								for_ba_world[buff].push_back(Xx.clone());

								//for_ba_world[buff].push_back(World[mask.at<short>(i)-1].clone());

								keys[(n-1+imcap)%imcap][prevPointInd]=worldsize;
								keys[n][curPointInd]=worldsize;
								worldsize++;

								isfixed.push_back(1);
								for_ba_ind.push_back(std::vector<std::pair<int,int> >());
								for_ba_ind.back().push_back(std::make_pair(for_ba_points[buff].size()-2,for_ba_points[buff][for_ba_points[buff].size()-2].size()-1));
								for_ba_ind.back().push_back(std::make_pair(for_ba_points[buff].size()-1,for_ba_points[buff].back().size()-1));
							}
							else
							{
								Mat p1(Matx21d(pnts[1][i].x,pnts[1][i].y));
								Mat p2(Matx21d(pnts[0][i].x,pnts[0][i].y));
								//std::cout<<Xx<<"{\n"<<for_ba_world[buff][keys[(n-1+imcap)%imcap][prevPointInd]]<<"}\n";
								for_ba_points[buff].back().push_back(std::make_pair(/*pnts[0][i]*/points[n][curPointInd].pt,keys[(n-1+imcap)%imcap][prevPointInd]));
								for_ba_ind[keys[n][curPointInd]].push_back(std::make_pair(for_ba_points[buff].size()-1,for_ba_points[buff].back().size()-1));
								//std::cin.get();
							}
						}
					}

					if(!newframe)
						buildMesh();

					std::cout<<for_ba_points[buff].size()<<" "<<for_ba_world[buff].size()<<"\n";
#endif
			}
			else
			{
#ifdef BUNDLE_ADJ
				Mat P=Mat::zeros(6,1,CV_64F);
				P.push_back(Kvec);
				for_ba_cams.push_back(P.clone());
#endif
			}
			n++;
			hasRound=hasRound||(n==imcap);
			n-=((n==imcap)?n:0);
#ifdef BUNDLE_ADJ
			if(newframe)
			{
				for_ba_world[1-buff].clear();
				for_ba_points[1-buff].clear();
			}
			newframe=0;
			if(for_ba_cams.size()==npan)
			{
				for(int i=0;i<for_ba_world[buff].size();i++)
				{
					std::cout<<for_ba_world[buff][i]<<"\n";
				}
				for(int i=0;i<for_ba_cams.size();i++)
				{
					Mat rot;
					Rodrigues(for_ba_cams[i].rowRange(0,3),rot);
					KK=Mat(Matx33d(for_ba_cams[i].at<double>(6)/4,0,for_ba_cams[i].at<double>(8),
						0,  for_ba_cams[i].at<double>(7)/4,for_ba_cams[i].at<double>(9),0,0,1));
					/*std::cout<<"R="<<rot<<"\n";
					std::cout<<"T="<<for_ba_cams[i].rowRange(3,6)<<"\n";*/
				}
				BundleAdjuster bund(&(for_ba_world[buff]),&for_ba_cams,&(for_ba_points[buff]),&for_ba_ind,&isfixed,15);
				//BundleAdjusterFM2 bund(&(for_ba_world[buff]),&for_ba_cams,&(for_ba_points[buff]),&for_ba_ind,&isfixed);
				Mat res=bund.adjust();
				//std::cout<<"res="<<res<<"\n";
				double* ptrres=res.ptr<double>();
				for(int i=0;i<for_ba_points[buff].size();i++)
				{
					double ni=0;
					for(int j=0;j<for_ba_points[buff][i].size();j++)
					{
						double xi=ptrres[0],yi=ptrres[1];
						ni+=sqrt(xi*xi+yi*yi);
						ptrres+=2;
					}
					std::cout<<ni/for_ba_points[buff][i].size()<<"\n";
				}
				Pano.setTo(0);
				for(int i=0;i<for_ba_cams.size();i++)
				{
					Mat rot;
					Rodrigues(for_ba_cams[i].rowRange(0,3),rot);
					KK=Mat(Matx33d(for_ba_cams[i].at<double>(6)/5,0,for_ba_cams[i].at<double>(8),
						0,  for_ba_cams[i].at<double>(7)/5,2*for_ba_cams[i].at<double>(9),0,0,1));
					std::cout<<"R="<<rot<<"\n";
					std::cout<<"T="<<for_ba_cams[i].rowRange(3,6)<<"\n";
					std::cout<<"K="<<KK<<"\n";
				}
				std::vector<Point3d> Xp;
				std::vector<Point2d> np;
				if(norm(res,NORM_INF)<20.5)
				{
					/*startpos=framepos;
					timestep*=1.2;
					timestep=min(timestep,2000.0);*/
					/*for(int i=0;i<for_ba_world[buff].size();i++)
					{
					projectPoints(for_ba_world[buff][i],RR,TT,KK,noArray(),np);
					circle(Pano,np[0],5,CV_RGB(255,0,0));
					}*/
					for(int i=0;i<triangles.size();i++)
					{
						auto tr=triangles[i];		
						Xp.clear();
						Xp.push_back(Point3d(for_ba_world[buff][tr.wp1]));
						Xp.push_back(Point3d(for_ba_world[buff][tr.wp2]));
						Xp.push_back(Point3d(for_ba_world[buff][tr.wp3]));
						//perspectiveTransform(Xp,Xp,GlobalH.inv());
						Rodrigues(for_ba_cams[0].rowRange(0,3),RR);
						TT=for_ba_cams[0].rowRange(3,6);
						projectPoints(Xp,RR,TT,KK,noArray(),np);
						//line(Pano,np[0],np[1],CV_RGB(255,0,0));
						//line(Pano,np[1],np[2],CV_RGB(255,0,0));
						//line(Pano,np[0],np[2],CV_RGB(255,0,0));
						double l=sqrt((np[0]-np[1]).ddot(np[0]-np[1]));
						auto v1=(tr.p1-tr.p2),v2=(tr.p3-tr.p2);
						auto nv1=(np[0]-np[1]),nv2=(np[2]-np[1]);
						double del=max(0.5/l,0.0001);
						for(double d1=0;d1<=1;d1+=del)
							for(double d2=0;d2<=1-d1;d2+=del)
							{
								Point2f p=tr.p2+d1*v1+d2*v2;
								Point2f newp=np[1]+ d1*nv1+d2*nv2;
								if(newp.inside(Rect_<float>(0,0,1000,1000)))
								{
									uchar* ptrr=Pano.ptr<uchar>(newp.y,newp.x);
									uchar* ptr2=images[tr.image].ptr<uchar>(p.y,p.x);
									if(*ptrr==0&&*(ptrr+1)==0&&*(ptrr+2)==0)
										for(int v=0;v<3;v++)
										{
											*ptrr=*ptr2;
											ptrr++;ptr2++;
										}
								}
							}
					}
					//buff=1-buff;
				}
				/*else
				{
				framepos=startpos;
				timestep/=1.5;
				}*/
				triangles.clear();
				Mat lc=for_ba_cams.back().clone();
				Mat LP;
				//composeProjectionMatrix(lc.rowRange(0,3),lc.rowRange(3,6),Mat::eye(3,3,CV_64F),LP);
				//LP.push_back(Mat(Matx14d(0,0,0,1)));
				//GlobalH*=LP;
				for_ba_cams.clear();
				Mat P=Mat::zeros(6,1,CV_64F);
				P.push_back(Kvec);
				for_ba_cams.push_back(P.clone());
				//	for_ba_cams.push_back(lc.clone());
				GlobalR=P.rowRange(0,3);
				Globalt=P.rowRange(3,6);
				newframe=1;
				buff=1-buff;
				for_ba_points[buff].push_back(std::vector<std::pair<Point2f,int> >());
				worldsize=0;
				for_ba_ind.clear();
				isfixed.clear();
				imshow("RES",Pano);
				//waitKey(100000000);
			}
#endif
		}
	}
	return 0;
}
#endif
