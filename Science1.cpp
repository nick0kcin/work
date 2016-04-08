// Science1.cpp: определяет точку входа для консольного приложения.
//

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
#include "FindRoads.h"
#include "genehough.h"

#pragma comment(linker, "/STACK:16777216")

using namespace cv;
using namespace cv::detail;
//using namespace std;
#define SUFRMATCH 1
#define HOMO 1
#define AFFINEEST 0
#define CORNFLOW 0
#define VIDEOEGDE 1
#define ROTAUTOCAB 0
#define RECTIFY_UNCAL 0
#define RECTIFY_CAL 1
#define DISPARITY 0
#define PHASECOR 0

#define At(M,i,j) (M->data+(i)*M->step[0]+(j)*M->step[1])
#define At4(M,i,j,k,m) (M->data+(i)*M->step[0]+(j)*M->step[1]+(k)*M->step[2]+(m)*M->step[3])

#define Att(M,t,i,j) ((t*)(M->data+(i)*M->step[0]+(j)*M->step[1]))





void rollingGuidanceFilter(InputArray src_, OutputArray dst_, int d,
                              double sigmaColor, double sigmaSpace,  int numOfIter, int borderType)
   {
        CV_Assert(!src_.empty());

        Mat guidance = src_.getMat();
        Mat src = src_.getMat();

        CV_Assert(src.size() == guidance.size());
        CV_Assert(src.depth() == guidance.depth() && (src.depth() == CV_8U || src.depth() == CV_32F) );

        if (sigmaColor <= 0)
            sigmaColor = 1;
        if (sigmaSpace <= 0)
            sigmaSpace = 1;

        dst_.create(src.size(), src.type());
        Mat dst = dst_.getMat();

        if (src.data == guidance.data)
            guidance = guidance.clone();
        if (dst.data == src.data)
            src = src.clone();

        int srcCnNum = src.channels();

        if (srcCnNum == 1 || srcCnNum == 3)
        {
            while(numOfIter--){
                ximgproc::jointBilateralFilter(guidance, src, guidance, d, sigmaColor, sigmaSpace, borderType);
            }
            guidance.copyTo(dst_);
        }
        else
        {
            CV_Error(Error::BadNumChannels, "Unsupported number of channels");
        }
    }

double FMError(Point2f p1,Point2f p2,Mat F)
{
	Matx31d xx(p2.x,p2.y,1);
	Matx31d xxx(p1.x,p1.y,1);
	Mat r= (Mat(xx.t())*F*Mat(xxx));
	return r.at<double>(0);
}
double Homoerror(Point2d p1, Point2d p2,Mat H)
{
	Matx31d xx(p2.x,p2.y,1);
	Matx31d xxx(p1.x,p1.y,1);
	Mat r=Mat(xx)-H*Mat(xxx);
	return r.dot(r);
}
double Homodiff(Point2d p1, Point2d p2,Mat H)
{
	Matx31d xx(p2.x,p2.y,1);
	Matx31d xxx(p1.x,p1.y,1);
	Mat r1=Mat(xx),r2=H*Mat(xxx);
	for(int i=0;i<3;i++)
	{
		r1.at<double>(i)/=r1.at<double>(2);
		r2.at<double>(i)/=r2.at<double>(2);
	}
   return (r1-r2).dot(r1-r2);
}

#if USELESS
Mat computePlanarMotionF1( std::vector<Point2f> x1,std::vector<Point2f> x2) //translate
{
	Mat A(x1.size(),3,CV_64F);
	for(int i=0;i<x1.size();i++)
	{
		A.at<double>(i,0)=x1[i].x*x2[i].y-x2[i].x*x1[i].y;
		A.at<double>(i,1)=x1[i].x-x2[i].x;
		A.at<double>(i,2)=x1[i].y-x2[i].y;
	}
	Mat ff=SVD(A.t()*A).vt.row(2);
	Mat F(3,3,CV_64F);
	F.setTo(0);
	F.at<double>(2,0)=-ff.at<double>(1);
	F.at<double>(2,1)=-ff.at<double>(2);
	F.at<double>(0,2)=ff.at<double>(1);
	F.at<double>(1,2)=ff.at<double>(2);
	F.at<double>(0,1)=ff.at<double>(0);
	F.at<double>(1,0)=-ff.at<double>(0);
	return F;
}
Mat computePlanarMotionF2( std::vector<Point2f> x1,std::vector<Point2f> x2) //translate with rotate
{
	Mat A(x1.size(),6,CV_64F);
	for(int i=0;i<x1.size();i++)
	{
		A.at<double>(i,0)=x1[i].x*x2[i].x+x1[i].y*x2[i].y;
		A.at<double>(i,1)=x1[i].y*x2[i].x-x2[i].y*x1[i].x;
	    A.at<double>(i,2)=x2[i].x;
		A.at<double>(i,3)=x2[i].y;
		A.at<double>(i,4)=x1[i].x;
		A.at<double>(i,5)=x1[i].y;
	}
	Mat ff=SVD(A.t()*A).vt.row(5);
	Mat F(3,3,CV_64F);
	F.setTo(0);
	F.at<double>(0,0)=ff.at<double>(0);
	F.at<double>(1,1)=ff.at<double>(0);
	F.at<double>(0,1)=ff.at<double>(1);
	F.at<double>(1,0)=-ff.at<double>(1);
	F.at<double>(0,2)=ff.at<double>(2);
	F.at<double>(1,2)=ff.at<double>(3);
	F.at<double>(2,0)=ff.at<double>(4);
	F.at<double>(2,1)=ff.at<double>(5);
	return F;
}
Mat computePlanarMotionF3( std::vector<Point2d> x1,std::vector<Point2d> x2) //translate with rotate
{
	Mat A(x1.size(),5,CV_64F);
	for(int i=0;i<x1.size();i++)
	{
		A.at<double>(i,0)=1;
	    A.at<double>(i,1)=x2[i].x;
		A.at<double>(i,2)=x2[i].y;
		A.at<double>(i,3)=x1[i].x;
		A.at<double>(i,4)=x1[i].y;
	}
	Mat ff=SVD(A.t()*A).vt.row(4);
	Mat F(3,3,CV_64F);
	F.setTo(0);
	F.at<double>(2,2)=ff.at<double>(0);
	F.at<double>(0,2)=ff.at<double>(1);
	F.at<double>(1,2)=ff.at<double>(2);
	F.at<double>(2,0)=ff.at<double>(3);
	F.at<double>(2,1)=ff.at<double>(4);
	return F;
}
Mat computeF(std::vector<Point2f> x1,std::vector<Point2f> x2)
{
	Mat A(x1.size(),9,CV_64F);
	for(int i=0;i<x1.size();i++)
	{
		Matx31d p1(x1[i].x,x1[i].y,1);
		Matx31d p2(x2[i].x,x2[i].y,1);
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
		{
			A.at<double>(i,j*3+k)=Mat(p1).at<double>(j)*Mat(p2).at<double>(k);
			}
	}
	Mat ff=SVD(A).u.row(8);
	Mat F(3,3,CV_64F);
	F.setTo(0);
	for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				F.at<double>(j,k)=ff.at<double>(j*3+k);
	Mat sv=SVD(F).w;
	Mat d=Mat::zeros(3,3,CV_64F);
	for(int i=0;i<2;i++)
		d.at<double>(i,i)=sv.at<double>(i);
	Mat FF=SVD(F).u*d*SVD(F).vt;
	return FF;
}

Mat EuclidEstimation(std::vector<Point2d> p1,std::vector<Point2d> p2)
{
	Mat A(p1.size()*2,4,CV_64F);
	Mat b(p1.size()*2,1,CV_64F);
	A.setTo(0);
	for(int i=0;i<A.rows/2;i++)
	{
		A.at<double>(2*i,0)=p1[i].x;
		A.at<double>(2*i,1)=p1[i].y;
		A.at<double>(2*i,2)=1;
		A.at<double>(2*i+1,0)=p1[i].y;
		A.at<double>(2*i+1,1)=-p1[i].x;
		A.at<double>(2*i+1,3)=1;
		b.at<double>(2*i)=p2[i].x;
		b.at<double>(2*i+1)=p2[i].y;
	}
	Mat C=A.t()*A;
	Mat d=A.t()*b;
	Mat R=C.inv()*d;
	Matx23d res(R.at<double>(0),R.at<double>(1),R.at<double>(2),-R.at<double>(1),R.at<double>(0),R.at<double>(3));
	return Mat(res);
}

Mat Ransac(std::vector<Point2f> p1,std::vector<Point2f> p2,double thresh,double percent,Mat (*estimator)(std::vector<Point2f>,std::vector<Point2f>),double (*error)(Point2f,Point2f,Mat), int minsamples)
{
	std::vector<Point2f> x1,x2;
	x1.resize(minsamples);
	x2.resize(minsamples);
	Mat res;
	std::vector<Point2f> inl1,inl2,resinl1,resinl2;
	int maxinl=0;
	if(p1.size()<minsamples)
		return Mat();
	for(int iter=0;iter<2000;iter++)
	{
		int inliers=0;
		inl1.clear();
		inl2.clear();
		for(int i=0;i<minsamples;i++)
		{
			int k=i+randu<int>()%(p1.size()-i);
			x1[i]=Point2d(p1[k]);
			x2[i]=Point2d(p2[k]);
			swap(p1[i],p1[k]);
			swap(p2[i],p2[k]);
		}
		Mat model=estimator(x1,x2);
		for(int i=0;i<p1.size();i++)
		{
			if(error(p1[i],p2[i],model)<thresh)
				 {
					 inliers++;
					 inl1.push_back(p1[i]);
					 inl2.push_back(p2[i]);
				 }
		}
		if(inliers>maxinl)
		{
			maxinl=inliers;
			resinl1=std::vector<Point2f>(inl1);
			resinl2=std::vector<Point2f>(inl2);
		}
		if(inliers>=percent*p1.size())
		{
			return estimator(inl1,inl2);
		}
	}
	return estimator(resinl1,resinl2);
}

#endif

Mat mergeImages(Mat im1,Mat im2)	//склейка изображений
{
	Mat res(im1.size(),CV_8UC1);
	unsigned char* data1=im1.data,
		*data2=im2.data,*data3=res.data;
	for(int i=0;i<im1.rows;i++)
		for(int j=0;j<im1.cols;j++)
		{
			if(*data1==0||*data2==0)
			{
				*data3=*data1+*data2;
			}
			else
			{
				*data3=*data1;
		     }
			data1++;
			data2++;
			data3++;
		}
			return res;
}

Point operator *(Mat m,Point p)
{
	if(m.rows==3)
	{
	Matx31d pp(p.x,p.y,1);
	Mat r=m*Mat(pp);
	return Point(r.at<double>(0)/r.at<double>(2),r.at<double>(1)/r.at<double>(2));
	}
	else
	{
		Matx21d pp(p.x,p.y);
		Mat r= m*Mat(pp);
		return Point(r.at<double>(0),r.at<double>(1));
	}
		}


void findImage(Mat map,Mat templ)
{
	Mat mapgx,mapgy,tempgx,tempgy;   
	Sobel(map,mapgx,CV_16S,1,0,5); //вычисление градиента шаблона и карты
	Sobel(map,mapgy,CV_16S,0,1,5);
	Sobel(templ,tempgx,CV_16S,1,0);
	Sobel(templ,tempgy,CV_16S,0,1);
	double s0=1.5,s1=2.0,ds=0.1,dth=5,dsp=2;
	//std::map<Vec4d,int> accum;
	int sz[4]={map.rows/dsp+1,map.cols/dsp+1,int((s1-s0)/ds)+1,int(360/dth)+1};
	Mat accum(4,sz,CV_16S);
	accum.setTo(0);
	Mat mp=map.clone();
	std::vector<Point2d> p1,p2;
	for(int i=0;i<map.rows;i+=3)
		for(int j=0;j<map.cols;j+=3)
			if(*At((&map),i,j)==255)
			{
					double v1x=*At((&mapgx),i,j);
					double v1y=*At((&mapgy),i,j);
					double ang1=atan2(v1y,v1x);
				for(int x=0;x<templ.rows;x++)
					for(int y=0;y<templ.cols;y++)
						if(*At((&templ),x,y)==255)   //перебираем все возможные варианты соответствия (i,j)-> (x,y)
						{
							double v2x=*At((&tempgx),x,y);
							double v2y=*At((&tempgy),x,y);
							double ang2=atan2(v2y,v2x);
							double ang[2]={ang1-ang2,ang1-ang2+M_PI};//  считаем,что угол поворота это угол между градиентами
							//double ang[72];
							//for(int k=0;k<72;k++)
							//	ang[k]=k*M_PI/36;
							for(int k=0;k<2;k++)
							//for(int k=0;k<72;k++)
							{
								while (ang[k]<0)
									ang[k]+=2*M_PI;
								while (ang[k]>=2*M_PI)
									ang[k]-=2*M_PI;
							for(double s=s0;s<=s1;s+=0.1)  //для разных масштабов вычисляем вектора переноса
							{
								double a=s*cos(ang[k]),b=s*sin(ang[k]);
								double xx=i-a*x-b*y;
								double yy=j+b*x-a*y;
								if(xx>=0&&yy>=0&&xx<map.rows&&yy<map.cols)
								{
									int thbin=int(ang[k]*180/(M_PI*dth));  //получили 4 параметра,однозначно определяющие евклидовое преобразование
									int sbin=int((s-s0)/ds);
									int xbin=(xx/dsp);
									int ybin=(yy/dsp);
									short* p=(short*)At4((&accum),(int)(xx/dsp),(int)(yy/dsp),int((s-s0)/ds),k);//голосуем за него
								(*p)++;
								}
							}
							}
						}
			}
			double minval,maxval;
 int id_min[4] = { 0, 0, 0, 0};
             int id_max[4] = { 0, 0, 0, 0};
             minMaxIdx(accum, &minval, &maxval, id_min, id_max);  //находим то,за которое  больше голосовали
			 double ss=s0+ds*id_max[2];
			 double x0=id_max[0]*dsp;
			 double y0=id_max[1]*dsp;
			 double a=ss*cos(id_max[3]*dth*M_PI/180),b=-ss*sin(id_max[3]*dth*M_PI/180);
			 Matx23d trans(a,b,x0,-b,a,y0);
			 Mat sh;
			 warpAffine(templ,sh,Mat(trans),Size(map.cols,map.rows));
			 cv::bitwise_or(sh/2,map,sh);
			// Mat tr=estimateRigidTransform(p2,p1,0);
			 //std::cout<<tr<<"\n";
		//	 warpAffine(templ,sh,tr,Size(map.cols,map.rows));
			 imshow("finding",sh);
}


void findGoodMatches(BFMatcher& matcher,UMat& descriptors1,UMat& descriptors2,std::vector<cv::KeyPoint>& points1,
					 std::vector<cv::KeyPoint>& points2,std::vector<Point2f>& p1,std::vector<Point2f>& p2,std::vector<cv::DMatch> goodmatches)
{
	std::vector<cv::DMatch> matches[2];
	 matcher.match(descriptors1,descriptors2,matches[0]);//находим соответствия из 1 во 2 изображение и наоборот
	 matcher.match(descriptors2,descriptors1,matches[1]);
			 double mn[2]={1e06,1e06}; //считаем хорошими те которые есть в обоих списках и расстояние между котрыми <3* мин расстояние
				 for(int j=0;j<2;j++)
					 for(int i=0;i<matches[j].size();i++)
				 mn[j]=std::min(double(matches[j][i].distance),mn[j]);
				 p1.clear();
				 p2.clear();
				 goodmatches.clear();
			 for(int i=0;i<matches[0].size();i++)
			 {
				 if(matches[1][matches[0][i].trainIdx].trainIdx==i)
				if(matches[0][i].distance < 3 *mn[0])
					if(matches[1][matches[0][i].trainIdx].distance < 3 *mn[1])
					{
					    goodmatches.push_back(DMatch(matches[0][i].queryIdx,matches[0][i].trainIdx,0,0.01));
						p1.push_back(points1[matches[0][i].queryIdx].pt);
						p2.push_back(points2[matches[0][i].trainIdx].pt);
					}
			 }
}


Mat GetGreenMap(Mat* im,int r) //для каждого пикселя считает меру того,насколько зеленого цвета в окрестности больше
{
	Mat res(im->rows,im->cols,CV_32F);
	Mat g=getGaussianKernel(2*r+1,-1);
	std::cout<<g<<"\n";
	for(int i=0;i<im->rows;i++)
		for(int j=0;j<im->cols;j++)
		{
		float v=0;
			for(int x=-r;x<=r;x++)
				for(int y=-r;y<=r;y++)
					if(Point(i+x,j+y).inside(Rect_<int>(0,0,im->rows,im->cols)))
					v+=(*(At(im,i+x,j+y)+1)-max(*At(im,i+x,j+y),*(At(im,x+i,y+j)+2)))/**g.at<double>(x+r)*g.at<double>(y+r)*/;
			float* pt=(float*)At((&res),i,j);
			v=max(v,0.0f);
			*pt=v;
			//v/=r*r;
		}
		normalize(res,res,0,1,CV_MINMAX,CV_32F);
		return res;
}
int main()
{
	VideoCapture cap("v.mov");
	//namedWindow("wind");
	double deltaT=1500;
	double deltaTincrease=1.2;
	double deltaTdecrease=0.7;
	double mindeltaT=100;
	int stitchedframes=0;
	Mat frame[3];
	Mat inp[3];
	Stitcher stitcher=Stitcher::createDefault();
	std::vector<cv::KeyPoint> points[3];
	std::vector<cv::KeyPoint> mappoints,panopoints;
	std::vector<cv::DMatch> matches[3];
	std::vector<DMatch> goodmatches;
	std::vector<Point2f> p1,p2,pm;
	std::vector<Point2f> fp1,fp2;
	std::vector<Mat> Homos;
	 std::vector<Mat> MyMap;
	 Mat Move(650,1200,CV_8SC3);
	 Point pos(400,600);
	 Mat ori=Mat::eye(3,3,CV_64F);
	 Mat ori2=Mat::eye(2,3,CV_64F);
	 	 ori2.at<double>(0,2)=300;
	 ori2.at<double>(1,2)=300;
	 Mat Hom=Mat::eye(3,3,CV_64F);
	 Hom.at<double>(0,0)=0.5;
	 Hom.at<double>(1,1)=0.5;
	  Hom.at<double>(0,2)=100;
	 Hom.at<double>(1,2)=1500;
	Mat px(1, 1, CV_64F), pxstr(1, 1, CV_64F);
	Mat R,t;
	Mat mask;
	Mat X;
	UMat descriptors[3];
	UMat mapdescr,panodescr;
	Mat F,H;
	BFMatcher matcher;
	Ptr<Feature2D> detector = xfeatures2d::SURF::create();
	Ptr<Feature2D> descriptor = xfeatures2d::SURF::create();
	int n=0;
	Mat immatch,immatch2;
	cap.set(CAP_PROP_POS_MSEC,83000);//131000 142000 44000 63000
	int inter=0;
	Mat map=imread("mapnew2.bmp");

	Mat roads=FindRoads(map);
	pyrDown(map,map);
	//pyrDown(map,map);
	rollingGuidanceFilter(map,map,3,6.3,4.9,5,4);
	imshow("mapcolor",map);
	imshow("roads",roads);
	Mat mapc=map.clone();
	cvtColor(map,map,CV_RGB2GRAY);
	//imshow("roads1",out2);
	detector->detect(map,mappoints);
	detector->compute(map,mappoints,mapdescr);
	//auto genhough=createGeneralizedHoughGuil();
	GeneralizedHoughGuilImpl* genhough=new GeneralizedHoughGuilImpl();
	genhough->setCannyHighThresh(140);
	genhough->setCannyLowThresh(60);
	genhough->setAngleEpsilon(1);
	genhough->setAngleStep(1);
	genhough->setAngleThresh(100);
	genhough->setLevels(360);
	genhough->setMaxAngle(360);
	genhough->setMaxScale(30);
	genhough->setMinAngle(0);
	genhough->setMinDist(0);
	genhough->setMinScale(0.1);
	genhough->setPosThresh(1);
	genhough->setScaleStep(0.01);
	genhough->setScaleThresh(3);
	genhough->setMaxBufferSize(10000);
#if TESTGENHOUGH
	genhough->setCannyHighThresh(140);
	genhough->setCannyLowThresh(60);
	genhough->setAngleEpsilon(1);
	genhough->setAngleStep(1);
	genhough->setAngleThresh(1000);
	genhough->setLevels(360);
	genhough->setMaxAngle(360);
	genhough->setMaxScale(3);
	genhough->setMinAngle(0);
	genhough->setMinDist(0);
	genhough->setMinScale(0.5);
	genhough->setPosThresh(100);
	genhough->setScaleStep(0.01);
	genhough->setScaleThresh(100);
	genhough->setMaxBufferSize(10000);
	Mat test1=imread("rect1.bmp");
	Mat test2=imread("rect2.bmp");
	cvtColor(test1,test1,CV_RGB2GRAY);
	cvtColor(test2,test2,CV_RGB2GRAY);
	genhough->setTemplate(test1,Point(-1,-1));
	 Mat posi;
		 Mat votes;
		 Mat ddx,ddy;
		pyrDown(map,map);
	Sobel(test2,ddx,CV_32F,1,0);
	Sobel(test2,ddy,CV_32F,0,1);
	Canny(test2,test2,60,140);
	Canny(test1,test1,60,140);
	imshow("test2",test2);
	imshow("test1",test1);
		 genhough->detect(test2,ddx,ddy,posi,votes);
		 std::cout<<posi<<"\n";
#endif
	/*std::vector<Vec4f> linesmap;
	HoughLinesP(map,linesmap,4000,M_PI/80,3000,10.0,3.0);
	for(int i=0;i<linesmap.size();i++)
		line(imlines,Point2f(linesmap[i][0],linesmap[i][1]),Point2f(linesmap[i][2],linesmap[i][3]),CV_RGB(255,0,0));
	imshow("mapline",imlines);*/
#if PHASECOR
				 Mat hann;
			 createHanningWindow(hann, Size(map.cols, map.rows), CV_64F);
			 Mat logmap,mapdouble;
			 logPolar(map,logmap,Point2f(map.cols/2,map.rows/2),200,InterpolationFlags::INTER_LINEAR);
			 imshow("logmap",logmap);
			 logmap.convertTo(logmap,CV_64F);
			 map.convertTo(mapdouble,CV_64F);
#endif

#if VIDEOEGDE
Mat green=GetGreenMap(&mapc,2);
	for(int i=0;i<green.rows;i++)
		for(int j=0;j<green.cols;j++)
		{
			float* pt=green.ptr<float>(i,j);
			*pt=((*pt<0.27)?(1.0):(0.0));
		}
			imshow("green",green);
		green.convertTo(green,CV_8U);
	Mat mapegdes;   //получение контуров с помощью канни
	Mat dx,dy;
		//pyrDown(map,map);
		//pyrDown(map,map);
	Sobel(map,dx,CV_32F,1,0);
	Sobel(map,dy,CV_32F,0,1);
	std::vector<std::vector<Point>> mcontours;
	std::vector<std::vector<Point>> pcontours;
	cv::Canny(map,mapegdes,70,210);
	mapegdes=mapegdes.mul(green); // рассматриваю только незеленые контуры
	//mapegdes=mapegdes.mul(out);
	findContours(mapegdes,mcontours,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);
		 Mat mconim(map.size(),CV_8SC3);
		 for(int i=0;i<mcontours.size();i++)
			 if(mcontours[i].size()>20/*&&(mcontours[i][0]-mcontours[i].back()).dot(mcontours[i][0]-mcontours[i].back())>20*/)
				 drawContours(mconim,mcontours,i,Scalar(rand()%256,rand()%256,rand()%256));
		 mapegdes=mapegdes(Range(mapegdes.rows/2,mapegdes.rows-1),Range(0,mapegdes.cols/2)); // контуров на карте все равно слишком много,беру пока только часть карты
		 dx=dx(Range(dx.rows/2,dx.rows-1),Range(0,dx.cols/2));
		 dy=dy(Range(dy.rows/2,dy.rows-1),Range(0,dy.cols/2));

	imshow("map",mapegdes);
	imshow("cmap",mconim);
#endif
	while(1)
	{
		char c= waitKey(500);
		bool success=0;//удачно ли приклеили кадр к панораме
		if(c==' ')
		{
 			std::cout<<inter++<<"\n";
		cap>>frame[n];
		//cap.set(CAP_PROP_POS_MSEC,cap.get(CAP_PROP_POS_MSEC)+1000);

			cv::resize(frame[n],frame[n],cv::Size(0,0),0.5,0.5);
		cv::cvtColor(frame[n],inp[n],CV_RGB2GRAY);
		GaussianBlur(inp[n],inp[n],Size(0,0),2);
	
#if SUFRMATCH
		detector->detect(inp[n],points[n]);
		 detector->compute(inp[n],points[n],descriptors[n]);
#endif
		Mat immap;
#if VIDEOEGDE
		Mat mini;
		cv::GaussianBlur(inp[n],mini,Size(0,0),6.8);
		pyrDown(mini,mini);
		pyrDown(mini,mini);
		pyrDown(mini,mini);

		 Mat egdes;
		 cv::Canny(mini,egdes,55,135);  
		 genhough->setTemplate(mini,Point(-1,-1));
		 Mat pos;
		 Mat votes;
		 genhough->detect(mapegdes,dx,dy,pos,votes);
		 std::cout<<pos<<"\n"<<votes<<"\n";
		// findImage(roads,egdes);  //пытаемся найти соотвествие между дорогами и контурами на видео
		 //findContours(egdes,pcontours,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);
		 //Mat pconim(inp[n].size(),CV_8SC3);
		 //for(int i=0;i<pcontours.size();i++)
			// if(pcontours[i].size()>20)
			//	 drawContours(pconim,pcontours,i,Scalar(255,0,0));
		 imshow("eg",egdes);
		 //imshow("winde",inp[n]);
		 //imshow("pcont",pconim);
#endif
#if PHASECOR  // https://en.wikipedia.org/wiki/Phase_correlation
		 //http://www.jprr.org/index.php/jprr/article/view/355/148
			 Mat im;
			 resize(inp[n],im,Size(map.cols,map.rows));
			// pyrDown(im1,im1);
			// pyrDown(im2,im2);
			 //GaussianBlur(im1,im1,Size(),3);
			 //GaussianBlur(im2,im2,Size(),3);
			 Mat log;
			 logPolar(im,log,Point(im.cols/2,im.rows/2),200,InterpolationFlags::INTER_LINEAR);
			 imshow("log1",log);
			 log.convertTo(log,CV_64F);
			 im.convertTo(im,CV_64F);
			 double resp=0,respon=0;
			 Point2d rotate=phaseCorrelate(log,logmap,hann,&respon);
			 Point2d translate= phaseCorrelate(im,mapdouble,hann,&resp);
			 std::cout<<translate<<' '<<resp<<"\n";
			 std::cout<<rotate<<respon<<"\n";
#endif
		 if(points[0].size()>0&&points[1].size()>0)
		 {
#if (SUFRMATCH)
			 matcher.match(descriptors[(n+1)%2],descriptors[n],matches[0]);//находим соответствия из 1 во 2 изображение и наоборот
			 matcher.match(descriptors[n],descriptors[(n+1)%2],matches[1]);
			 std::vector<Point2d> flow;
			 flow.clear();
			 double mn[2]={1e06,1e06}; //считаем хорошими те которые есть в обоих списках и расстояние между котрыми <3* мин расстояние
				 for(int j=0;j<2;j++)
					 for(int i=0;i<matches[j].size();i++)
				 mn[j]=std::min(double(matches[j][i].distance),mn[j]);
			 goodmatches.clear();
			 for(int i=0;i<matches[0].size();i++)
			 {
				 if(matches[1][matches[0][i].trainIdx].trainIdx==i)
				if(matches[0][i].distance < 3 *mn[0])
					if(matches[1][matches[0][i].trainIdx].distance < 3 *mn[1])
						//if(matches[0][i].queryIdx==matches[1][i].queryIdx)
					{
						if(goodmatches.empty()|| points[1-n][goodmatches.back().queryIdx].pt!=points[1-n][matches[0][i].queryIdx].pt||
							points[n][goodmatches.back().trainIdx].pt!=points[n][matches[0][i].trainIdx].pt)
						goodmatches.push_back(DMatch(matches[0][i].queryIdx,matches[0][i].trainIdx,0,0.01));
							
					}
			 }
			 p1.resize(goodmatches.size());
			 p2.resize(goodmatches.size());
			 fp1.resize(goodmatches.size());
			 fp2.resize(goodmatches.size());
			 for(int i=0;i<goodmatches.size();i++) //p1[i]<->p2[i]
			 {
					 p1[i]=points[1-n][ goodmatches[i].queryIdx].pt/*-Point2f(inp[1-n].cols/2,inp[1-n].rows/2)*/;
					 p2[i]=points[n][ goodmatches[i].trainIdx].pt/*-Point2f(inp[n].cols/2,inp[n].rows/2)*/;
				 flow.push_back(p2[i]-p1[i]);

			 }
#if DISPARITY //вычисление карты смещения
			 Mat R1,R2,P1,P2;
			 //вычисление ректификацирующего преобразования
	#if RECTIFY_UNCAL //instr - полученна калибровкой, сама калибровка в Science2.cpp
			 Mat F=findFundamentalMat(p1,p2);
			 Mat H1,H2;
			 Mat inst=Mat(Matx33d(2337.983953926237, -0.09271596967581647, 562.8852861944566,0, 2327.653717544988, -63.22961930103139,0, 0, 1));
			 stereoRectifyUncalibrated(p1,p2,F,Size(inp[n].cols,inp[n].rows),H1,H2);
			  R1=inst.inv()*H1*inst;
			  R2=inst.inv()*H2*inst;
			  P1=inst;
			  P2=inst;
			 std::cout<<H1<<"\n"<<H2<<"\n"<<R1<<"\n"<<R2<<"\n";
#endif
#if RECTIFY_CAL
			 Mat inst=Mat(Matx33d(2386.265926154927, 0, 567.4823032255524,0, 2373.991088807663, -80.27460712440958,0, 0, 1));
			 perspectiveTransform(p1,fp1,inst); 
			 perspectiveTransform(p2,fp2,inst);
			 Mat E=findEssentialMat(fp1,fp2);
			 recoverPose(E,fp1,fp2,R,t);
			// R=Mat::eye(3,3,CV_64F);
			// t=Mat(Matx31d(1,0,0));
			 Mat Q;
			 std::cout<<R<<"\n"<<t<<"\n";
			 
			 stereoRectify(inst,noArray(),inst,noArray(),Size(inp[n].cols,inp[n].rows),R,t,R1,R2,P1,P2,Q,1024,0);
			 std::cout<<R1<<"\n"<<R2<<"\n"<<P1<<"\n"<<P2<<"\n"<<Q<<"\n";
#endif
			 Mat map11,map12,map21,map22; //ректификация
			 initUndistortRectifyMap(inst,noArray(),R1,P1,Size(inp[n].cols,inp[n].rows),CV_32F,map11,map12);
			 initUndistortRectifyMap(inst,noArray(),R2,P2,Size(inp[n].cols,inp[n].rows),CV_32F,map21,map22);
			 Mat temp1,temp2;
			 remap(inp[n],temp1,map11,map12,InterpolationFlags::INTER_CUBIC);
			 remap(inp[1-n],temp2,map21,map22,InterpolationFlags::INTER_CUBIC);
			 imshow("im1",temp1);
			 imshow("im2",temp2);
			 //построение карты глубины
			 auto bm=StereoSGBM::create(16,256,11);
			 Mat disp,disp8;
			 bm->compute(temp2,temp1,disp);
		 normalize(disp,disp8,0,255,CV_MINMAX,CV_8U);
			// disp.convertTo(disp8,CV_8U);
			 imshow("disp",disp8);
#endif
#endif
#if CORNFLOW
			 Mat flows; //попыткм получить карту глубины для случая перемещения используя оптический поток
			 calcOpticalFlowFarneback(inp[n],inp[1-n],flows,0.5,3,20,2,7,1.6,0);
			 Mat dist(inp[n].rows,inp[n].cols,CV_8U);
			// std::cout<<flows<<"\n";
			 for(int i=0;i<dist.rows;i++)
				 for(int j=0;j<dist.cols;j++)
					 if(cvIsInf(*AT<double>(flows,i,j)))
						 dist.at<char>(i,j)=0.0;
					 else
						 dist.at<char>(i,j)=sqrt(*AT<double>(flows,i,j)**AT<double>(flows,i,j)+*(AT<double>(flows,i,j)+1)**(AT<double>(flows,i,j)+1));
			 medianBlur(dist,dist,5);
			 			 for(int i=0;i<dist.rows;i++)
				 for(int j=0;j<dist.cols;j++)
					 if(dist.at<char>(i,j)>0)
						 dist.at<char>(i,j)=255;
					 else
						 dist.at<char>(i,j)=0;
			 Mat mas;
			 bitwise_not(dist,dist);
			 bitwise_and(inp[n],dist,mas);
			 imshow("dist",dist);
			 imshow("mas",mas);
#endif
#if HOMO //склейка карты из видео
			 /*std::vector<Mat> stim;
			 stim.push_back(frame[n]);
			 stim.push_back(frame[1-n]);
			 Mat pan0;
			 Stitcher::Status status=stitcher.stitch(stim,pan0);
			 if(status==Stitcher::Status::OK)
				 imshow("stitch",pan0);*/
			// std::cin.get();
			 if(p1.size()>4)
			 {
			 Mat H2=findHomography(p2,p1,mask,RANSAC,2);
			 for(int i=0;i<3;i++)
				 for(int j=0;j<3;j++)
					 H2.at<double>(i,j)=int(H2.at<double>(i,j)*1e04)*1e-04; //округление
			 double det=determinant(H2);
			 //Matx33d sca(1/sqrt(det),0,0,0,1/sqrt(det),0,0,0,1);
			//H2*=Mat();
				 std::cout<<H2<<"\n";
				 	 std::cout<<Hom<<"\n";
					 double herr=0;
					 for(int i=0;i<p2.size();i++)
					 {
						 herr+=Homodiff(p2[i],p1[i],H2);
					 }
					 herr=sqrt(herr)/p2.size(); //вычислили средную ошибку в пикслелях при применении гомографии
					 std::cout<<herr<<"\n";
					 if(herr<0.7)
					 {
						  Hom=Hom*H2;
						  std::cout<<determinant(H2)<<"\n";
	/*Point p11=Hom*Point(0,0);
	Point p12=Hom*Point(im2.cols,0);
	Point p21=Hom*Point(0,im2.rows);
	Point p22=Hom*Point(im2.cols,im2.rows);
	Point hl(min(0,min(min(p11.x,p12.x),min(p21.x,p22.x))),
		min(0,min(min(p11.y,p12.y),min(p21.y,p22.y))));
	Point downr(max(MyMap.cols,max(max(p11.x,p12.x),max(p21.x,p22.x))),
		max(MyMap.rows,max(max(p11.y,p12.y),max(p21.y,p22.y))));*/
	Mat piece1;
	Mat im1=inp[n].clone();
	warpPerspective(im1,piece1,Hom,Size(1000,2100),InterpolationFlags::WARP_FILL_OUTLIERS,BorderTypes::BORDER_CONSTANT);//получаем кусок,который нужно приклеить
	MyMap[MyMap.size()-1]=mergeImages(MyMap.back(),piece1);
	success=1; 
	imshow("down",piece1);
	imshow("res",MyMap.back());
					 }
					 else
					 {
						 //frame[n]=frame[1-n];// если матрица гомографии получилась плохая,пропускаем кадр
					 }
			 }
			 else
			 {
				// frame[n]=frame[1-n];
			 }
#endif
#if AFFINEEST
			 if(p1.size()>3) //вычисление преобразования между кадрами как афинного (получилось немного хуже чем гомография)
			 {
			 Mat trf=estimateRigidTransform(p1,p2,0);
			 Mat eim;
			 std::cout<<trf<<"\n";
			 Mat e1=ori2(Range(0,2),Range(0,2))*trf(Range(0,2),Range(0,2));
			 Mat e2=ori2.col(2)+trf.col(2);
			 for(int i=0;i<2;i++)
			 {
				 for(int j=0;j<2;j++)
					 ori2.at<double>(i,j)=e1.at<double>(i,j);
				 ori2.at<double>(i,2)=e2.at<double>(i);
			 }
			 Mat img=frame[(n+2)%3].clone();
			// resize(img,img,Size(0,0),0.2,0.2);
			 warpAffine(img,eim,ori2,Size(1200,700));
			 imshow("eu",eim);
			 }
#endif

#if ROTAUTOCAB
			 Homos.push_back(findHomography(p2,p1,RANSAC,0.5)); //калибровка камеры при условии,что она неподвижна
			 Mat intern;                                       //думал,что получится применить там,где коптер разворачивается,
			 //                                                 но результаты были не очень
			 if(Homos.size()>=3)
			 {
			 cv::detail::calibrateRotatingCamera(Homos,intern);
			 std::cout<<(intern/intern.at<double>(2,2))<<"\n";
			 }
#endif


				//	drawMatches(frame[n],points[n],frame[(n+2)%3],points[(n+2)%3],goodmatches,immatch);
					 //drawMatches(inp[(n+2)%3],points[(n+2)%3],inp[(n+1)%3],points[(n+1)%3],matches[2],immatch2);
			// imshow("wind",immatch);
			// imshow("wind2",immatch2);
		 }
		 else
		 {
#if HOMO
			 //MyMap=inp[n];
			 Mat oup=inp[n];
			// pyrDown(inp[n],oup);
			 MyMap.push_back(Mat());
			 warpPerspective(oup,MyMap.back(),Hom,Size(1000,2100));
			 imshow("piece0",MyMap.back());
#endif
			  cap.set(CAP_PROP_POS_MSEC,cap.get(CAP_PROP_POS_MSEC)+deltaT);
		 n=(n+1)%2;
			 }
			 if(success)// если хорошо склеилось,увеличиваю временнной промежуток между след кадрами
			 {
				 deltaT*=deltaTincrease;
				 cap.set(CAP_PROP_POS_MSEC,cap.get(CAP_PROP_POS_MSEC)+deltaT);
				 stitchedframes++;
		 n=(n+1)%2;
			 }
			 else
				 if(p1.size()&&p2.size())
			 {  //беру новый кадр вместо старого пораньше
				 cap.set(CAP_PROP_POS_MSEC,cap.get(CAP_PROP_POS_MSEC)-deltaT+deltaT*deltaTdecrease);
				 deltaT*=deltaTdecrease;
				 if(deltaT<mindeltaT||stitchedframes>50)
				 {
					 Mat pano; //пытаюсь сопоставить панораму и карту с помощью точечных соответствий
					 resize(MyMap.back(),pano,Size(map.cols,map.rows));
					 detector->detect(pano,panopoints);
					 detector->compute(pano,panopoints,panodescr);
					 std::vector<Point2f> mp,pp;
					 std::vector<DMatch> goodmatch;
					 findGoodMatches(matcher,mapdescr,panodescr,mappoints,panopoints,mp,pp,goodmatch);
					 Mat immm;
					 drawMatches(map,mappoints,pano,panopoints,goodmatches,immm);

					 imshow("mtcgf",immm);

					 waitKey(10000000);

					 MyMap.push_back(Mat());
					 Hom.at<double>(0,0)=0.5;
					 Hom.at<double>(1,1)=0.5;
					 Hom.at<double>(2,2)=1;
					 Hom.at<double>(0,1)=0;
					 Hom.at<double>(1,0)=0;
					 Hom.at<double>(2,0)=0;
					 Hom.at<double>(2,1)=0;
					 warpPerspective(inp[n],MyMap.back(),Hom,Size(4000,4100));
					 n=1-n;
				 }
			 }
			 std::cout<<"time"<<cap.get(CAP_PROP_POS_MSEC)<<"\n";
		}
	} 
	return 0;
}

