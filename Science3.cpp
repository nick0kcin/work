#include "stdafx.h"
#if _FILE==3
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
#include "ContourMatcher.h"

using namespace cv;




int panoLowThr=10,panoHighThr=60;
int mapLowThr=5,mapHighThr=50;
std::vector<std::vector<Point>> mcontours;
std::vector<std::vector<Point>> pcontours;
Mat moduloGradient(Mat im)
{
	Mat dx,dy;
	Sobel(im,dx,CV_16U,1,0);
	Sobel(im,dy,CV_16U,0,1);
    dx.convertTo(dx,CV_32F);
	dy.convertTo(dy,CV_32F);
	dx=dx.mul(dx);
	dy=dy.mul(dy);
	Mat gr;
	cv::sqrt((dx+dy),gr);
	normalize(gr,gr,0,1,CV_MINMAX,CV_32F);
	return gr;
}



void mergeContours(std::vector<std::vector<Point > >& cont,std::vector<double>& lengths,double r2)
{
	for(int i=0;i<cont.size();i++)
	{
		Point p1=cont[i].front();
		Point p2=cont[i].back();
		double dist=(p1-p2).ddot(p1-p2);
		if(dist>9)
		{
			int nearest=-1;
			double maxlen=10000;
			int ori=0;
		for(int j=0;j<cont.size();j++)
			if(i!=j/*&&lengths[j]<lengths[i]*/)
		{
			//double d=0.1+1.3*min(lengths[j]/lengths[i],lengths[i]/lengths[j]);
			Point p=cont[j].front();
			double dis=(p-p2).ddot(p-p2);
			if(dis<r2)
				if(maxlen>dis)
				{
					maxlen=dis;
					nearest=j;
					ori=0;
				}

			 dis=(p-p1).ddot(p-p1);
			if(dis<r2)
				if(maxlen>dis)
				{
					maxlen=dis;
					nearest=j;
					ori=3;
				}

				 p=cont[j].back();
				 dis=(p-p2).ddot(p-p2);
			if(dis<r2)
				if(maxlen>dis)
				{
					maxlen=dis;
					nearest=j;
					ori=1;
				}

				 dis=(p-p1).ddot(p-p1);
			if(dis<r2)
				if(maxlen>dis)
				{
					maxlen=dis;
					nearest=j;
					ori=2;
				}


		}
		if(nearest!=-1)
		{
			int ind=-1;
			if(ori==2)
			{
				ind=min(nearest,i);
				swap(nearest,i);
				ori=0;
			}
			if(ori==3)
			{
				std::reverse(cont[i].begin(),cont[i].end());
				ori=0;
			}
			if(ori==1)
				for(int j=int(cont[nearest].size())-1;j>=0;j--)
				cont[i].push_back(cont[nearest][j]);
			else
				if(ori==0)
			for(int j=0;j<cont[nearest].size();j++)
				cont[i].push_back(cont[nearest][j]);


			cont.erase(cont.begin()+nearest);
			lengths[i]+=lengths[nearest];
			lengths.erase(lengths.begin()+nearest);
			if(nearest<i)
				i--;
			if(ind!=-1)
				i=ind-1;
		}
		}
	}
}

void correctContours(std::vector<std::vector<Point > >& mcontours)
{
		for(int i=0;i<mcontours.size();i++)
	{
		std::vector<int> cut;
		cut.push_back(0);
		for(int j=1;j<int(mcontours[i].size())-1;j++)
		{
			auto p1=mcontours[i][j]-mcontours[i][j-1];
			auto p2=mcontours[i][j+1]-mcontours[i][j];
			double coss=p1.ddot(p2)/sqrt(p1.ddot(p1)*p2.ddot(p2));
			if(coss<-0.7)
				cut.push_back(j);
		}
		if(cut.size()>=2)
		{
			int m1=0,m2=0;
			for(int j=1;j<cut.size();j++)
			{
				if(cut[j]-cut[j-1]>m2-m1)
				{
					m1=cut[j-1];
					m2=cut[j];
				}
			}
		mcontours[i].erase(mcontours[i].begin()+m2,mcontours[i].end());
		mcontours[i].erase(mcontours[i].begin(),mcontours[i].begin()+m1);
		}
	}
}

void filterContours(std::vector<std::vector<Point > >& mcontours,int val)
{
			 for(int i=0;i<mcontours.size();i++)
			 if(mcontours[i].size()<=val)
			 {
				 mcontours.erase(mcontours.begin()+i);
				 i--;
			 }
}
void filterCorners(std::vector<std::vector<Point > >& mcontours,int val,std::vector<std::vector<Point > >& corn)
{
			 for(int i=0;i<corn.size();i++)
			 if(corn[i].size()<=val)
			 {
				 mcontours.erase(mcontours.begin()+i);
				 corn.erase(corn.begin()+i);
				 i--;
			 }
}
int main()
{
	/*Mat test1=imread("matchtest1.bmp");
	Mat test2=imread("matchtest2.bmp");
	cvtColor(test1,test1,CV_RGB2GRAY);
	cvtColor(test2,test2,CV_RGB2GRAY);
	Mat egd1,egd2;
	Canny(test1,egd1,10,50);
	Canny(test2,egd2,10,50);
	imshow("test1",egd1);
	imshow("test2",egd2);
	std::vector<std::vector<Point > > cont1,cont2;
	findContours(egd1,cont1,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	findContours(egd2,cont2,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	Mat im1(test1.size(),CV_8UC3),im2(test2.size(),CV_8UC3);
	correctContours(cont1);
	correctContours(cont2);
	for(int i=0;i<cont1.size();i++)
	 drawContours(im1,cont1,i,Scalar(rand()%256,rand()%256,rand()%256));
	for(int i=0;i<cont2.size();i++)
	  drawContours(im2,cont2,i,Scalar(rand()%256,rand()%256,rand()%256));

	  CornersExtractor text(9,64,175,1000);
			 CornersExtractor text2(9,64,175,1000);

			 std::vector<std::vector<Point> > tcorn;
			  std::vector<std::vector<Point> > tcorn2;
			 text.extract(cont1,tcorn);
			 text2.extract(cont2,tcorn2);

			 std::vector<std::vector<char> > tres(cont1.size(),std::vector<char>(cont2.size(),0));
	ContourMatcherBouagar tmatcher(cont1,tcorn,150,4,1);
	tmatcher.precomputeQuery(cont2,tcorn2);
			 tmatcher.match(tres);


std::vector<DMatch> tmatches;
int ti=1,tj=4;
			 std::vector<KeyPoint> tpnt[2];
			 for(int i=0;i<tcorn[ti].size();i++)
				 tpnt[0].push_back(KeyPoint(tcorn[ti][i],1));
			 for(int i=0;i<tcorn2[tj].size();i++)
				 tpnt[1].push_back(KeyPoint(tcorn2[tj][i],1));
			 for(int i=0;i<tmatcher.matches_[ti][tj].size();i++)
				 tmatches.push_back(DMatch(tmatcher.matches_[ti][tj][i].first,tmatcher.matches_[ti][tj][i].second,0));



	 Mat tmim;
	drawMatches(im1,tpnt[0],im2,tpnt[1],tmatches,tmim);


	  imshow("cont1",im1);
	  imshow("cont2",im2);
	  imshow("mt",tmim);
	waitKey(100000000);*/

	Mat map=imread("mapnew2.bmp");
	int numberOfPano=5;
	std::string st="pano"+std::string(1,'0'+numberOfPano)+".bmp";
	Mat pano=imread(st);

	Mat mapgray,panogray;
	cvtColor(map,mapgray,CV_RGB2GRAY);
	cvtColor(pano,panogray,CV_RGB2GRAY);

	GaussianBlur(mapgray,mapgray,Size(0,0),2);
	//medianBlur(mapgray,mapgray,3);

	//Mat eql;
	//equalizeHist(mapgray,eql);
	//Mat cmp(map.rows,map.cols,CV_8U);
	//Mat cmp2(map.rows,map.cols,CV_8U);
	//cmp.setTo(0);
	//cmp2.setTo(0);
	//for(int i=0;i<map.rows;i++)
	//	for(int j=0;j<map.cols;j++)
	//		if(*eql.ptr<unsigned char>(i,j)>190)
	//			*cmp.ptr<unsigned char>(i,j)=255;
	//		else
	//			if(*eql.ptr<unsigned char>(i,j)>190)
	//				*cmp2.ptr<unsigned char>(i,j)=255;
	//Mat eqlegdes;
	//Canny(eql,eqlegdes,100,200);
	//imshow("eqle",eqlegdes);
	//imshow("eql",eql);
	//imshow("eql1",cmp);
	//imshow("eql2",cmp2);
	//Mat  lapl;
	//Laplacian(eql,lapl,CV_32F,3,3);
	////normalize(cmp,cmp,0,1,CV_MINMAX,CV_32F);
	////lapl=lapl.mul(cmp);
	////cmp=cmp+green;
	//lapl.convertTo(lapl,CV_8U);
	//imshow("lapl",lapl);
	//imshow("cmp",cmp);
	//mapgray=mapgray+lapl;
	//for(int i=0;i<map.rows;i++)
	//	for(int j=0;j<map.cols;j++)
	//		if(*eql.ptr<unsigned char>(i,j)<170)
	//	{
	//		*eql.ptr<unsigned char>(i,j)=0;
	//	}
	//		else
	//			*eql.ptr<unsigned char>(i,j)=1;

	//imshow("eql",eql);
	imshow("mp",map);
	//Mat grad=moduloGradient(eql);
	//imshow("grad",grad);


			imshow("map",mapgray);

	Mat mapegdes,panoegdes;
	Canny(mapgray,mapegdes,mapLowThr,mapHighThr);
	Canny(panogray,panoegdes,panoLowThr,panoHighThr);
	
	//mapegdes=mapegdes.mul(eql);

	findContours(mapegdes,mcontours,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	findContours(panoegdes,pcontours,cv::RetrievalModes::RETR_LIST,ContourApproximationModes::CHAIN_APPROX_SIMPLE);


	correctContours(mcontours);// findContours makes only closed curves, but in reality some of them are open
	correctContours(pcontours); // make such curves open

	filterContours(mcontours,15); // reject curves with little number of points
	filterContours(pcontours,5);

	std::vector<double> length(mcontours.size(),0);
	for(int i=0;i<mcontours.size();i++)
		length[i]=arcLength(mcontours[i],0);

	std::vector<double> length2(pcontours.size(),0);
	for(int i=0;i<pcontours.size();i++)
		length2[i]=arcLength(pcontours[i],0);

	mergeContours(mcontours,length,200); // merge adjacent contours
	mergeContours(pcontours,length2,200);

	filterContours(mcontours,25);
	filterContours(pcontours,20);
	Mat mcontim(map.size(),CV_8UC3);
	Mat pcontim(pano.size(),CV_8UC3);

			 CornersExtractor ext(81,144,175,1000);
			 CornersExtractor ext2(9,36,175,1000);

			 std::vector<std::vector<Point> > corn; 
			  std::vector<std::vector<Point> > corn2;
			 ext.extract(mcontours,corn);
			 ext2.extract(pcontours,corn2);

			 filterCorners(mcontours, 8,corn);
			 filterCorners(pcontours,8,corn2);
			 int sz=pcontours.size();
			 for(int i=0;i<sz;i++) //inner angle in descriptor is sensible to change path`s direction 
			 {                      //so for each curve add its inverse copy
				 pcontours.push_back(std::vector<Point>(pcontours[i].size(),Point()));
				 corn2.push_back(std::vector<Point>(corn2[i].size(),Point()));
				 for(int j=0;j<pcontours[i].size();j++)
					 pcontours.back()[pcontours[i].size()-j-1]=pcontours[i][j];
				 for(int j=0;j<corn2[i].size();j++)
					 corn2.back()[corn2[i].size()-j-1]=corn2[i][j];
			 }

			 std::vector<std::vector<char> > res(mcontours.size(),std::vector<char>(pcontours.size(),0));
			 ContourMatcherBouagar matcher(mcontours,corn,150,4,1);
			 matcher.precomputeQuery(pcontours,corn2);
			 matcher.match(res);

			/* std::vector<std::pair<std::pair<int,int>,double> > match;
			 ContourMatcher matcher(corn,3);
			 matcher.match(corn2,match);*/

			 std::vector<DMatch> matches;
			 std::vector<KeyPoint> pnt[2];
			 int c1=9,c2=71;   //show match between c1 train curve and c2 query curve
			 for(int i=0;i<corn[c1].size();i++)
				 pnt[0].push_back(KeyPoint(corn[c1][i],1));
			 for(int i=0;i<corn2[c2].size();i++)
				 pnt[1].push_back(KeyPoint(corn2[c2][i],1));
			 for(int i=0;i<matcher.matches_[c1][c2].size();i++)
				 matches.push_back(DMatch(matcher.matches_[c1][c2][i].first,matcher.matches_[c1][c2][i].second,0));

	//visualize contours, corners and matches
			 for(int i=0;i<pcontours.size();i++)
			 {
				 auto c=CV_RGB(i+100,i+100,i+100);
				 for(int j=1;j<pcontours[i].size();j++)
					 line(pcontim,pcontours[i][j-1],pcontours[i][j],c);
				 circle(pcontim,pcontours[i].front(),3,c);
				 circle(pcontim,pcontours[i].back(),3,c);
			 }

			  for(int i=0;i<mcontours.size();i++)
			 {
				 auto c=CV_RGB(i+100,i+100,i+100);
				 for(int j=1;j<mcontours[i].size();j++)
					 line(mcontim,mcontours[i][j-1],mcontours[i][j],c);
				 circle(mcontim,mcontours[i].front(),3,c);
				 circle(mcontim,mcontours[i].back(),3,c);
			 }
			 for(int i=0;i<mcontours.size();i++)
			 {
				 for(int j=0;j<corn[i].size();j++)
					 circle(mcontim,corn[i][j],3,CV_RGB(255,0,0));
			 }
			 for(int i=0;i<pcontours.size();i++)
			 {
				 for(int j=0;j<corn2[i].size();j++)
					 circle(pcontim,corn2[i][j],3,CV_RGB(255,0,0));
			 }
			 Mat mim;
			 drawMatches(mcontim,pnt[0],pcontim,pnt[1],matches,mim);

		 imshow("mapcont",mcontim);
		 imshow("panocont",pcontim.t());
		 imshow("mapegdes",mapegdes);
		 imshow("panoegdes",panoegdes);
		 imshow("mim",mim);

		 waitKey(10000000000);
	return 0;
}
#endif