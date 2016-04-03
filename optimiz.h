
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

double LevMarqEstimation(Mat (*Jf)(std::vector<std::vector<double> >,std::vector<double>),
						 Mat (*Ef)(std::vector<std::vector<double> >,std::vector<double>),
						 std::vector<std::vector<double> >& consts,std::vector<double>& params,double eps)
{
	Mat J,e;
	double lambda=1;
	double n1,n2;
	std::vector<double> newx(params.size(),0);
	do
	{
		J=Jf(consts,params);
		e=Ef(consts,params);
		Mat JJ=J.t()*J;
		Mat eJ=-J.t()*e;
		double det=determinant(JJ);
		again:for(int i=0;i<JJ.rows;i++)
			if(det<1e-03)
				JJ.at<double>(i,i)+=lambda;
			else
				JJ.at<double>(i,i)*=1+lambda;
			//  std::cout<<JJ.inv()<<"\n"<<e<<"\n";
		Mat dx=JJ.inv(DecompTypes::DECOMP_SVD)*eJ;
		//std::cout<<dx<<"\n";
		double ndx=dx.dot(dx);
		for(int i=0;i<newx.size();i++)
			newx[i]=params[i]+dx.at<double>(i);
		Mat ne=Ef(consts,newx);
		 n1=e.dot(e),n2=ne.dot(ne);
		 if(abs(n1-n2)<1e-12)
			return e.dot(e);
		if(ne.dot(ne)<e.dot(e))
		{
			lambda /= 4;
			lambda=max(lambda,1e-07);
			for(int i=0;i<params.size();i++)
				params[i]=newx[i];
		}
		else
		{
			lambda *= 25;
			lambda=std::min(lambda,10000000.0);
			goto again;
		}
		if(abs(n1-n2)<1e-12)
			return e.dot(e);
	}
	while (n1>=eps);
	return e.dot(e);
}