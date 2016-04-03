#include "vector"
#include "iostream"
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

enum CalibrationType
{
	FOCUS_ONLY=2,NO_SKEW=4,FULL_MODEL=5
};

class CipollaFunction:public cv::MinProblemSolver::Function
{
	CalibrationType type;
    std::vector<cv::Mat> F;
	std::vector<double> W;
	double principialx,principialy;
public:
	CipollaFunction(std::vector<cv::Mat>& f,std::vector<double>& w,CalibrationType tp,double px=0.0,double py=0.0)
	{
		F=f;
		W=w;
		type=tp;
		principialx=px;
		principialy=py;
	}
	double calc(const double* x) const
		{
			double fx=*x,fy=*(x+1),princx=((type!=CalibrationType::FOCUS_ONLY)?(*(x+2)):(principialx)),
				princy=((type!=CalibrationType::FOCUS_ONLY)?(*(x+3)):(principialy)),
			skew=((type==CalibrationType::FULL_MODEL)?(*(x+4)):0.0);
			cv::Mat K=cv::Mat(cv::Matx33d(fx,skew,princx,0,fy,princy,0,0,1));
			double res=0.0;
			for(int i=0;i<F.size();i++)
					if(!F[i].empty())
			{
				cv::Mat E=K.t()*F[i]*K;
				cv::Mat vals =cv::SVD(E).w;
				double v1=*((double*)vals.data),v2=*((double*)(vals.data+vals.step[0]));
				res+=W[i]*(v1-v2)/v2;
			}
			return res;
		};
		int getDims() const
		{
			return (int)type;
		};

};

void calibrateCipolla(std::vector<double> w,std::vector<cv::Mat>& f,double& val,cv::Mat& K,CalibrationType type=CalibrationType::NO_SKEW)
	{
		double sum=0;
		for(int i=0;i<w.size();i++)
			sum+=w[i];
		for(int i=0;i<w.size();i++)
			w[i]/=sum;
		cv::Mat step,x0;
		double px=K.at<double>(0,2),py=K.at<double>(1,2);
		switch (type)
		{
		case FOCUS_ONLY:
			step=cv::Mat(cv::Matx<double,2,1>(20,20));
		x0=cv::Mat(cv::Matx<double,2,1>(K.at<double>(0,0),K.at<double>(1,1)));
			break;
		case NO_SKEW:
			step=cv::Mat(cv::Matx<double,4,1>(20,20,10,10));
		x0=cv::Mat(cv::Matx<double,4,1>(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2)));
			break;
		case FULL_MODEL:
			step=cv::Mat(cv::Matx<double,5,1>(20,20,10,10,0.001));
		x0=cv::Mat(cv::Matx<double,5,1>(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),K.at<double>(0,1)));
			break;
		default:
			break;
		}
		cv::Ptr<cv::MinProblemSolver::Function> pt(new CipollaFunction(f,w,type,px,py));
		auto solver=cv::DownhillSolver::create(pt,step,cv::TermCriteria((TermCriteria::MAX_ITER)+(TermCriteria::EPS),20000,(1e-08)));
		val=solver->minimize(x0);
		double fx=*((double*)x0.data),fy=*((double*)(x0.data+x0.step[0])),
			princx=((type!=CalibrationType::FOCUS_ONLY)?(*((double*)(x0.data+x0.step[0]*2))):(px)),
			princy=((type!=CalibrationType::FOCUS_ONLY)?(*((double*)(x0.data+x0.step[0]*3))):(py)),
			skew=((type==CalibrationType::FULL_MODEL)?(*((double*)(x0.data+x0.step[0]*4))):0.0);
		K=cv::Mat(cv::Matx33d(fx,skew,princx,0,fy,princy,0,0,1));
	};


