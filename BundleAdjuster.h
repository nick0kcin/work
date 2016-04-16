#include <vector>
#include <opencv\cv.h>
#include <iostream>
using namespace cv;
class BundleAdjuster
{
	std::vector<Mat>* X;
	std::vector<Mat>* P;
	std::vector<std::vector<std::pair<Point2f,int> > > *corresp;
	std::vector<std::vector<std::pair<int,int> > > * ind;


	std::vector<Mat> U,V,E,ee,stepA,stepB,e_ee_e;
	std::vector<std::vector<Mat> > A,B,W,e,Y,S,e2;
	Mat step;
	Mat stepa,stepb;
	Mat SS,EE;
	Mat error,error2;
	int maxiter;
	double eps;


	Mat computeJAij(int cam,int pnt)
	{
		Mat res(2,12,CV_64F);
		res.setTo(0);
		double *pt=res.ptr<double>();
			double* ptx=(*X)[(*corresp)[cam][pnt].second].ptr<double>();
			for(int j=0;j<4;j++)
			{
			pt[j]=*(ptx+j);
			pt[j+8]=-(*(ptx+j))*(*corresp)[cam][pnt].first.x;
			pt[j+16]=*(ptx+j);
			pt[j+20]=-(*(ptx+j))*(*corresp)[cam][pnt].first.y;
		}
		return res;
	};
	Mat computeJBij(int cam,int pnt)
		{
		Mat res(2,3,CV_64F);
		res.setTo(0);
		double* ptr=(*P)[cam].ptr<double>();
				double *pt=res.ptr<double>();
				for(int j=0;j<3;j++)
				{
					pt[j]=ptr[j]-ptr[j+8]*(*corresp)[cam][pnt].first.x;
					pt[3+j]=ptr[j+4]-ptr[j+8]*(*corresp)[cam][pnt].first.y;
				}
		return res;
	};
  void computeError(int cam,int pnt,std::vector<std::vector<Mat> > & er)
		{
		double* ptr=er[cam][pnt].ptr<double>();
		Mat Padd=Mat(Size(4,3),CV_64F,step.ptr<double>(12*cam));
		double * xpt=step.ptr<double>(P->size()*12+(*corresp)[cam][pnt].second*3);
				Mat Xadd=Mat(Matx41d(xpt[0],xpt[1],xpt[2],0.0));
				Mat p=((*P)[cam]+Padd)*((*X)[(*corresp)[cam][pnt].second]+Xadd);
				double* pt=p.ptr<double>();
				*ptr=pt[0]/pt[2]-(*corresp)[cam][pnt].first.x;
				ptr++;
				*ptr=pt[1]/pt[2]-(*corresp)[cam][pnt].first.y;
				ptr++;
	};


	double computenewError()
	{

		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
				computeError(i,j,e2);
		return norm(error2);
	};
	void mulDiag(Mat& m,double d)
	{
		double* pt=m.ptr<double>();
		for(int i=0;i<m.rows;i++)
		{
			*pt*=d;
			pt+=m.cols+1;
		}
	}
	void computeStep(double lambda,bool newJ,double& errorNorm)
	{
		if(newJ)
		{
		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
		{
			A[i][j]=computeJAij(i,j);
			B[i][j]=computeJBij(i,j);
			computeError(i,j,e);
		}
			errorNorm=norm(error);
			for(int i=0;i<X->size();i++)
			{
				V[i]=Mat::zeros(B[0][0].cols,B[0][0].cols,CV_64F);
				E[P->size()+i]=Mat::zeros(B[0][0].cols,1,CV_64F);
			}
			for(int i=0;i<P->size();i++)
			{
				U[i]=Mat::zeros(A[i][0].cols,A[i][0].cols,CV_64F);
				E[i]=Mat::zeros(A[i][0].cols,1,CV_64F);
			}
		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
		{
			U[i]+=A[i][j].t()*A[i][j];
			V[(*corresp)[i][j].second]+=B[i][j].t()*B[i][j];
			W[i][j]=A[i][j].t()*B[i][j];
			E[i]+=A[i][j].t()*e[i][j];
			E[P->size()+(*corresp)[i][j].second]+=B[i][j].t()*e[i][j];
		}
		}
			for(int i=0;i<U.size();i++)
		mulDiag(U[i],1+lambda);
			for(int i=0;i<V.size();i++)
		mulDiag(V[i],1+lambda);

		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
				Y[i][j]=W[i][j]*V[(*corresp)[i][j].second].inv(DECOMP_SVD);

		SS.setTo(0);
		EE.setTo(0);
		for(int i=0;i<ind->size();i++)
			for(int j=0;j<(*ind)[i].size();j++)
			{
				for(int k=0;k<(*ind)[i].size();k++)
					S[(*ind)[i][j].first][(*ind)[i][k].first]-=Y[(*ind)[i][j].first][(*ind)[i][j].second]*W[(*ind)[i][k].first][(*ind)[i][k].second].t();
				ee[(*ind)[i][j].first]-=Y[(*ind)[i][j].first][(*ind)[i][j].second]*E[P->size()+i];
			}
		for(int i=0;i<P->size();i++)
		{
			S[i][i]+=U[i];
			ee[i]+=E[i];
		}
		stepa=SS.inv(DECOMP_SVD)*EE;
		for(int i=0;i<X->size();i++)
			e_ee_e[i]=E[P->size()+i];
		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
				e_ee_e[(*corresp)[i][j].second]-=W[i][j].t()*stepA[i];
		for(int i=0;i<X->size();i++)
			stepB[i]=V[i].inv()*e_ee_e[i];


	};

	void relax()
	{
		double* pts=step.ptr<double>();
		for(int i=0;i<P->size();i++)
		{
			Mat Padd(Size(4,3),CV_64F,pts);
			(*P)[i]+=Padd;
			pts+=12;
		}
		for(int i=0;i<X->size();i++)
		{
			double* pt=(*X)[i].ptr<double>();
			for(int i=0;i<3;i++)
				pt[i]+=pts[i];
			pts+=3;
		}
	};

	public:
		BundleAdjuster(std::vector<Mat>* X_,std::vector<Mat>* P_,std::vector<std::vector<std::pair<Point2f,int> > >* corresp_,
			std::vector<std::vector<std::pair<int,int> > > * ind_,int maxiter_=1000,double eps_=1e-08)
		{
			X=X_;
			P=P_;
			corresp=corresp_;
			ind=ind_;
			maxiter=maxiter_;
			eps=eps_;
			A.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<A.size();i++)
				A[i].assign((*corresp)[i].size(),Mat());
			B.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<B.size();i++)
				B[i].assign((*corresp)[i].size(),Mat());
			V.assign(X->size(),Mat());
			U.assign(P->size(),Mat());
			W.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<W.size();i++)
				W[i].assign((*corresp)[i].size(),Mat());
			e.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<e.size();i++)
				e[i].assign((*corresp)[i].size(),Mat());
			E.assign(corresp->size()+X->size(),Mat());
			Y.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<Y.size();i++)
				Y[i].assign((*corresp)[i].size(),Mat());

			SS=Mat::zeros(12*P->size(),12*P->size(),CV_64F);
			S.assign(P->size(),std::vector<Mat>(P->size(),Mat()));
			for(int i=0;i<P->size();i++)
				for(int j=0;j<P->size();j++)
					S[i][j]=SS(Range(i*12,i*12+12),Range(j*12,j*12+12));

			EE=Mat::zeros(12*P->size(),1,CV_64F);
			ee.assign(P->size(),Mat());
			for(int j=0;j<P->size();j++)
				ee[j]=EE.rowRange(Range(j*12,j*12+12));

			step=Mat(P->size()*12+X->size()*3,1,CV_64F);
			step.setTo(0);
			stepa=step.rowRange(0,P->size()*12);
			stepb=step.rowRange(P->size()*12,step.rows);
			stepA.assign(P->size(),Mat());
			stepB.assign(X->size(),Mat());
			for(int i=0;i<P->size();i++)
				stepA[i]=stepa.rowRange(i*12,i*12+12);
			for(int i=0;i<X->size();i++)
				stepB[i]=stepb.rowRange(i*3,i*3+3);
			e_ee_e.assign(X->size(),Mat());
			int total=0;
			for(int i=0;i<corresp->size();i++)
				total+=(*corresp)[i].size();
			error=Mat::zeros(2*total,1,CV_64F);
			int cur=0;
			for(int i=0;i<corresp->size();i++)
				for(int j=0;j<(*corresp)[i].size();j++)
				{
					e[i][j]=error.rowRange(cur,cur+2);
					cur+=2;
				}
			error2=Mat::zeros(2*total,1,CV_64F);
			 cur=0;
			 e2.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<e2.size();i++)
				e2[i].assign((*corresp)[i].size(),Mat());

			for(int i=0;i<corresp->size();i++)
				for(int j=0;j<(*corresp)[i].size();j++)
				{
					e2[i][j]=error2.rowRange(cur,cur+2);
					cur+=2;
				}
		}
	double adjust()
	{
		double norm1,norm2;
		double lambda=0.001;
		bool newJ=1;
		for(int iter=0;iter<maxiter;iter++)
		{
			step.setTo(0);
			computeStep(lambda,newJ,norm1);
			norm2=computenewError();
			if(abs(norm1-norm2)<eps)
				return norm1;
			if(norm1>norm2)
			{
				relax();
				lambda*=10;
				lambda=min(lambda,1e09);
				newJ=1;
				if(abs(norm2)<eps)
				return norm2;
			}
			else
			{
				for(int i=0;i<U.size();i++)
		mulDiag(U[i],1/(1+lambda));
			for(int i=0;i<V.size();i++)
		mulDiag(V[i],1/(1+lambda));

				lambda/=10;
				lambda=max(lambda,1e-14);
				newJ=0;
			}

		}
	};
};