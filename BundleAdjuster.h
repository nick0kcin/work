#include <vector>
#include <opencv\cv.h>
#include <opencv2\calib3d.hpp>
#include <opencv2\ccalib.hpp>
#include <opencv2\calib3d\calib3d_c.h>
#include <iostream>
using namespace cv;

void projectPoints2( InputArray _opoints,
                        InputArray _rvec,
                        InputArray _tvec,
                        InputArray _cameraMatrix,
                        OutputArray _ipoints,
                        OutputArray _jacobian,
                        double aspectRatio=0 )
{
    Mat opoints = _opoints.getMat();
    int npoints = opoints.checkVector(3), depth = opoints.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_64F));

    CvMat dpdrot, dpdt, dpdf, dpdc, dpddist;
    CvMat *pdpdrot=0, *pdpdt=0, *pdpdf=0, *pdpdc=0, *pdpddist=0;

    _ipoints.create(npoints, 1, CV_MAKETYPE(depth, 2), -1, true);
    CvMat c_imagePoints = _ipoints.getMat();
    CvMat c_objectPoints = opoints;
    Mat cameraMatrix = _cameraMatrix.getMat();

    Mat rvec = _rvec.getMat(), tvec = _tvec.getMat();
    CvMat c_cameraMatrix = cameraMatrix;
    CvMat c_rvec = rvec, c_tvec = tvec;


    if( _jacobian.needed() )
    {
        _jacobian.create(npoints*2, 3+3+2+2, CV_64F);
        Mat jacobian = _jacobian.getMat();
        pdpdrot = &(dpdrot = jacobian.colRange(0, 3));
        pdpdt = &(dpdt = jacobian.colRange(3, 6));
        pdpdf = &(dpdf = jacobian.colRange(6, 8));
        pdpdc = &(dpdc = jacobian.colRange(8, 10));
    }

	cvProjectPoints2( &c_objectPoints, &c_rvec, &c_tvec, &c_cameraMatrix, NULL,
                      &c_imagePoints, pdpdrot, pdpdt, pdpdf, pdpdc, NULL, aspectRatio );
}







class LevMarqSparse
{
protected:	std::vector<Mat>* X;
	std::vector<Mat>* P;
	std::vector<std::vector<std::pair<Point2f,int> > > *corresp;
	std::vector<std::vector<std::pair<int,int> > > * ind;
	std::vector<char> * isfixed;
	Mat K;


	std::vector<Mat> U,V,E,ee,stepA,e_ee_e,Vinv,newX,newP;
	std::vector<std::vector<Mat> > A,B,W,e,Y,S,e2;
	Mat stepa;
	Mat SS,EE;
	Mat error,error2;
	int maxiter;
	double eps;

	 int cameraParams;
virtual void computeError()=0;
virtual double computenewError()=0;

	void mulDiag(Mat& m,double d)
	{
		double* pt=m.ptr<double>();
		for(int i=0;i<m.rows;i++)
		{
			//*pt*=d;
			(*pt)+=d;
			pt+=m.cols+1;
		}
	}
	void computeStep(double lambda,bool newJ)
	{
		if(newJ)
		{
			computeError();
			for(int i=0;i<X->size();i++)
			{
				V[i].setTo(0);
				E[i+P->size()].setTo(0);
			}
			for(int i=0;i<P->size();i++)
			{
				U[i].setTo(0);
				E[i].setTo(0);
			}
			auto Uit=U.begin(),Eit=E.begin();
			auto Wit=W.begin();
		for(int i=0;i<corresp->size();i++)
		{
			auto Witj=Wit->begin();
			for(int j=0;j<(*corresp)[i].size();j++)
		{
			int p=(*corresp)[i][j].second;
			Mat Aa=A[i][j],Bb=B[i][j],Ee=e[i][j];
			Mat Aat,Bbt;
			transpose(Aa,Aat);
			transpose(Bb,Bbt);
			Mat r1=*Uit,r2=V[p],r3=*Eit,r4=E[P->size()+p];
			gemm(Aat,Aa,1,r1,1,r1,0);
			gemm(Bbt,Bb,1,r2,1,r2,0);
			gemm(Aat,Bb,1,noArray(),1,*Witj,0);
			gemm(Aat,Ee,1,r3,1,r3,0);
			gemm(Bbt,Ee,1,r4,1,r4,0);
			Witj++;
		}
			Uit++,Eit++,Wit++;
		}
		}
			for(int i=0;i<U.size();i++)
		mulDiag(U[i],1+lambda);
			for(int i=0;i<V.size();i++)
			{
				mulDiag(V[i],/*1+*/lambda);
				invert(V[i],Vinv[i],DECOMP_SVD);
			}
		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
				gemm(W[i][j],Vinv[(*corresp)[i][j].second],1,noArray(),1,Y[i][j],0);

		SS.setTo(0);
		EE.setTo(0);
		for(int i=0;i<ind->size();i++)
			for(int j=0;j<(*ind)[i].size();j++)
			{
				auto p1=(*ind)[i][j];
				Mat Yy=Y[p1.first][p1.second];
				for(int k=j;k<(*ind)[i].size();k++)
				{
					auto p2=(*ind)[i][k];
					Mat Ss=S[p1.first][p2.first];
					gemm(Yy,W[p2.first][p2.second],-1,Ss,1,Ss,GEMM_2_T);
				}
				Mat tee=ee[p1.first];
				gemm(Yy,E[P->size()+i],-1,tee,1,tee,0);
			}
			completeSymm(SS,0);
		for(int i=0;i<P->size();i++)
		{
			Mat ts=S[i][i],te=ee[i];
			add(ts,U[i],ts);
			add(te,E[i],te);
		}
		solve(SS,EE,stepa,DECOMP_SVD);
		//double nr=norm(stepa,NORM_L2SQR);
		for(int i=0;i<P->size();i++)
		{
			Mat np=stepa.rowRange(i*cameraParams,(i+1)*cameraParams);
			subtract((*P)[i],np,newP[i]);
		}
		for(int i=0;i<X->size();i++)
			e_ee_e[i].setTo(0);
		for(int i=0;i<corresp->size();i++)
			for(int j=0;j<(*corresp)[i].size();j++)
			{
				auto p=e_ee_e[(*corresp)[i][j].second];
				gemm(W[i][j],stepA[i],-1,p,1,p,GEMM_1_T);
			}
			for(int i=0;i<X->size();i++)
			{
				Mat te=e_ee_e[i];
				add(te,E[P->size()+i],te);
			}

		for(int i=0;i<X->size();i++)
		{
			gemm(Vinv[i],e_ee_e[i],-1,(*X)[i],1,newX[i],GEMM_3_T);
			transpose(newX[i],newX[i]);
			//nr+=norm((newX[i]-(*X)[i]),NORM_L2SQR);
		}

		//std::cout<<nr<<"\n";
	};

	void relax()
	{
		//double* pts=(double*)step.data;
		for(int i=0;i<P->size();i++)
		{
			newP[i].copyTo((*P)[i]);
			//Mat Padd(cameraParams,1,CV_64F,pts);
			//(*P)[i]-=Padd;
			//pts+=cameraParams;
		}
		for(int i=0;i<X->size();i++)
		{
			newX[i].copyTo((*X)[i]);
		/*	double* pt=(*X)[i].ptr<double>();
			for(int i=0;i<3;i++)
				pt[i]+=pts[i];
			pts+=3;*/
		}
	};

	public:
		LevMarqSparse(int camparam_,std::vector<Mat>* X_,std::vector<Mat>* P_,std::vector<std::vector<std::pair<Point2f,int> > >* corresp_,
			std::vector<std::vector<std::pair<int,int> > > * ind_,std::vector<char> * fix_,int maxiter_=50,double eps_=1e-5)
		{
			cameraParams=camparam_;
			X=X_;
			P=P_;
			corresp=corresp_;
			ind=ind_;
			isfixed=fix_;
			maxiter=maxiter_;
			eps=eps_;
			newX.assign(X->size(),Mat(1,3,CV_64F));
			newP.assign(P->size(),Mat());
			for(int i=0;i<newX.size();i++)
				newX[i]=Mat(1,3,CV_64F);
			for(int i=0;i<P->size();i++)
				newP[i]=Mat(cameraParams,1,CV_64F);

			A.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<A.size();i++)
				A[i].assign((*corresp)[i].size(),Mat(2,cameraParams,CV_64F));
			for(int i=0;i<A.size();i++)
				for(int j=0;j<A[i].size();j++)
					A[i][j]=Mat(2,cameraParams,CV_64F);

			B.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<B.size();i++)
				B[i].assign((*corresp)[i].size(),Mat(2,3,CV_64F));
			for(int i=0;i<B.size();i++)
				for(int j=0;j<B[i].size();j++)
					B[i][j]=Mat(2,3,CV_64F);

			V.assign(X->size(),Mat(3,3,CV_64F));
			Vinv.assign(X->size(),Mat(3,3,CV_64F));
			for(int i=0;i<V.size();i++)
			{
				V[i]=Mat(3,3,CV_64F);
				Vinv[i]=Mat(3,3,CV_64F);
			}
			U.assign(P->size(),Mat(cameraParams,cameraParams,CV_64F));
			for(int i=0;i<U.size();i++)
				U[i]=Mat(cameraParams,cameraParams,CV_64F);
			E.assign(corresp->size()+X->size(),Mat());
			for(int i=0;i<E.size();i++)
				E[i]=Mat();

				for(int i=0;i<X->size();i++)
			{
				V[i]=Mat::zeros(3,3,CV_64F);
				E[P->size()+i]=Mat::zeros(3,1,CV_64F);
			}
			for(int i=0;i<P->size();i++)
			{
				U[i]=Mat::zeros(cameraParams,cameraParams,CV_64F);
				E[i]=Mat::zeros(cameraParams,1,CV_64F);
			}





			W.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<W.size();i++)
				W[i].assign((*corresp)[i].size(),Mat());
			for(int i=0;i<W.size();i++)
				for(int j=0;j<W[i].size();j++)
					W[i][j]=Mat();

			e.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<e.size();i++)
				e[i].assign((*corresp)[i].size(),Mat());

			Y.assign(corresp->size(),std::vector<Mat>());
			for(int i=0;i<Y.size();i++)
				Y[i].assign((*corresp)[i].size(),Mat(cameraParams,3,CV_64F));
			for(int i=0;i<Y.size();i++)
				for(int j=0;j<Y[i].size();j++)
					Y[i][j]=Mat(cameraParams,3,CV_64F);

			SS=Mat::zeros(cameraParams*P->size(),cameraParams*P->size(),CV_64F);
			S.assign(P->size(),std::vector<Mat>(P->size(),Mat()));
			for(int i=0;i<P->size();i++)
				for(int j=0;j<P->size();j++)
					S[i][j]=SS(Range(i*cameraParams,i*cameraParams+cameraParams),Range(j*cameraParams,j*cameraParams+cameraParams));

			EE=Mat::zeros(cameraParams*P->size(),1,CV_64F);
			ee.assign(P->size(),Mat());
			for(int j=0;j<P->size();j++)
				ee[j]=EE.rowRange(Range(j*cameraParams,j*cameraParams+cameraParams));

			//step=Mat(P->size()*cameraParams+X->size()*3,1,CV_64F);
			//step.setTo(0);
			//stepa=step.rowRange(0,P->size()*cameraParams);
			stepa=Mat(P->size()*cameraParams,1,CV_64F);
			//stepb=step.rowRange(P->size()*cameraParams,step.rows);
			stepA.assign(P->size(),Mat());
			//stepB.assign(X->size(),Mat());
			for(int i=0;i<P->size();i++)
				stepA[i]=stepa.rowRange(i*cameraParams,i*cameraParams+cameraParams);
			//for(int i=0;i<X->size();i++)
			//	stepB[i]=stepb.rowRange(i*3,i*3+3);

			e_ee_e.assign(X->size(),Mat(3,1,CV_64F));
			for(int i=0;i<e_ee_e.size();i++)
				e_ee_e[i]=Mat(3,1,CV_64F);

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
	Mat adjust()
	{
		double norm1,norm2;
		double lambda=10.245560;
		bool newJ=1;
		for(int iter=0;iter<maxiter;)
		{
			//long long t=cv::getTickCount();
		 	computeStep(lambda,newJ);
			if(!iter)
				norm1=norm(error);
			//step=-step;
			//double dx=norm(step);
				norm2=computenewError();
			if(abs(norm1-norm2)<eps)
				return error;
			if(norm1>norm2)
			{
				relax();
				iter++;
				//if(norm1-norm2>25)
				{
				lambda/=10;
				lambda=max(lambda,1e-09);
				}
				norm1=norm2;
				newJ=1;
				if(abs(norm2)<eps)
				return error2;
			}
			else
			{
				for(int i=0;i<U.size();i++)
		mulDiag(U[i],/*1/(1+lambda)*/-lambda);
			for(int i=0;i<V.size();i++)
				mulDiag(V[i],/*1/(1+lambda)*/-lambda);
				lambda*=10;
				//lambda=min(lambda,1e16);
				newJ=0;
			}

			//t=getTickCount()-t;
			//std::cout<<t*1000/getTickFrequency()<<"\n";
		}
		return error;
	};
};

class BundleAdjuster :public LevMarqSparse
{
public:
	BundleAdjuster(std::vector<Mat>* X_,std::vector<Mat>* P_,std::vector<std::vector<std::pair<Point2f,int> > >* corresp_,
			std::vector<std::vector<std::pair<int,int> > > * ind_,std::vector<char> * fix_,int maxiter_=50,double eps_=1e-5):
	LevMarqSparse(10,X_,P_,corresp_, ind_, fix_, maxiter_,eps_)
	{
	};
	double computenewError()
		{
		Mat Pnew(10,1,CV_64F);
		double* erptr=(double*)error2.data;
		for(int cam=0;cam<(*corresp).size();cam++)
		{
			Pnew=newP[cam];
		Mat rvec=Pnew.rowRange(0,3);
		Mat tvec=Pnew.rowRange(3,6);
		double* pt=Pnew.ptr<double>(6);
		double fx=pt[0],fy=pt[1],cx=pt[2],cy=pt[3];
		double ar[9]={fx,0,cx,0,fy,cy,0,0,1};
		Mat K(3,3,CV_64F,ar);
		Mat Proj;
		for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		{
		auto p=(*corresp)[cam][pnt];
		projectPoints(newX[p.second],rvec,tvec,K,noArray(),Proj);
		double px=*((double*)Proj.data),py=*((double*)(Proj.data+8));
		erptr[0]=px-p.first.x;
		erptr[1]=py-p.first.y;
		erptr+=2;
		}
		}
		return norm(error2);
	};

	  void computeError()
		{
		double* erptr=(double*)error.data;
		for(int cam=0;cam<(*corresp).size();cam++)
		{
		Mat Pnew=(*P)[cam];
		Mat rvec=Pnew.rowRange(0,3);
		Mat tvec=Pnew.rowRange(3,6);
		double* pt=Pnew.ptr<double>(6);
		double fx=pt[0],fy=pt[1],cx=pt[2],cy=pt[3];
		double ar[9]={fx,0,cx,0,fy,cy,0,0,1};
		Mat K=Mat(3,3,CV_64F,ar);
		Mat Proj(1,2,CV_64F);
		double R[9];
		double* t=(double*)tvec.data;
		Mat _R(3,3,CV_64F,R);
		Rodrigues(rvec,_R);
		for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		{
		auto p=(*corresp)[cam][pnt];
		Mat x=(*X)[p.second];
		projectPoints2(x,rvec,tvec,K,Proj,cam!=0?A[cam][pnt]:noArray());
		if(cam==0)
			A[cam][pnt].setTo(0);
		double px=*((double*)Proj.data),py=*((double*)(Proj.data+8));
		erptr[0]=px-p.first.x;
		erptr[1]=py-p.first.y;
		erptr+=2;
		//if(!(*isfixed)[p.second])
		{
		double *xp=(double*)x.data;
		double xx = (R[0] * xp[0] + R[1] * xp[1] + R[2] * xp[2] + t[0]);
        double yy = (R[3] * xp[0] + R[4] * xp[1] + R[5] * xp[2] + t[1]);
        double zz = R[6] * xp[0] + R[7] * xp[1] + R[8] * xp[2] + t[2];
		double xs=xx/zz;
		double ys=yy/zz;
		double coeff[6]={zz,0,-xx,0,zz,-yy};
		Mat der(2,3,CV_64F,coeff);
		Mat bB=B[cam][pnt];
		gemm(der,_R,1/(zz*zz),noArray(),1,bB);
		bB.row(0)*=fx;
		bB.row(1)*=fy;
		}
		//else
		//	B[cam][pnt].setTo(0);
		}
		}
	};


};


class BundleAdjusterFM: public LevMarqSparse
{
public:	BundleAdjusterFM(std::vector<Mat>* X_,std::vector<Mat>* P_,std::vector<std::vector<std::pair<Point2f,int> > >* corresp_,
			std::vector<std::vector<std::pair<int,int> > > * ind_,std::vector<char> * fix_,int maxiter_=50,double eps_=1e-5):
	LevMarqSparse(9,X_,P_,corresp_, ind_, fix_, maxiter_,eps_)
	{
	};
	   double computenewError()
  {
	  double* erptr=(double*)error2.data;
	  for(int cam=0;cam<(*corresp).size();cam++)
	  {
		  double* Pnew=newP[cam].ptr<double>();
		  double a=Pnew[0],b=Pnew[1],c=Pnew[2],
			  d=Pnew[3],e=Pnew[4],f=Pnew[5],
			  x=Pnew[6],y=Pnew[7],z=Pnew[8];
		  for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		  {
			  auto p=(*corresp)[cam][pnt];
			  double* _x=newX[p.second].ptr<double>(0);
			  double _X=_x[0],_Y=_x[1],_Z=_x[2];
			  double nx,ny,nz;
			  if(z==-1)
			  {
				 nx=  x + _X*(b + y*(a*x + b*y)) + _Y*(d + y*(c*x + d*y)) + _Z*(f + y*(e*x + f*y));
                 ny= y - _X*(a + x*(a*x + b*y)) - _Y*(c + x*(c*x + d*y)) - _Z*(e + x*(e*x + f*y));
                 nz=    - _X*(a*y - b*x) - _Y*(c*y - d*x) - _Z*(e*y - f*x) - 1;
			  }
			  else if(y==-1)
			  {
				 nx= x - _X*(b + z*(a*x + b*z)) - _Y*(d + z*(c*x + d*z)) - _Z*(f + z*(e*x + f*z));
                 ny=     _X*(a*z - b*x) + _Y*(c*z - d*x) + _Z*(e*z - f*x) - 1;
                 nz= z + _X*(a + x*(a*x + b*z)) + _Y*(c + x*(c*x + d*z)) + _Z*(e + x*(e*x + f*z));
			  }
			  else if(x==-1)
			  {
				nx=   - _X*(a*z - b*y) - _Y*(c*z - d*y) - _Z*(e*z - f*y) - 1;
                ny=y + _X*(b + z*(a*y + b*z)) + _Y*(d + z*(c*y + d*z)) + _Z*(f + z*(e*y + f*z));
                nz=z - _X*(a + y*(a*y + b*z)) - _Y*(c + y*(c*y + d*z)) - _Z*(e + y*(e*y + f*z));
			  }
			  else
			  {
				  nx=_X;
				  ny=_Y;
				  nz=_Z;
			  }

			  double zinv=1/nz,xpr=nx*zinv,ypr=ny*zinv;
			  erptr[0]=xpr-p.first.x;
			  erptr[1]=ypr-p.first.y;
			  erptr+=2;
		  }
	  }
	  return norm(error2);
  };
	      void computeError()
  {
	  double* erptr=(double*)error.data;
	  Mat Ja,Jb;
	  double _pnt[3];
	  Mat pn(3,1,CV_64F,_pnt);
	  Mat xg;
	  for(int cam=0;cam<(*corresp).size();cam++)
	  {
		  double* Pnew=(*P)[cam].ptr<double>();
		  double a=Pnew[0],b=Pnew[1],c=Pnew[2],
			  d=Pnew[3],e=Pnew[4],f=Pnew[5],
			  x=Pnew[6],y=Pnew[7],z=Pnew[8];
		  for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		  {
			  auto p=(*corresp)[cam][pnt];
			  double* _x=(*X)[p.second].ptr<double>(0);
			  double _X=_x[0],_Y=_x[1],_Z=_x[2];
			  double nx,ny,nz;
			  if(z==-1)
			  {
				 nx=  x + _X*(b + y*(a*x + b*y)) + _Y*(d + y*(c*x + d*y)) + _Z*(f + y*(e*x + f*y));
                 ny= y - _X*(a + x*(a*x + b*y)) - _Y*(c + x*(c*x + d*y)) - _Z*(e + x*(e*x + f*y));
                 nz=    - _X*(a*y - b*x) - _Y*(c*y - d*x) - _Z*(e*y - f*x) - 1;
				 double ja[27]={
_X*x*y, _X*(y*y + 1), _Y*x*y, _Y*(y*y + 1),_Z*x*y, _Z*(y*y + 1),_X*a*y + _Y*c*y + _Z*e*y + 1, _X*(a*x + 2*b*y) + _Y*(c*x + 2*d*y) + _Z*(e*x + 2*f*y),0,
 -_X*(x*x + 1),-_X*x*y,-_Y*(x*x + 1),-_Y*x*y, -_Z*(x*x + 1),-_Z*x*y, -_X*(2*a*x + b*y) - _Y*(2*c*x + d*y) - _Z*(2*e*x + f*y),1-_Y*d*x-_Z*f*x - _X*b*x,0,
 -_X*y,         _X*x,         -_Y*y,         _Y*x,         -_Z*y,         _Z*x,    _X*b + _Y*d + _Z*f,    - _X*a - _Y*c - _Z*e,0};
				 Ja=Mat(3,9,CV_64F,ja);
				 double jb[9]={
					    b + y*(a*x + b*y),   d + y*(c*x + d*y),   f + y*(e*x + f*y),
 - a - x*(a*x + b*y), - c - x*(c*x + d*y), - e - x*(e*x + f*y),
           b*x - a*y,           d*x - c*y,           f*x - e*y};
				 Jb=Mat(3,3,CV_64F,jb);
			  }
			  else if(y==-1)
			  {
				 nx= x - _X*(b + z*(a*x + b*z)) - _Y*(d + z*(c*x + d*z)) - _Z*(f + z*(e*x + f*z));
                 ny=     _X*(a*z - b*x) + _Y*(c*z - d*x) + _Z*(e*z - f*x) - 1;
                 nz= z + _X*(a + x*(a*x + b*z)) + _Y*(c + x*(c*x + d*z)) + _Z*(e + x*(e*x + f*z));
				 double ja[27]={
-_X*x*z, -_X*(z*z + 1),-_Y*x*z, -_Y*(z*z + 1),-_Z*x*z, -_Z*(z*z + 1),1 - _Y*c*z - _Z*e*z -_X*a*z,0,-_X*(a*x + 2*b*z)-_Y*(c*x + 2*d*z)-_Z*(e*x + 2*f*z),
_X*z,-_X*x,_Y*z,-_Y*x,_Z*z,-_Z*x,                 - _X*b - _Y*d - _Z*f,0,                                       _X*a + _Y*c + _Z*e,
_X*(x*x + 1),_X*x*z, _Y*(x*x + 1),_Y*x*z, _Z*(x*x + 1),_Z*x*z, _X*(2*a*x + b*z) + _Y*(2*c*x + d*z) + _Z*(2*e*x + f*z),0,_X*b*x + _Y*d*x + _Z*f*x + 1};
				  Ja=Mat(3,9,CV_64F,ja);

				double jb[9]={ - b - z*(a*x + b*z), - d - z*(c*x + d*z), - f - z*(e*x + f*z),
          a*z - b*x,           c*z - d*x,           e*z - f*x,
   a + x*(a*x + b*z),   c + x*(c*x + d*z),   e + x*(e*x + f*z)};
				 Jb=Mat(3,3,CV_64F,jb);
			  }
			  else if(x==-1)
			  {
				nx=   - _X*(a*z - b*y) - _Y*(c*z - d*y) - _Z*(e*z - f*y) - 1;
                ny=y + _X*(b + z*(a*y + b*z)) + _Y*(d + z*(c*y + d*z)) + _Z*(f + z*(e*y + f*z));
                nz=z - _X*(a + y*(a*y + b*z)) - _Y*(c + y*(c*y + d*z)) - _Z*(e + y*(e*y + f*z));
				double ja[27]={
-_X*z,_X*y,-_Y*z,_Y*y,-_Z*z,_Z*y,                     0,     _X*b + _Y*d + _Z*f,                                   - _X*a - _Y*c - _Z*e,
_X*y*z, _X*(z*z + 1),_Y*y*z, _Y*(z*z + 1),_Z*y*z, _Z*(z*z + 1),0,_X*a*z + _Y*c*z + _Z*e*z + 1, _X*(a*y + 2*b*z) + _Y*(c*y + 2*d*z) + _Z*(e*y + 2*f*z),
-_X*(y*y + 1),-_X*y*z, -_Y*(y*y + 1),-_Y*y*z, -_Z*(y*y + 1),-_Z*y*z,0, -_X*(2*a*y + b*z)-_Y*(2*c*y + d*z)-_Z*(2*e*y + f*z),1-_Y*d*y-_Z*f*y-_X*b*y};
				Ja=Mat(3,9,CV_64F,ja);

			double jb[9]={          b*y - a*z,           d*y - c*z,           f*y - e*z,
   b + z*(a*y + b*z),   d + z*(c*y + d*z),   f + z*(e*y + f*z),
 - a - y*(a*y + b*z), - c - y*(c*y + d*z), - e - y*(e*y + f*z)};
			Jb=Mat(3,3,CV_64F,jb);
			  }
			  else
			  {
				  nx=_X;
				  ny=_Y;
				  nz=_Z;
			  }

			  double zinv=1/nz,xpr=nx*zinv,ypr=ny*zinv;
			  erptr[0]=xpr-p.first.x;
			  erptr[1]=ypr-p.first.y;
			  erptr+=2;




			  if(cam==0)
			  {
				  A[cam][pnt].setTo(0);
				  Jb=Mat::eye(3,3,CV_64F);
			  }
			  else
			  {
				  addWeighted(Ja.row(0),zinv,Ja.row(2),-xpr*zinv,0,A[cam][pnt].row(0));
				  addWeighted(Ja.row(1),zinv,Ja.row(2),-ypr*zinv,0,A[cam][pnt].row(1));
			  }
			  if(!(*isfixed)[p.second])
			  {
				  addWeighted(Jb.row(0),zinv,Jb.row(2),-xpr*zinv,0,B[cam][pnt].row(0));
				  addWeighted(Jb.row(1),zinv,Jb.row(2),-ypr*zinv,0,B[cam][pnt].row(1));
			  }
			  else
				  B[cam][pnt].setTo(0);
		  }
	  }
  };

		  static Mat construstCameraMat(Mat cmv)
		  {
			  double * cm=cmv.ptr<double>();
			  double a=cm[0],b=cm[1],c=cm[2],
			  d=cm[3],e=cm[4],f=cm[5],
			  x=cm[6],y=cm[7],z=cm[8];
			  Mat Fund,ex,M;
			  if(z==-1)
			  {
				  double  Ex[9]={ 0,1,y,
					               -1,0,-x,
								   -y,x,0};
				    
				  double F[9]={a,b,a*x+b*y,
					           c,d,c*x+d*y,
							   e,f,e*x+f*y};
				  Fund=Mat(3,3,CV_64F,F);
				  ex=Mat(3,3,CV_64F,Ex);
				   M=ex*Fund;
				  M.push_back(Mat(Matx13d(x,y,-1)));
				  return M.t();
			  }
			  else if(y==-1)
			  {
				   double  Ex[9]={ 0,-z,-1,
					               z,0,-x,
								   1,x,0};
				    
				  double F[9]={a,a*x+b*z,b,
					           c,c*x+d*z,d,
							   e,e*x+f*z,f};
				  Fund=Mat(3,3,CV_64F,F);
				  ex=Mat(3,3,CV_64F,Ex);
				  M=ex*Fund;
				  M.push_back(Mat(Matx13d(x,-1,z)));
				  return M.t();

			  }
			  else if(x==-1)
			  {
				   double  Ex[9]={ 0,-z,y,
					                z,0,1,
								   -y,-1,0};
				    
				  double F[9]={a*y+b*z,a,b,
					           c*y+d*z,c,d,
							   e*y+f*z,e,f};
				  Fund=Mat(3,3,CV_64F,F);
				  ex=Mat(3,3,CV_64F,Ex);
				   M=ex*Fund;
				  M.push_back(Mat(Matx13d(-1,y,z)));
				  return M.t();
			  }
			  else
				  return Mat::eye(3,4,CV_64F);
			 
		  };
		   static Mat construstFundMat(Mat cmv)
		  {
			  double * cm=cmv.ptr<double>();
			  double a=cm[0],b=cm[1],c=cm[2],
			  d=cm[3],e=cm[4],f=cm[5],
			  x=cm[6],y=cm[7],z=cm[8];
			  Mat Fund,ex,M;
			  if(z==-1)
			  {
				    
				  double F[9]={a,b,a*x+b*y,
					           c,d,c*x+d*y,
							   e,f,e*x+f*y};
				  Fund=Mat(3,3,CV_64F,F);
				  return Fund;
			  }
			  else if(y==-1)
			  {
	
				  double F[9]={a,a*x+b*z,b,
					           c,c*x+d*z,d,
							   e,e*x+f*z,f};
				  Fund=Mat(3,3,CV_64F,F);
				  return Fund;

			  }
			  else if(x==-1)
			  {
				  double F[9]={a*y+b*z,a,b,
					           c*y+d*z,c,d,
							   e*y+f*z,e,f};
				  Fund=Mat(3,3,CV_64F,F);
				  return Fund;
			  }
			 
		  };
};

class BundleAdjusterFM2: public LevMarqSparse
{
public:	BundleAdjusterFM2(std::vector<Mat>* X_,std::vector<Mat>* P_,std::vector<std::vector<std::pair<Point2f,int> > >* corresp_,
			std::vector<std::vector<std::pair<int,int> > > * ind_,std::vector<char> * fix_,int maxiter_=50,double eps_=1e-5):
	LevMarqSparse(12,X_,P_,corresp_, ind_, fix_, maxiter_,eps_)
	{
	};
	   void computeError()
  {
	  double* erptr=(double*)error.data;
	  Mat Ja,Jb;
	  double _pnt[3];
	  Mat pn(3,1,CV_64F,_pnt);
	  Mat xg;
	  for(int cam=0;cam<(*corresp).size();cam++)
	  {
		  Mat Pnew=(*P)[cam].reshape(0,3);
		  for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		  {
			  auto p=(*corresp)[cam][pnt];
			  Mat x=(*X)[p.second];
			  convertPointsToHomogeneous(x,xg);
			  xg=xg.reshape(1,4);
			  gemm(Pnew,xg,1,noArray(),0,pn,0);
			  double zinv=1/_pnt[2],xpr=_pnt[0]*zinv,ypr=_pnt[1]*zinv;
			  erptr[0]=xpr-p.first.x;
			  erptr[1]=ypr-p.first.y;
			  erptr+=2;
			  matMulDeriv(Pnew,xg,Ja,Jb);
			  Jb=Jb(Range::all(),Range(0,3));
			  if(cam==0)
				  A[cam][pnt].setTo(0);
			  else
			  {
				  addWeighted(Ja.row(0),zinv,Ja.row(2),-xpr*zinv,0,A[cam][pnt].row(0));
				  addWeighted(Ja.row(1),zinv,Ja.row(2),-ypr*zinv,0,A[cam][pnt].row(1));
			  }
			//  if(!(*isfixed)[p.second])
			  {
				  addWeighted(Jb.row(0),zinv,Jb.row(2),-xpr*zinv,0,B[cam][pnt].row(0));
				  addWeighted(Jb.row(1),zinv,Jb.row(2),-ypr*zinv,0,B[cam][pnt].row(1));
			  }
			  //else
			//	  B[cam][pnt].setTo(0);
		  }
	  }
  };
double computenewError()
{
	double* erptr=(double*)error2.data;
	Mat Ja,Jb;
	double _pnt[3];
	Mat pn(3,1,CV_64F,_pnt);
	Mat xg;
	for(int cam=0;cam<(*corresp).size();cam++)
	{
		Mat Pnew=newP[cam].reshape(0,3);
		for(int pnt=0;pnt<(*corresp)[cam].size();pnt++)
		{
			auto p=(*corresp)[cam][pnt];
			Mat x=newX[p.second];
			convertPointsToHomogeneous(x,xg);
			xg=xg.reshape(1,4);
			gemm(Pnew,xg,1,noArray(),0,pn,0);
			double zinv=1/_pnt[2],xpr=_pnt[0]*zinv,ypr=_pnt[1]*zinv;
			erptr[0]=xpr-p.first.x;
			erptr[1]=ypr-p.first.y;
			erptr+=2;			
		}
	}
	return norm(error2);
}; 


};
