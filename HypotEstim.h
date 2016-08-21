#include "vector"
#include "opencv\cv.hpp"

using namespace cv;

void composeProjectionMatrix(Mat Rvec,Mat t,Mat K,Mat&P)
{
	Mat R;
	Rodrigues(Rvec,R);
	P=R.t();
	P.push_back(t.t());
	gemm(K,P,1,noArray(),1,P,GEMM_2_T);
}
Mat composeProjectionMatrix(Mat cam)
{
	Mat R;
	Rodrigues(cam.rowRange(0,3),R);
	Mat P=R.t();
	P.push_back(cam.rowRange(3,6).t());
	Mat K(Matx33d(cam.at<double>(6),0,cam.at<double>(8),
				  0,cam.at<double>(7),cam.at<double>(9),
				  0,0,1));
	gemm(K,P,1,noArray(),1,P,GEMM_2_T);
	return P;
}

Mat findFundMat(std::vector<Point2f> points1,std::vector<Point2f> points2,Mat& mask,double par1,Rect2f imrect)
{
	std::vector<Point2f> p1,p2;
	std::vector<int> ind;
	bool* maskptr=mask.ptr<bool>();
	for(int i=0;i<mask.rows;i++)
		if(maskptr[i]&&points2[i].inside(imrect))
		{
			p1.push_back(points1[i]);
			p2.push_back(points2[i]);
			ind.push_back(i);
		}
		Mat mask2;
		Mat F=findFundamentalMat(p1,p2,mask2,8,par1);
		bool* mask2ptr=mask2.ptr<bool>();
		mask.setTo(0);
		for(int i=0;i<mask2.rows;i++)
				maskptr[ind[i]]=mask2ptr[i];
		return F;
}


void filterMatches(std::vector<KeyPoint>& points_query,std::vector<KeyPoint>& points_train,std::vector<DMatch> &matches12,std::vector<DMatch> &matches21,
				   std::vector<DMatch>& matches)
{
	matches.clear();
		std::vector<Point2f> p1,p2;
		std::vector<int > ind;
		float mindist=1000000;
		for(int i=0;i<matches12.size();i++)
			mindist=min(mindist,matches12[i].distance);
	for(int i=0;i<matches12.size();i++)
	{
		//std::lower_bound(matches21.begin(),matches21.end(),matches12[i].queryIdx);
		if(matches21[matches12[i].trainIdx].trainIdx==i&&matches12[i].distance<5*mindist)
		{
		p1.push_back(points_query[matches12[i].queryIdx].pt);
		p2.push_back(points_train[matches12[i].trainIdx].pt);
		ind.push_back(i);
		}
	}
	Mat mask;
	findFundamentalMat(p1,p2,mask,8,1.0);
	int m=countNonZero(mask);
	bool* maskptr=mask.ptr<bool>();
	for(int i=mask.rows-1;i>=0;i--)
		if(maskptr[i])
			matches.push_back(matches12[ind[i]]);
}



Mat P2EHomo(Mat P)
{
	Mat R=P(Range(0,3),Range(0,3)).inv();
	Mat t=-R*P.col(3);
	R=R.t();
	R.push_back(t.t());
	R=R.t();
	R.push_back(Mat(Matx14d(0,0,0,1)));
	return R;
}

Mat  computeElation(Mat P1,Mat P2)//P1=P2*R
{
	Mat R=Mat::eye(4,4,CV_64F);
	double k=norm(P1.col(3))/norm(P2.col(3));
	std::cout<<norm(P1.col(3))<<' '<<norm(P2.col(3))<<"\n";
	if(P1.col(3).dot(P2.col(3))<0)
		k=-k;
	//Mat F=P1(Range(0,3),Range(0,3))-P2(Range(0,3),Range(0,3));
	//std::cout<<F<<"\n";
	//for(int i=0;i<3;i++)
	//	F.row(i)/=P2.col(3).at<double>(i);
	//std::cout<<F<<"\n";
	//Mat v= (F.row(0)+F.row(1)+F.row(2))/3;
	//for(int i=0;i<3;i++)
	//	R.at<double>(3,i)=v.at<double>(i);
	R.at<double>(3,3)=k;
	return R;
}

Mat isotropicScaleNormalization(std::vector<Point2f> p,Mat& res)
{
	Point2f meanv(0,0);
	for(int i=0;i<p.size();i++)
		meanv+=p[i];
	meanv/=(float)p.size();
	float scale=0;
	for(int i=0;i<p.size();i++)
		scale+=sqrt((meanv-p[i]).ddot(meanv-p[i]));
	scale/=p.size();
	scale=sqrt(2.0)/scale;
	Mat trans(Matx33d(scale,0,-scale*meanv.x,
		              0,scale,-scale*meanv.y,
		              0,0,1));
	double* r1=res.ptr<double>(0,0);
	double* r2=res.ptr<double>(1,0);
	double* r3=res.ptr<double>(2,0);
	for(int i=0;i<p.size();i++)
	{
		*r1=scale*(p[i]-meanv).x;
		*r2=scale*(p[i]-meanv).y;
		*r3=1;
		r1++,r2++,r3++;
	}
		
	return trans;

};


void factorisation(std::vector<std::vector<Point2f> > pnts,std::vector<Mat>& cams,std::vector<Mat> &world)
{
	Mat W(pnts.size()*3,pnts[0].size(),CV_64F);
	std::vector<Mat> trns;
	for(int i=0;i<pnts.size();i++)
	{
				Mat cp1=W(Range(i*3,(i+1)*3),Range::all());
				trns.push_back(isotropicScaleNormalization(pnts[i],cp1));
	}
	Mat U,V,D;
	Mat WW=W.clone();
	for(int iter=0;iter<5;iter++)
	{
	SVDecomp(WW,D,U,V);
	std::cout<<D<<"\n";
	for(int i=5;i<D.rows;i++)
		D.at<double>(i)=0;
	Mat Dd(D.rows,D.rows,CV_64F);
	Dd.setTo(0);
	for(int i=0;i<D.rows;i++)
		Dd.at<double>(i,i)=D.at<double>(i);
	Mat P=U*Dd;
	for(int i=0;i<cams.size();i++)
		cams[i]=trns[i].inv()*P(Range(i*3,(i+1)*3),Range(0,4)).clone();
	for(int i=0;i<world.size();i++)
		world[i]=V(Range(0,4),Range(i,i+1)).clone();
	for(int i=0;i<cams.size();i++)
	{
		Mat row=cams[i].row(2);
		for(int j=0;j<world.size();j++)
		{
			double lambda=row.dot(world[j].t());
			for(int k=0;k<3;k++)
				WW.at<double>(i*cams.size()+k,j)=W.at<double>(i*cams.size()+k,j)*lambda;
		}
	}
	}
};


void afiineFactorisation(std::vector<std::vector<Point2f> > pnts,std::vector<Mat>& cams,std::vector<Mat> &world)
{
	Mat W(pnts.size()*2,pnts[0].size(),CV_64F);
	std::vector<Point2f>meanv(pnts.size(),Point2f(0,0));
	for(int i=0;i<pnts.size();i++)
	{
		for(int j=0;j<pnts[i].size();j++)
			meanv[i]+=pnts[i][j];
		meanv[i]/=(float)pnts[i].size();
		for(int j=0;j<pnts[i].size();j++)
		{
			W.at<double>(i*2,j)=pnts[i][j].x-meanv[i].x;
			W.at<double>(i*2+1,j)=pnts[i][j].y-meanv[i].y;
	}
	}
		Mat U,V,D;
		SVDecomp(W,D,U,V);
		Mat M=U.colRange(0,4);
		for(int i=0;i<3;i++)
			M.col(i)*=D.at<double>(i);
		for(int i=0;i<cams.size();i++)
		{
			cams[i]=M.rowRange(i*2,i*2+2).clone();
			cams[i].at<double>(0,3)=meanv[i].x;
			cams[i].at<double>(1,3)=meanv[i].y;
		}
		Mat X=V.rowRange(0,4);
		for(int i=0;i<world.size();i++)
		{
			world[i]=X.col(i).clone();
			world[i].at<double>(3)=1;
		}
}
Mat findPureMotionFundMat( std::vector<Point2f> x1,std::vector<Point2f> x2,Mat mask=Mat()) //translate
{
	int sz=mask.empty()?x1.size():countNonZero(mask);
	bool b=mask.empty();
	Mat A(sz,3,CV_64F);
	int k=0;
	for(int i=0;i<x1.size();i++)
		if(b||mask.at<bool>(i)==1)
	{
		A.at<double>(k,0)=x1[k].x*x2[k].y-x2[k].x*x1[k].y;
		A.at<double>(k,1)=x1[k].x-x2[k].x;
		A.at<double>(k,2)=x1[k].y-x2[k].y;
		k++;
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

class Ransac
{
	double thresh_,percent_;
	int maxiter_;
protected:
	int minsamples_;
	std::vector<Point2f> x1,x2;
	Mat* mask_;
public:
	Ransac(double thresh,double percent,int maxiter,Mat *mask)
		{
			thresh_=thresh;
			percent_=percent;
			maxiter_=maxiter;
			mask_=mask;
	};
	virtual Mat estimator(std::vector<Point2f>&,std::vector<Point2f>&,Mat*)=0;
	virtual double error(Point2f,Point2f,Mat)=0;
	Mat doJob(std::vector<Point2f>& p1,std::vector<Point2f>& p2)
		{
	Mat res;
	int maxinl=0;
	if(p1.size()<minsamples_)
		return Mat();
	(*mask_)=Mat(p1.size(),1,CV_16U);
	for(int iter=0;iter<maxiter_;iter++)
	{
		int inliers=0;
		mask_->setTo(0);
		for(int i=0;i<minsamples_;i++)
		{
			int k=randu<int>()%p1.size();
			while(mask_->at<short>(k)==1)
				k=randu<int>()%p1.size();
			mask_->at<short>(k)=1;
		}
		Mat model=estimator(p1,p2,mask_);
		for(int i=0;i<p1.size();i++)
		{
			if(error(p1[i],p2[i],model)<thresh_)
				 {
					 inliers++;
					 mask_->at<short>(i)=1;
				 }
		}
		if(inliers>maxinl)
		{
			maxinl=inliers;
			res=estimator(p1,p2,mask_);
		}
		if(inliers>=percent_*p1.size())
		{
			return estimator(p1,p2,mask_);
		}
	}
	return res;
}
	
};


class PureMotionFundRansac : public Ransac
{
public:	PureMotionFundRansac(double thresh,double percent,int maxiter,Mat* mask):Ransac(thresh,percent,maxiter,mask)
	{
		//__super::Ransac(thresh,percent,maxiter);
		minsamples_=2;
		x1.assign(3,Point2f());
		x2.assign(3,Point2f());
	}
	Mat estimator(std::vector<Point2f>& x1 ,std::vector<Point2f>& x2,Mat* mask)
	{
	int sz=mask==0?x1.size():countNonZero((*mask));
	bool b=mask==0;
	Mat A(sz,3,CV_64F);
	int k=0;
	for(int i=0;i<x1.size();i++)
		if(b||mask->at<short>(i)==1)
	{
		A.at<double>(k,0)=x1[i].x*x2[i].y-x2[i].x*x1[i].y;
		A.at<double>(k,1)=x1[i].x-x2[i].x;
		A.at<double>(k,2)=x1[i].y-x2[i].y;
		k++;
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
};
	double error( Point2f m1,Point2f m2, Mat Fund )
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
};


class EuclidRansac :public Ransac
{
	double cx,cy;
public:	EuclidRansac(double thresh,double percent,int maxiter,Mat* mask,double cx_,double cy_):Ransac(thresh,percent,maxiter,mask)
	{
		//__super::Ransac(thresh,percent,maxiter);
		minsamples_=2;
		x1.assign(3,Point2f());
		x2.assign(3,Point2f());
		cx=cx_;
		cy=cy_;
	}
	Mat estimator(std::vector<Point2f>& p1 ,std::vector<Point2f>& p2,Mat* mask)
{
	int sz=mask==0?x1.size():countNonZero((*mask));
	bool bl=mask==0;
	Mat A(sz*2,2,CV_64F);
	Mat b(sz*2,1,CV_64F);
	A.setTo(0);
	int i=0;
	for(int k=0;k<A.rows/2;k++)
		if(bl||mask->at<short>(k))
	{
		A.at<double>(2*i,0)=p1[i].x-cx;
		A.at<double>(2*i,1)=p1[i].y-cy;
		//A.at<double>(2*i,2)=1;
		A.at<double>(2*i+1,0)=p1[i].y-cy;
		A.at<double>(2*i+1,1)=-(p1[i].x-cx);
		//A.at<double>(2*i+1,3)=1;
		b.at<double>(2*i)=p2[i].x-cx;
		b.at<double>(2*i+1)=p2[i].y-cy;
		i++;
	}
	Mat C=A.t()*A;
	Mat d=A.t()*b;
	Mat R=C.inv()*d;
	Matx23d res(R.at<double>(0),R.at<double>(1),cx-R.at<double>(0)*cx-R.at<double>(1)*cy,
		-R.at<double>(1),R.at<double>(0),cy+R.at<double>(1)*cx-R.at<double>(0)*cy);
	return Mat(res);
};
	/*double error( Point2f m1,Point2f m2, Mat M )
	{
		double tx,ty;
		double* m=M.ptr<double>();
		tx=m2.x-m[0]*m1.x-m[1]*m1.y;
		ty=m2.y-m[3]*m1.x-m[4]*m1.y;
		double x=m[2],y=m[5];
		double l1=sqrt(x*x+y*y);
		double l2=sqrt(tx*tx+ty*ty);
		return 2*abs(tx*y-ty*x)/l1/l2;
	};*/
	double error( Point2f m1,Point2f m2, Mat M )
	{
		return 0;
		/*double tx,ty;
		double* m=M.ptr<double>();
		tx=m2.x-cx-m[0]*(m1.x-cx)-m[1]*(m1.y-cy);
		ty=m2.y-cy-m[3]*(m1.x-cx)-m[4]*(m1.y-cy);
		double x=m[2],y=m[5];
		double l1=sqrt(x*x+y*y);
		double l2=sqrt(tx*tx+ty*ty);
		return 2*abs(tx*y-ty*x)/l1/l2;*/
	};
};


class planarRotationRansac:public Ransac
{
public:
planarRotationRansac(double thresh,double percent,int maxiter,Mat* mask):Ransac(thresh,percent,maxiter,mask)
					 {
		minsamples_=7;
		x1.assign(7,Point2f());
		x2.assign(7,Point2f());
					 };

	Mat estimator(std::vector<Point2f>& x1 ,std::vector<Point2f>& x2,Mat* mask)
	{
	int sz=mask==0?x1.size():countNonZero((*mask));
	bool b=mask==0;
	Mat A(sz,7,CV_64F);
	int k=0;
	for(int i=0;i<x1.size();i++)
		if(b||mask->at<short>(i)==1)
	{
		A.at<double>(k,0)=x1[i].x*x2[i].x+x2[i].y*x1[i].y;
		A.at<double>(k,1)=-x1[i].x*x2[i].y+x2[i].x*x1[i].y;
		A.at<double>(k,2)=x2[i].x;
		A.at<double>(k,3)=x2[i].y;
		A.at<double>(k,4)=x1[i].x;
		A.at<double>(k,5)=x1[i].y;
		A.at<double>(k,6)=1;
		k++;
	}
	Mat ff=SVD(A.t()*A).vt.row(6);
	//std::cout<<A*ff.t()<<"\n";
	double* f=ff.ptr<double>();
	double FF[9]={f[0],f[1],f[2],
		          -f[1],f[0],f[3],
				  f[4],f[5],f[6]};
	Mat F(3,3,CV_64F,FF);
	//std::cout<<F<<"\n";
	//for(int i=0;i<x1.size();i++)
	//{
	//	std::cout<<error(x1[i],x2[i],F)<<" "<<error(x2[i],x1[i],F)<<"\n";
	//	}
	return F.clone();
};

	double error( Point2f m1,Point2f m2, Mat Fund )
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
static	void estimateMotion(Mat F,Mat& R,Mat& t)
{
						double det=sqrt(determinant(F(Range(0,2),Range(0,2))));
						double a=F.at<double>(1,0)/det,b=F.at<double>(0,0)/det;
						Mat qw(Matx22d(-b,-a,a,-b));
						Mat we(Matx21d(F.at<double>(2,0),F.at<double>(2,1)));
						Mat r=qw.inv()*we;
						R=Mat(Matx33d(a,b,0,
							      -b,a,0,
								   0             , 0            ,1));
						t=Mat(Matx31d(r.at<double>(0),r.at<double>(1),det));
						t/=norm(t);
};
};

class planarRansac:public Ransac
{
public:
planarRansac(double thresh,double percent,int maxiter,Mat* mask):Ransac(thresh,percent,maxiter,mask)
					 {
		minsamples_=7;
		x1.assign(5,Point2f());
		x2.assign(5,Point2f());
					 };

	Mat estimator(std::vector<Point2f>& x1 ,std::vector<Point2f>& x2,Mat* mask)
	{
	int sz=mask==0?x1.size():countNonZero((*mask));
	bool b=mask==0;
	Mat A(sz,5,CV_64F);
	int k=0;
	for(int i=0;i<x1.size();i++)
		if(b||mask->at<short>(i)==1)
	{
		A.at<double>(k,0)=x2[i].x;
		A.at<double>(k,1)=x2[i].y;
		A.at<double>(k,2)=x1[i].x;
		A.at<double>(k,3)=x1[i].y;
		A.at<double>(k,4)=1;
		k++;
	}
		//std::cout<<SVD(A).w<<"\n";
	Mat ff=SVD(A.t()*A).vt.row(4);
	//std::cout<<A*ff.t()<<"\n";
	double* f=ff.ptr<double>();
	double FF[9]={0,0,f[0],
		          0,0,f[1],
				  f[2],f[3],f[4]};
	Mat F(3,3,CV_64F,FF);
	//std::cout<<F<<"\n";
	//for(int i=0;i<x1.size();i++)
	//{
	//	std::cout<<error(x1[i],x2[i],F)<<" "<<error(x2[i],x1[i],F)<<"\n";
	//	}
	return F.clone();
};

	double error( Point2f m1,Point2f m2, Mat Fund )
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
	static void estimateMotion(Mat F,Mat& R,Mat& t)
	{
		std::cout<<SVD(F).u.col(2)<<"\n"<<SVD(F).u.col(2).t()*F<<"\n";
						Mat qw(Matx22d(-F.at<double>(0,2),F.at<double>(1,2),-F.at<double>(1,2),-F.at<double>(0,2)));
						Mat we(Matx21d(F.at<double>(2,0),F.at<double>(2,1)));
						Mat r=qw.inv()*we;
						r/=norm(r);
						R=Mat(Matx33d(r.at<double>(0),r.at<double>(1),0,
							      -r.at<double>(1),r.at<double>(0),0,
								   0             , 0            ,1));
						t=Mat(Matx31d(-F.at<double>(1,2),F.at<double>(0,2),0));
						t/=norm(t);
	};
};

class PlanarMotionFund : public cv::MinProblemSolver::Function
{
	std::vector<Mat> p1,p2;
public:
	PlanarMotionFund(std::vector<Point2f> x1,std::vector<Point2f> x2)
	{
		p1.assign(x1.size(),Mat());
		p2.assign(x2.size(),Mat());
		for(int i=0;i<x1.size();i++)
		{
			p1[i]=Mat(Matx31d(x1[i].x,x1[i].y,1));
			p2[i]=Mat(Matx31d(x2[i].x,x2[i].y,1));
		}
	}
	double calc(const double* x) const
	{
		double res=0;
		Mat Ex=Mat(Matx33d(0,-x[2],x[1],
			             x[2], 0, -x[0],
					   - x[1], x[0] ,0));
		Mat Estrx=Mat(Matx33d(0,-x[5],x[4],
			             x[5], 0, -x[3],
					   - x[4], x[3] ,0));
		Mat lx=Mat(Matx33d(0,-x[8],x[7],
			             x[8], 0, -x[6],
					   - x[7], x[6] ,0));
		Mat F=Estrx*lx*Ex;
		for(int i=0;i<p1.size();i++)
		{
			Mat l1=F*p1[i],l2=p2[i].t()*F;
			double* l=l1.ptr<double>();
			double* ll=l2.ptr<double>();
			double d=p2[i].dot(F*p1[i]);
			d*=d;
			d/=l[0]*l[0]+l[1]*l[1]+ll[0]*ll[0]+ll[1]*ll[1];
			res+=d;
		}
		return res;
	};
	int getDims() const
	{
		return 9;
	};
};

//class PlanarMotionFundRansac : public Ransac
//{
//	cv::Ptr<MinProblemSolver::Function> func;
//	public:	PlanarMotionFundRansac(double thresh,double percent,int maxiter,Mat* mask):Ransac(thresh,percent,maxiter,mask)
//	{
//		//__super::Ransac(thresh,percent,maxiter);
//		minsamples_=6;
//		x1.assign(6,Point2f());
//		x2.assign(6,Point2f());
//		func=cv::Ptr<MinProblemSolver::Function>(new PlanalMotionFund(
//	};
//	Mat estimator(std::vector<Point2f>& x1 ,std::vector<Point2f>& x2,Mat* mask)
//	{
//		func=cv::Ptr<MinProblemSolver::Function>(new PlanalMotionFund(x1,x2));
//		DownhillSolver::create(
//	}
//}


class MotionFund : public cv::MinProblemSolver::Function
{
	std::vector<Mat> p1,p2;
public:
	MotionFund(std::vector<Point2f> x1,std::vector<Point2f> x2)
	{
		p1.assign(x1.size(),Mat());
		p2.assign(x2.size(),Mat());
		for(int i=0;i<x1.size();i++)
		{
			p1[i]=Mat(Matx31d(x1[i].x,x1[i].y,1));
			p2[i]=Mat(Matx31d(x2[i].x,x2[i].y,1));
		}
	}
	double calc(const double* v) const
	{
		double res=0;
		double a=v[0],b=v[1],c=v[2],d=v[3],x=v[4],y=v[5],xs=v[6],ys=v[7];
		Mat F=Mat(Matx33d(b,           a,     -a*y-b*x,
		                  -d,         -c,      c*y+d*x,
						  d*ys-b*xs, c*ys-a*xs, ys*(-c*y-d*x)+xs*(a*y+b*x) ));
		for(int i=0;i<p1.size();i++)
		{
			Mat l1=F*p1[i],l2=p2[i].t()*F;
			double* l=l1.ptr<double>();
			double* ll=l2.ptr<double>();
			double d=p2[i].dot(F*p1[i]);
			d*=d;
			d/=l[0]*l[0]+l[1]*l[1]+ll[0]*ll[0]+ll[1]*ll[1];
			res+=d;
		}
		return res;
	};
	int getDims() const
	{
		return 8;
	};
};