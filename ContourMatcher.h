#define _USE_MATH_DEFINES
#include <vector>
#include "opencv\cv.hpp"
#include "opencv\cv.h"
#include "math.h"
#include <queue>
#include <set>
using namespace cv;

const double insertion_cost=2.0;
const double deletion_cost=2.0;
const double INF=10000000.0;

class CornersExtractor //13. Chetverikov D, Szabo Zs (1999) A simple and efficient algorithm for detection of high curvature points in
                       //planar curves
{
	std::vector<double> angles;
	std::vector<std::pair<int,int> > low;
	std::vector<std::pair<int,int> > high;
	double dmin, dmax, amax;
public:	int cap;
 CornersExtractor(double dmin_,double dmax_,double amax_,int cap_)
		{
			dmin=dmin_;
			dmax=dmax_;
			amax=amax_;
			cap=cap_;
			angles.assign(cap,0);
			low.assign(cap,std::pair<int,int>());
			high.assign(cap,std::pair<int,int>());
		}
void extract(std::vector<std::vector<Point> >& curves,std::vector<std::vector<Point> >& corners)
{
	corners.assign(curves.size(),std::vector<Point>());
	for(int i=0;i<curves.size();i++)
	{
		for(int j=1;j<int(curves[i].size())-1;j++)
		{
			angles[j]=0;
			int jminus=1,jplus=1;
			Point p=curves[i][j];
			while(j-jminus>=0&&(p-curves[i][j-jminus]).ddot(p-curves[i][j-jminus])<dmin)
			{
				jminus++;
			}
			if(j-jminus<0)
				continue;
			while(j+jplus<curves[i].size()&&(p-curves[i][j+jplus]).ddot(p-curves[i][j+jplus])<dmin)
			{
				jplus++;
			}
			if(j+jplus==curves[i].size())
				continue;
			low[j]=std::make_pair(jminus,jplus);
				int jminusm=jminus,jplusm=jplus;
			while(j-jminus>=0&&(p-curves[i][j-jminus]).ddot(p-curves[i][j-jminus])<=dmax)
			{
				jminus++;
			}

			while(j+jplus<curves[i].size()&&(p-curves[i][j+jplus]).ddot(p-curves[i][j+jplus])<=dmax)
			{
				jplus++;
			}
			
			high[j]=std::make_pair(jminus,jplus);
			
			for(int jm=jminusm;jm<jminus;jm++)
			{
				double b=(p-curves[i][j-jm]).ddot(p-curves[i][j-jm]);
				for(int jp=jplusm;jp<jplus;jp++)
				{
					double a=(p-curves[i][j+jp]).ddot(p-curves[i][j+jp]);
					double c=(curves[i][j-jm]-curves[i][j+jp]).ddot(curves[i][j-jm]-curves[i][j+jp]);
					double ang=acos(0.5*(a+b-c)/sqrt(a*b))*180/M_PI;
					if(ang<amax)
						angles[j]=max(angles[j],180-ang);
				}
			}
		}

		for(int j=1;j<int(curves[i].size())-1;j++)
			if(angles[j]>0)
		{
			bool ex=0;
			for(int k=1;k<high[j].first;k++)
				if(angles[j]<angles[j-k])
				{
					ex=1; break;	
				}
				if(ex) continue;
				for(int k=1;k<high[j].second;k++)
				if(angles[j]<angles[j+k])
				{ex=1;break;}
				if(ex) continue;
			/*for(int k=-2;k<=2;k++)
				if(j+k>=0&&j+k<curves[i].size()&&angles[j+k]>angles[j])
				{
					ex=1;
					break;
				}
				if(ex) continue;*/
				corners[i].push_back(curves[i][j]);
		}
	}
};
}; 

//class ContourMatcher
//{
//	std::vector<std::vector<double> > traincurv;
//	std::vector<std::vector<double> > querycurv;
//	std::vector<std::vector<double> > dists;
//	std::vector<std::vector<double> > temp;
//	std::vector<std::vector<double> > matches;
//	int maxscale;
//
//	double computedist(int trainind,int queryind)
//	{
//		for(int i=0;i<traincurv[trainind].size()/2;i++)
//		{
//			dists[i][0]=0;
//			dists[i+1+traincurv[trainind].size()/2][0]=1000000000;
//		}
//		for(int i=1;i<querycurv[queryind].size();i++)
//		dists[0][i]=dists[0][i-1]+deletion_cost;
//		for(int i=1;i<=querycurv[queryind].size();i++)
//		{
//			for(int j=1;j<=traincurv[trainind].size();j++)
//			{
//				dists[j][i]=10000000;
//				for(int k=1;k<=min(maxscale,i);k++)
//				{
//					temp[0][k]=temp[0][k-1]+querycurv[queryind][i-k];
//					dists[j][i]=min(dists[j][i],dists[j][i-k]+deletion_cost*k);
//				}
//				for(int k=1;k<=min(maxscale,j);k++)
//				{
//					temp[k][0]=temp[k-1][0]-traincurv[trainind][j-k];
//					dists[j][i]=min(dists[j][i],dists[j-k][i]+insertion_cost*k);
//				}
//				for(int k=1;k<=min(maxscale,i);k++)
//					for(int l=1;l<=min(maxscale,j);l++)
//					{
//						temp[l][k]=temp[l-1][k]+traincurv[trainind][j-l];
//						dists[j][i]=min(dists[j][i],dists[j-l][i-k]+abs(temp[l][k]));
//					}
//				/*double m1=dists[j-1][i-1]+abs(traincurv[trainind][j-1]-querycurv[queryind][i-1]);
//				double m2=dists[j][i-1]+deletion_cost;
//				double m3=dists[j-1][i]+insertion_cost;
//				dists[j][i]=min(m1,min(m2,m3));*/
//			}
//		}
//		double res=dists[traincurv[trainind].size()/2+1][querycurv[queryind].size()];
//		for(int i=traincurv[trainind].size()/2+2;i<traincurv[trainind].size();i++)
//			res=min(res,dists[i][querycurv[queryind].size()]);
//		return res;
//	}
//
//	
//
//
//public: ContourMatcher(std::vector<std::vector<Point> > contours,int maxscale_)
//		{
//			Point2f dp,d2p;
//			int maxlen=0;
//			traincurv.assign(contours.size(),std::vector<double>());
//			for(int i=0;i<traincurv.size();i++)
//			{
//				maxlen=std::max(maxlen,(int)contours[i].size());
//				traincurv[i].assign(2*contours[i].size(),0);
//				for(int j=0;j<contours[i].size();j++)
//				{
//					int next=(j+1)%contours[i].size();
//		int prev=(j-1+contours[i].size())%contours[i].size();
//		dp=(contours[i][next]-contours[i][prev])/2.0;
//		d2p=contours[i][next]+contours[i][prev]-2*contours[i][j];
//		if(dp.ddot(dp)>1e-12)
//		{
//			double cross=d2p.cross(dp);
//			traincurv[i][j]=cross/pow(dp.ddot(dp),1.5);
//			traincurv[i][j+contours[i].size()]=cross/pow(dp.ddot(dp),1.5);
//		}
//		else
//		{
//		traincurv[i][j]=0;
//		traincurv[i][j+contours[i].size()]=0;
//		}
//				}
//			}
//			dists.assign(2*maxlen+1,std::vector<double>());
//			temp.assign(maxscale_+1,std::vector<double>(maxscale_+1,0));
//			maxscale=maxscale_;
//		};
//		void match(std::vector<std::vector<Point> > contours,std::vector<std::pair<std::pair<int,int>,double> >& match)
//	{
//		Point2f dp,d2p;
//			int maxlen=0;
//			querycurv.assign(contours.size(),std::vector<double>());
//			for(int i=0;i<querycurv.size();i++)
//			{
//				maxlen=std::max(maxlen,(int)contours[i].size());
//				querycurv[i].assign(contours[i].size(),0);
//				for(int j=0;j<querycurv[i].size();j++)
//				{
//					int next=(j+1)%querycurv[i].size();
//		int prev=(j-1+querycurv[i].size())%querycurv[i].size();
//		dp=(contours[i][next]-contours[i][prev])/2.0;
//		d2p=contours[i][next]+contours[i][prev]-2*contours[i][j];
//		if(dp.ddot(dp)>1e-12)
//		{
//			double cross=d2p.cross(dp);
//			querycurv[i][j]=cross/pow((dp.ddot(dp)),1.5);
//		}
//		else
//		querycurv[i][j]=0;
//				}
//			}
//			for(int i=0;i<dists.size();i++)
//			dists[i].assign(maxlen+1,0);
//			matches.assign(querycurv.size(),std::vector<double>(traincurv.size(),0));
//			for(int i=0;i<querycurv.size();i++)
//				for(int j=0;j<traincurv.size();j++)
//				{
//					if(i==2&&j==22)
//					{
//						int st=9;
//					}
//					matches[i][j]=computedist(j,i)/traincurv[j].size();
//				}
//
//			std::vector<double> opt1(querycurv.size(),0);
//			std::vector<double> opt2(traincurv.size(),0);
//		 for(int i=0;i<querycurv.size();i++)
//		 {
//			 double mn=10000;
//			 int ind=-1;
//			 for(int j=0;j<traincurv.size();j++)
//				 if(matches[i][j]<mn)
//			 {
//				 mn=matches[i][j];
//				 ind=j;
//			 }
//				 opt1[i]=ind;
//		 }
//		 for(int i=0;i<traincurv.size();i++)
//		 {
//			 double mn=10000;
//			 int ind=-1;
//			 for(int j=0;j<querycurv.size();j++)
//				 if(matches[j][i]<mn)
//			 {
//				 mn=matches[j][i];
//				 ind=j;
//			 }
//				 opt2[i]=ind;
//		 }
//
//		 for(int i=0;i<traincurv.size();i++)
//			 if(opt2[i]!=-1&&opt1[opt2[i]]==i)
//			 {
//				 match.push_back(std::make_pair(std::make_pair(i,opt2[i]),matches[opt2[i]][i]));
//			 }
//
//
//	};
//};

class ContourMatcherBouagar  //http://perso.usthb.dz/~slarabi/Papers/paperbouagar2015.pdf
{
	struct descriptor
	{
		double angle,distance,length;
		bool coarse;
	};
	std::vector<std::vector<descriptor> > Descriptor[2];//0 -train 1-query
	std::vector<std::vector<double> > D;
	std::vector<std::vector<int> > coarses[2]; //indexes of corners for coarse matching
	std::vector<std::vector<double> > coarseSimilarity;
public:	std::vector<std::vector<std::vector<std::pair<int,int> > > > matches_;
	double theta0;
	uint maxgap,minmatchsize;
	void computeDescriptors(std::vector<Point>& corners,Point2d massCenter,double length,int ind,int curve)
	{
		double maxdist=0;
		for(int i=1;i<int(corners.size())-1;i++)
		{
			Point p1=corners[i]-corners[i-1];
			Point p2=corners[i]-corners[i+1];
			double coss=p1.ddot(p2)/sqrt(p1.ddot(p1)*p2.ddot(p2));
			double ori=p1.cross(p2);
			if(ori>=0)
			{
				Descriptor[ind][curve][i].angle=acos(coss)*180*M_1_PI;
				if(Descriptor[ind][curve][i].angle<theta0)
				{
				Descriptor[ind][curve][i].coarse=1;
				coarses[ind][curve].push_back(i);
				}
			else
				Descriptor[ind][curve][i].coarse=0;

			}
			else
			{
				Descriptor[ind][curve][i].angle=360-acos(coss)*180*M_1_PI;
				if(Descriptor[ind][curve][i].angle>360-theta0)
				{
				Descriptor[ind][curve][i].coarse=1;
				coarses[ind][curve].push_back(i);
				}
			else
				Descriptor[ind][curve][i].coarse=0;

			}
			Descriptor[ind][curve][i].distance=sqrt((corners[i].operator cv::Point_<double>()-massCenter).ddot(corners[i].operator cv::Point_<double>()-massCenter));
			Descriptor[ind][curve][i].length=sqrt((corners[i]-corners[i+1]).ddot(corners[i]-corners[i+1]))/length;
			maxdist=max(maxdist,Descriptor[ind][curve][i].distance);
		}

		for(int i=0;i<corners.size();i++)
			Descriptor[ind][curve][i].distance/=maxdist;

	};
	double computeDistance(bool iscoarse,int train,int curve1,int curve2,int i,int j)
	{
		if(iscoarse)
		{
			double d1=Descriptor[train][curve1][coarses[train][curve1][i]].angle-Descriptor[1-train][curve2][coarses[1-train][curve2][j]].angle;
		double d2=Descriptor[train][curve1][coarses[train][curve1][i]].distance-Descriptor[1-train][curve2][coarses[1-train][curve2][j]].distance;
		double d3=Descriptor[train][curve1][coarses[train][curve1][i]].length-Descriptor[1-train][curve2][coarses[1-train][curve2][j]].length;
		return sqrt(0.00001*d1*d1+3*d2*d2+2*d3*d3);
		}
		else
		{
		double d1=Descriptor[train][curve1][i].angle-Descriptor[1-train][curve2][j].angle;
		double d2=Descriptor[train][curve1][i].distance-Descriptor[1-train][curve2][j].distance;
		double d3=Descriptor[train][curve1][i].length-Descriptor[1-train][curve2][j].length;
		return sqrt(0.00001*d1*d1+3*d2*d2+2*d3*d3);
		}
	};
	void coarseMatching(int curve1,int curve2)
	{
	 std::vector<std::pair<int,int> > curmatch;
		int ind=0;
		int c1=curve1;
		int c2=curve2;
		if(coarses[0][curve1].size()<coarses[1][curve2].size()) //c1 longer than c2
		{
			swap(c1,c2);
			ind=1;
		}
		coarseSimilarity[curve1][curve2]=INF;
		double simil;
		for(int i=0;i<coarses[ind][c1].size();i++) //map c2 to c1 starts at position i in c1
		{
			curmatch.clear();
			simil=0;
			int j=i;
			int k=0;
			int matchsize=0;
			while(k<coarses[1-ind][c2].size()&&j<coarses[ind][c1].size()) 
			{
				double d1=INF;
				int newk=k,newj=j;                                               //find new correspondence
				for(int dk=int(matchsize>0);dk<=min(maxgap,coarses[1-ind][c2].size()-k-1);dk++) //dk and dj can be 0 only if it is the first correspondence
					for(int dj=int(matchsize>0);dj<=min(maxgap,coarses[ind][c1].size()-j-1);dj++)
						{
							double dst=computeDistance(1,ind,c1,c2,j+dj,k+dk);
							if(dst<d1)
							{
								d1=dst;
								newj=j+dj;
								newk=k+dk;
							}
						}
						if(d1<INF)
						{
						simil+=d1;
						if(!ind)
							curmatch.push_back(std::make_pair(coarses[ind][c1][newj],coarses[1-ind][c2][newk]));
						else
							curmatch.push_back(std::make_pair(coarses[1-ind][c2][newk],coarses[ind][c1][newj]));
						matchsize++;
						}
						else
						{
							newk++;  //we out of range
							newj++;
						}
						k=newk;
						j=newj;
			}
			if(matchsize>minmatchsize)
			{
			simil/=matchsize;
			if(simil<coarseSimilarity[curve1][curve2])
			{
				coarseSimilarity[curve1][curve2]=simil;
				matches_[curve1][curve2]=std::vector<std::pair<int,int> >(curmatch);
			}
			}

			curmatch.clear();//map c2 in inverse direction
			 j=i;
			 k=0;
			 simil=0;
			 matchsize=0;
			while(k<coarses[1-ind][c2].size()&&j>=0)
			{
				double d1=INF;
				int newk=k,newj=j;
				for(int dk=int(matchsize>0);dk<=min(maxgap,coarses[1-ind][c2].size()-k-1);dk++)
					for(int dj=int(matchsize>0);dj<=min(maxgap,uint(j));dj++)
						{
							double dst=computeDistance(1,ind,c1,c2,j-dj,k+dk);
							if(dst<d1)
							{
								d1=dst;
								newj=j-dj;
								newk=k+dk;
							}
						}
						if(d1<INF)
						{
						simil+=d1;
						if(!ind)
							curmatch.push_back(std::make_pair(coarses[ind][c1][newj],coarses[1-ind][c2][newk]));
						else
							curmatch.push_back(std::make_pair(coarses[1-ind][c2][newk],coarses[ind][c1][newj]));
						}
						else
						{
							newk++;
							newj--;
							matchsize--;
						}
						k=newk;
						j=newj;
						matchsize++;
			}
			if(matchsize>minmatchsize)
			{
				simil/=matchsize;
			if(simil<coarseSimilarity[curve1][curve2])
			{
				coarseSimilarity[curve1][curve2]=simil;
				matches_[curve1][curve2]=std::vector<std::pair<int,int> >(curmatch);
			}
			}




		}
	};
public:ContourMatcherBouagar(std::vector<std::vector<Point> >&traincurve,std::vector<std::vector<Point> > corners,double theta0_,int minmatchsize_,int maxgap_)
	   {
		   theta0=theta0_;
		   minmatchsize=minmatchsize_;
		   maxgap=maxgap_;
		   Descriptor[0].assign(traincurve.size(),std::vector<descriptor>());
		   coarses[0].assign(traincurve.size(),std::vector<int>());
		   for(int i=0;i<Descriptor[0].size();i++)
			   Descriptor[0][i].assign(corners[i].size(),descriptor());
		   for(int i=0;i<traincurve.size();i++)
		   {
			   double len=arcLength(traincurve[i],0);
			   auto Mom=moments(traincurve[i]);
			   Point2d c(Mom.m10/Mom.m00,Mom.m01/Mom.m00);
			   computeDescriptors(corners[i],c,len,0,i);
		   }
	   };
	   void precomputeQuery(std::vector<std::vector<Point> >&querycurve,std::vector<std::vector<Point> > corners)
	   {
		   coarses[1].assign(querycurve.size(),std::vector<int>());
		   Descriptor[1].assign(querycurve.size(),std::vector<descriptor>());
		   		   for(int i=0;i<Descriptor[1].size();i++)
			   Descriptor[1][i].assign(corners[i].size(),descriptor());
		   for(int i=0;i<querycurve.size();i++)
		   {
			   double len=arcLength(querycurve[i],0);
			   auto Mom=moments(querycurve[i]);
			   Point2d c(Mom.m10/Mom.m00,Mom.m01/Mom.m00);
			   computeDescriptors(corners[i],c,len,1,i);
		   }
		   coarseSimilarity.assign(Descriptor[0].size(),std::vector<double>(Descriptor[1].size(),100000));
		   matches_.assign(Descriptor[0].size(),std::vector<std::vector<std::pair<int,int> > >(Descriptor[1].size(),std::vector<std::pair<int,int> >())); 
	   };
	   void match(std::vector<std::vector<char> >& matches)
	   {
		   for(int i=0;i<Descriptor[0].size();i++)
			   for(int j=0;j<Descriptor[1].size();j++)
				   coarseMatching(i,j);

		   for(int i=0;i<Descriptor[0].size();i++)
		   {
			   double mn=INF,mx=0;
			   for(int j=0;j<Descriptor[1].size();j++)
			   {
				   mn=min(mn,coarseSimilarity[i][j]);
				   if(coarseSimilarity[i][j]<INF)
				   mx=max(mx,coarseSimilarity[i][j]);
			   }
			   double thresh=mn+(mx-mn)*0.25;
			   for(int j=0;j<Descriptor[1].size();j++)
				   if(coarseSimilarity[i][j]<=thresh)
			   {
				   matches[i][j]=1;
			   }
		   }
	   }

};