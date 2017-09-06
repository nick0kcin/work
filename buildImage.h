#include <opencv\cv.hpp>
#include <vector>

using namespace cv;

struct Triangle
{
	int wp1,wp2,wp3;
	Triangle(int wp1_,int wp2_,int wp3_)
	{
		wp1=wp1_;
		wp2=wp2_;
		wp3=wp3_;
	};
};

bool pointInsideTriangle(Point2f p1,Point2f p2,Point2f p3,Point2f p)
{
	double d1=(p2-p1).cross(p-p2);
	double d2=(p3-p2).cross(p-p3);
	double d3=(p1-p3).cross(p-p1);
	return (d1>=0&&d2>=0&&d3>=0)||(d1<=0&&d2<=0&&d3<=0);
}

void buildMesh(std::vector<Point2f> points,std::vector<Triangle>& triangles)
{
	std::vector<int> qLine;
	Mat mask(points.size(),1,CV_8U);
	mask.setTo(1);
		float mx=10000,my=10000;
			int ind=-1;
			bool* maskptr=mask.ptr<bool>();
			for(int i=0;i<points.size();i++)
				if(maskptr[i])
			{
				if(points[i].x==mx&&points[i].y<=my)
					{
						my=points[i].y;
						ind=i;
					}
				else
				if(points[i].x<mx)
					{
						mx=points[i].x;
						my=10000;
						ind =i;
					}
			}
				float mindist=1000000;
				int ind2=-1;
				Point2f pp=points[ind];
				for(int i=0;i<points.size();i++)
					if(i!=ind&&maskptr[i])
					{
						float dst=(pp-points[i]).ddot(pp-points[i]);
						if(mindist>dst)
						{
							mindist=dst;
							ind2=i;
						}
					}
					qLine.clear();
					qLine.push_back(ind);
					qLine.push_back(ind2);
					maskptr[ind2]=0;
					maskptr[ind]=0;
					while(1)
					//for(int j=0;j<81;j++)
					{
	/*					if(j==50)
						{
							int stop=0;
						}*/
						double mind=100000;
						int cur=-1;
						for(int i=0;i<points.size();i++)
							if(maskptr[i])
						{
							double dst=1000000;
							Point2f pp=points[i];
						for(auto it=qLine.begin();it!=qLine.end();it++)
							dst=min(dst,norm(points[*it]-pp));
						if(dst==0)
						{
							maskptr[i]=0;
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
								Point2f p=points[cur];
								bool c=0;
								auto st=qLine.begin();
								auto en=qLine.begin();
								for(auto it=qLine.begin();it!=qLine.end();)
								{
									auto it2=it;
									it2++;
									if(it2==qLine.end())
										it2=qLine.begin();
									auto np=points[*it2]-points[*it];
									auto n2=p-points[*it2];
									double cr=np.cross(n2);
									double dot=np.ddot(n2);
									if(cr>1e-06||(abs(cr)<1e-06&&dot>0))
									{
										if(!c)
										{
											st=it;
											//qLine.insert(st,std::make_pair(points[1-n][cur].pt,mask[1-n][cur]));
											c=1;
										}
										en=it;
										triangles.push_back(Triangle(*it,*it2,cur));
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
								rem++;
								qLine.insert(rem,cur);
								}
								maskptr[cur]=0;
							}
							else
								break;
					}

					//Mat iml(1000,1000,CV_8UC3);

					//for(int i=0;i<triangles.size();i++)
					//{
					//	line(iml,points[triangles[i].wp1],points[triangles[i].wp2],CV_RGB(255,0,0));
					//	line(iml,points[triangles[i].wp2],points[triangles[i].wp3],CV_RGB(255,0,0));
					//	line(iml,points[triangles[i].wp1],points[triangles[i].wp3],CV_RGB(255,0,0));
					//	circle(iml,points[triangles[i].wp1],4,CV_RGB(0,255,0));
					//	circle(iml,points[triangles[i].wp2],4,CV_RGB(0,255,0));
					//	circle(iml,points[triangles[i].wp3],4,CV_RGB(0,255,0));
					//}

					//imshow("omnomnom",iml);
					//waitKey(100000);
}


bool isBlack(uchar* p)
{
	return *p==0&&*(p+1)==0&&*(p+2)==0;
}
bool isBlack(Mat& im,Point p)
{
	uchar* pt=im.ptr<uchar>(p.y,p.x);
	return isBlack(pt);
}
void projectImage(Mat& Pano,Mat & im,std::vector<Point2f>& points,std::vector<Triangle>& triangles,std::vector<Mat>& world,Mat cam)
{
	std::vector<Point2f> newpoints;
	std::vector<Point2d> pp;
	for(int i=0;i<world.size();i++)
	{
		Mat p=cam*world[i];
		if(p.at<double>(2)<0)
		{
			double dsad=p.at<double>(2);
			int ss=0;
		}
		convertPointsFromHomogeneous(p.t(),pp);
		newpoints.push_back(pp[0]);
	}
	std::vector<Point2f> np(3,Point2f());
	for(int i=0;i<triangles.size();i++)
	{
		auto tr=triangles[i];
		np[0]=newpoints[tr.wp1];
		np[1]=newpoints[tr.wp2];
		np[2]=newpoints[tr.wp3];
		//line(Pano,np[0],np[1],CV_RGB(255,0,0));
		//line(Pano,np[1],np[2],CV_RGB(255,0,0));
		//line(Pano,np[0],np[2],CV_RGB(255,0,0));
	//	if(isBlack(Pano,np[0])||isBlack(Pano,np[1])||isBlack(Pano,np[2]))
		{
			double l=norm(np[0]-np[1]);
			Point2f p1=points[tr.wp1];
			Point2f p2=points[tr.wp2];
			Point2f p3=points[tr.wp3];
			auto v1=(p1-p2),v2=(p3-p2);
			auto nv1=(np[0]-np[1]),nv2=(np[2]-np[1]);
			double del=max(0.5/l,0.0001);
			for(double d1=0;d1<=1;d1+=del)
				for(double d2=0;d2<=1-d1;d2+=del)
				{
					Point2f p=p2+d1*v1+d2*v2;
					Point2f newp=np[1]+ d1*nv1+d2*nv2;
					if(newp.inside(Rect_<float>(0,0,Pano.cols,Pano.rows)))
					{
						uchar* ptrr=Pano.ptr<uchar>(newp.y,newp.x);
						if(!p.inside(Rect_<float>(0,0,im.cols,im.rows)))
						{
							int dgd=0;
						}
						uchar* ptr2=im.ptr<uchar>(p.y,p.x);
						if(isBlack(ptrr))
							for(int v=0;v<3;v++)
							{
								*ptrr=*ptr2;
								ptrr++;ptr2++;
							}
					}
				}
		}
	}
};