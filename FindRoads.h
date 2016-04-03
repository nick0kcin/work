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
#include "genhough.h"
#include <queue>
#include <map>


#define At(M,i,j) (M->data+(i)*M->step[0]+(j)*M->step[1])
#define At4(M,i,j,k,m) (M->data+(i)*M->step[0]+(j)*M->step[1]+(k)*M->step[2]+(m)*M->step[3])

#define Att(M,t,i,j) ((t*)(M->data+(i)*M->step[0]+(j)*M->step[1]))


using namespace cv;

unsigned char roadpixel[3]={192,124,126};

double Euclid3Square(unsigned char* p,unsigned char* q)
{
	double res=0;
	for(int i=0;i<3;i++)
		res+=(*(q+i)-*(p+i))*(*(q+i)-*(p+i));
	return res;
}

void bfs(Mat* im,unsigned char* col,Point p0,Mat* out,double (*dist)(unsigned char*,unsigned char*),double v,unsigned char& val)
{
	if(*At(out,p0.x,p0.y)==0)
	{
	std::queue<Point>* q=new std::queue<Point>();
	q->push(p0);
	*At(out,p0.x,p0.y)=val;
	while (!q->empty())
	{
		Point p=q->front();
		q->pop();
		bool b=0;
		for(int x=-1;x<=1;x++)
			for(int y=-1;y<=1;y++)
				if((x!=0||y!=0)&&(p+Point(x,y)).inside(Rect(0,0,im->rows,im->cols))&&*At(out,p.x+x,p.y+y)==0)
					if(dist(col,At(im,p.x+x,p.y+y))<v)
					{
						q->push(p+Point(x,y));
						*At(out,p.x+x,p.y+y)=val;
						b=1;
					}
					//if(!b)
					//	*At(out,p.x,p.y)=val+1;
	}
	val+=5;
	}
}





Mat FindRoads(Mat& map,double coeff1=800,double coeff2=850)
{
	Mat maplab;
	cvtColor(map,maplab,CV_RGB2Lab);// перевод в цветовую схему лаб,чтобы оценивать цветовое отличие
	Mat out(map.rows,map.cols,CV_8UC1);
	Mat out2(map.rows,map.cols,CV_8UC1);
	out.setTo(0);
	out2.setTo(0);
	for(int i=0;i<map.rows;i++)
		for(int j=0;j<map.cols;j++)
			if(Euclid3Square(At((&maplab),i,j),roadpixel)<coeff1)   //выбираем пиксели -кандидаты на то,чтобы быть дорогой
				*At((&out),i,j)=255;
	erode(out,out,getStructuringElement(cv::MorphShapes::MORPH_RECT,Size(3,3)),Point(-1,-1),2); // подавляем малые группы пикселей
	//dilate(out,out,getStructuringElement(cv::MorphShapes::MORPH_RECT,Size(3,3)),Point(-1,-1),1);
	unsigned char val=25;
	for(int i=0;i<map.rows;i++)
		for(int j=0;j<map.cols;j++)
			if(*At((&out),i,j)==255)                                                      // из каждого кандидата запускаем поиск в ширину
				bfs(&maplab,At((&maplab),i,j),Point(i,j),&out2,Euclid3Square,coeff2,val); // те пиксели,куда смогли добраться -дорога
	                                                                                      // точки до которыхдобрались из разных начальных помечаем разным цветом

			   std::set<char> neibours[256];   //находим области,которые целиком внутри других областей
		   for(int i=0;i<map.rows;i++)
		   for(int j=0;j<map.cols;j++)
		   {
			   	 for(int x=-1;x<=1;x++)
				   for(int y=-1;y<=1;y++)
			   if(((x==0)^(y==0))&&i+x>=0&&j+y>=0&&i+x<map.rows&&j+y<map.cols&&*At((&out2),i+x,j+y)!=*At((&out2),i,j))//находим соседей в 4связной области
				   neibours[*At((&out2),i,j)].insert(*At((&out2),i+x,j+y));
		   }
		   for(int i=0;i<map.rows;i++)
		       for(int j=0;j<map.cols;j++)
				   if(neibours[*At((&out2),i,j)].size()==1&&*neibours[*At((&out2),i,j)].begin()!=0) //объединяем их с соответстыующими внешними
			   {
				   *At((&out2),i,j)=*neibours[*At((&out2),i,j)].begin();
			   }


		for(int i=0;i<map.rows;i++)
		   for(int j=0;j<map.cols;j++) //выделяем границы
		   {
			   bool ex=0;
			   for(int x=-1;x<=1;x++)
				   for(int y=-1;y<=1;y++)
					   if((x!=0||y!=0)&&i+x>=0&&j+y>=0&&i+x<map.rows&&j+y<map.cols&&abs(*At((&out2),i,j)-*At((&out2),i+x,j+y))>1)
						   ex=1;
			   if(ex)
				   (*At((&out2),i,j))++;
		   }
			    	for(int i=0;i<map.rows;i++)//удаляем внутрености
		   for(int j=0;j<map.cols;j++)
			   if(*At((&out2),i,j)%5!=1)
				   *At((&out2),i,j)=0;

		for(int i=0;i<map.rows;i++)//заполняе белым цветом
		   for(int j=0;j<map.cols;j++)
			   if(*At((&out2),i,j)>1)
			   {
				   *At((&out2),i,j)=255;
				/*   *At((&map),i,j)=255;
				   *(At((&map),i,j)+1)=0;
				   *(At((&map),i,j)+2)=0;*/
			   }
			  // imshow("maplab",maplab);
			   return out2;
};