---
layout: post
title: Compile test code with opencv
---

之前用opencv仅限于在caffe代码里面调用，今天需要写个测试程序才发现编译需要一点技巧，参考http://www.cnblogs.com/woshijpf/p/3840530.html。

加上-L是因为找不到报错：“/usr/bin/ld: cannot find -lcufft”, 在这里http://blog.sciencenet.cn/blog-1583812-841855.html找到了答案

```
g++ opencv_imshow.cpp -o imshow -L /usr/local/cuda/lib64/ `pkg-config --cflags --libs opencv` 

```

以下代码是为了测试opencv的resize函数的输出

```
//jin: compare output of imresize by opencv and by the version of Boss
//20160401 14:16
#include  <stdio.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;   
using namespace cv;

int bilinear_image_resize(const unsigned char *src, int width, int height,
								unsigned char *dst, int nw, int nh)
{
	double dx, dy, x, y;
	int x1, y1, x2, y2;	
	int i, j;	
	
	unsigned char *pout = dst;
	double f1, f2, f3, f4, f12, f34, val;
	
	
	//输入图像的采样间隔
	dx = (double) (width) / (double) (nw);
	dy = (double) (height) / (double) (nh);
	
	for (j=0; j<nh; ++j) 
	{
		y = j * dy;
		y1 = (int) y;
		y2 = y1 + 1;
		y2 = y2>height-1? height-1 : y2;
		for (i=0; i<nw; ++i) 
		{
			x = i * dx;
			x1 = (int) x;
			x2 = x1 + 1;
			x2 = x2>width-1? width-1 : x2;
			
			f1 = *(src + width * y1 + x1);
			f3 = *(src + width * y1 + x2);
			f2 = *(src + width * y2 + x1);
			f4 = *(src + width * y2 + x2);
			
			f12 = (f1  + (y-y1) * (f2 - f1));
			f34 = (f3  + (y-y1) * (f4 - f3));
			
			//+0.5是为了四舍五入，加0.00001是为了尽量避开取整时的大误差情况
			val = f12 + (x - x1) * (f34 - f12) + 0.5;
			if(val > 255)
				*pout = 255;
			else
				*pout = (unsigned char) val;
			++pout;
		}
	}
	
	return 0;
}

int main( int argc, char** argv )
{

//0 read an image
char *imageName = "48220154_3.bmp";

 Mat image;
 image = imread( imageName, CV_LOAD_IMAGE_GRAYSCALE);

 if( !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

//1 resize it using cv::imresize
cv::Mat mat;

cv::resize(image, mat, Size(12, 12), INTER_LINEAR);

printf("print output of cv::resize:\n");
std::cout << mat << std::endl;

// cv::cvtColor( image, gray_image,  COLOR_RGB2GRAY );
// imwrite( "../../images/Gray_Image.jpg", gray_image );

//2 resize it using bilinear_image_resize
unsigned char *dst2 = (unsigned char*) malloc(12*12*sizeof(unsigned char));
if(!dst2){
    printf("malloc failed\n");
    return -1;
}

int wd = image.cols,  ht =image.rows;
bilinear_image_resize(image.data, wd, ht, dst2, 12, 12);
//jin: this is a WRONG way to construct a Mat// OKay now, no need to set the 5th argument step
Mat mat2(12, 12, CV_8UC1, dst2);

//print dst2
printf("print output of bilinear_image_resize:\n");
std::cout << mat2 << std::endl;


 return 0;
}
```
