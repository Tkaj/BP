#include <string>


#include <fstream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <dirent.h>



using namespace cv;
using namespace std;


void homografije(){
	Point3d x1e(1, 1, 9);
	Point3d x2e(1, 1, 1);
	Point3d x3e(1, 9, 1);

	Point3d x4e(9, 1, 9);
	Point3d x5e(9, 1, 1);
	Point3d x6e(9, 9, 1);

	Matx14d x1(1.1614318151707870e-002, 7.9548893655233439e-003, -3.4251734905348899e-002, 1);
	Matx14d x2(1.1680559498246559e-002, 7.8508645215715177e-003, -3.4454732864969029e-002, 1);
	Matx14d x3(1.1613024281996297e-002, 7.3288674624992500e-003, -3.4270192847181302e-002, 1);

	Matx14d x4(1.0617425488710846e-002, 7.5591201526105066e-003, -3.2739418022367024e-002, 1);
	Matx14d x5(1.0634792823109494e-002, 7.4613940128910681e-003, -3.2899054296414890e-002, 1);
	Matx14d x6(1.0549011127588846e-002, 6.9482200706375702e-003, -3.2648748138794259e-002, 1);
	Matx14d nul(0, 0, 0, 0);

	Mat_<Matx14d> A1;
	Mat_<Matx14d> A1raw;
	A1raw.push_back(-x1);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x1*x1e.x);
	A1raw = A1raw.t();
	Mat_<Matx14d> A2raw;
	A2raw.push_back(nul);
	A2raw.push_back(-x1);
	A2raw.push_back(nul);
	A2raw.push_back(x1*x1e.y);
	A2raw = A2raw.t();
	Mat_<Matx14d> A3raw;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x1);
	A3raw.push_back(x1*x1e.z);
	A3raw = A3raw.t();

	A1.push_back(A1raw);
	A1.push_back(A2raw);
	A1.push_back(A3raw);
	//---------------------------------------------------------
	/*Mat_<Matx14d> A2;
	A1raw = 0;
	A1raw.push_back(-x2);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x2*x2e.x);
	A1raw = A1raw.t();
	A2raw = 0;
	A2raw.push_back(nul);
	A2raw.push_back(-x2);
	A2raw.push_back(nul);
	A2raw.push_back(x2*x2e.y);
	A2raw = A2raw.t();
	A3raw = 0;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x2);
	A3raw.push_back(x2*x2e.z);
	A3raw = A3raw.t();

	A2.push_back(A1raw);
	A2.push_back(A2raw);
	A2.push_back(A3raw);
	//---------------------------------------------------------
	Mat_<Matx14d> A3;
	A1raw = 0;
	A1raw.push_back(-x3);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x3*x3e.x);
	A1raw = A1raw.t();
	A2raw = 0;
	A2raw.push_back(nul);
	A2raw.push_back(-x3);
	A2raw.push_back(nul);
	A2raw.push_back(x3*x3e.y);
	A2raw = A2raw.t();
	A3raw = 0;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x3);
	A3raw.push_back(x3*x3e.z);
	A3raw = A3raw.t();

	A3.push_back(A1raw);
	A3.push_back(A2raw);
	A3.push_back(A3raw);
	//---------------------------------------------------------
	Mat_<Matx14d> A4;
	A1raw = 0;
	A1raw.push_back(-x4);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x4*x4e.x);
	A1raw = A1raw.t();
	A2raw = 0;
	A2raw.push_back(nul);
	A2raw.push_back(-x4);
	A2raw.push_back(nul);
	A2raw.push_back(x4*x4e.y);
	A2raw = A2raw.t();
	A3raw = 0;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x4);
	A3raw.push_back(x4*x4e.z);
	A3raw = A3raw.t();

	A4.push_back(A1raw);
	A4.push_back(A2raw);
	A4.push_back(A3raw);
	//---------------------------------------------------------
	Mat_<Matx14d> A5;
	A1raw = 0;
	A1raw.push_back(-x5);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x5*x5e.x);
	A1raw = A1raw.t();
	A2raw = 0;
	A2raw.push_back(nul);
	A2raw.push_back(-x5);
	A2raw.push_back(nul);
	A2raw.push_back(x5*x5e.y);
	A2raw = A2raw.t();
	A3raw = 0;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x5);
	A3raw.push_back(x5*x5e.z);
	A3raw = A3raw.t();

	A5.push_back(A1raw);
	A5.push_back(A2raw);
	A5.push_back(A3raw);
	//---------------------------------------------------------
	Mat_<Matx14d> A6;
	A1raw = 0;
	A1raw.push_back(-x6);
	A1raw.push_back(nul);
	A1raw.push_back(nul);
	A1raw.push_back(x6*x6e.x);
	A1raw = A1raw.t();
	A2raw = 0;
	A2raw.push_back(nul);
	A2raw.push_back(-x6);
	A2raw.push_back(nul);
	A2raw.push_back(x6*x6e.y);
	A2raw = A2raw.t();
	A3raw = 0;
	A3raw.push_back(nul);
	A3raw.push_back(nul);
	A3raw.push_back(-x6);
	A3raw.push_back(x6*x6e.z);
	A3raw = A3raw.t();

	A6.push_back(A1raw);
	A6.push_back(A2raw);
	A6.push_back(A3raw);*/
	//-----------------------------------------------
	Mat A1 = Mat(A1);
	//A.push_back(A1);
	//A.push_back(A2);
	//A.push_back(A3);
	//A.push_back(A4);
	//A.push_back(A5);
	//A.push_back(A6);

	cout << A1 << endl;
 
}


int main()
{
	
	homografije();	
	system("PAUSE");
	/*
	Point3d b1(1, 1, 9);
	Point3d b2(1, 1, 1);
	Point3d b3(1, 9, 1);
	
	Point3d b4(9, 1, 9);
	Point3d b5(9, 1, 1);
	Point3d b6(9, 9, 1);
	
	vector<Point3d> B;
	
	
	B.push_back(b1); B.push_back(b2); B.push_back(b3); B.push_back(b4); B.push_back(b5); B.push_back(b6); 
	Mat Bmat = Mat(B);
	cout << Bmat << endl;
	Point3d b13D(1.1614318151707870e-002, 7.9548893655233439e-003, -3.4251734905348899e-002);
	Point3d b23D(1.1680559498246559e-002, 7.8508645215715177e-003, -3.4454732864969029e-002);
	Point3d b33D(1.1613024281996297e-002, 7.3288674624992500e-003, -3.4270192847181302e-002);
	
	Point3d b43D(1.0617425488710846e-002, 7.5591201526105066e-003, -3.2739418022367024e-002);
	Point3d b53D(1.0634792823109494e-002, 7.4613940128910681e-003, -3.2899054296414890e-002);
	Point3d b63D(1.0549011127588846e-002, 6.9482200706375702e-003, -3.2648748138794259e-002);
	
	vector<Point3d> B3D;
	B3D.push_back(b13D); B3D.push_back(b23D); B3D.push_back(b33D); B3D.push_back(b43D); B3D.push_back(b53D); B3D.push_back(b63D);
	Mat B3Dmat = Mat(B3D);
	
	B3Dmat.convertTo(B3Dmat, CV_32FC2);
	Bmat.convertTo(Bmat, CV_32FC2);
	Mat H = findHomography(Bmat, B3Dmat, CV_RANSAC);


	//cout << H << endl;
	*/
	system("PAUSE");	
	return 1;
}