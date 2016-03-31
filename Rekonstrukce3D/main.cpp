/*#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"*/
#include <string>

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"



using namespace std;
using namespace cv;

int nalezeniNazvuFotek(String adresa, vector<String> &nazvyFotek, int maxPocet)
{
	String nazevSouboru;
	Mat frame; // obrazek
	int citac = 10; // citac pro vstupni soubory
	
	while (true) {
		if (citac < 100){nazevSouboru = adresa + "image0" + to_string(citac) + ".jpg";}
		else{nazevSouboru = adresa + "image" + to_string(citac) + ".jpg";}
		
		cout << "Nacitam soubor se jmenem: " << nazevSouboru << endl;
		waitKey(30);
		frame = imread(nazevSouboru, CV_LOAD_IMAGE_GRAYSCALE); // nacteni fotky ve stupni sedi
		if (frame.empty()) {
			cout << "Nazev fotky neexistuje: " << nazevSouboru << endl;
			break;
		}
		cout << "existuje " << endl;
		citac++;
		if (maxPocet + 10 <= citac){
			cout << "Omezeni maxPoctu fotek: " << maxPocet << endl;
			break;
		}

		nazvyFotek.push_back(nazevSouboru);
	}
	return 0;
}

int nalezeniVyzBodSurf(Mat obrazek, vector<vector<KeyPoint>> &klice)
{

	int nelezeno = -1; // navratova prom > pocet naleyenzch bodu
	
	vector<KeyPoint> keypoints; // klicove body
	Mat img_keypoints; //obrazek s klicovzmi body



		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;
		SurfFeatureDetector detector(minHessian);
		detector.detect(obrazek, keypoints);
		klice.push_back(keypoints);
		
	
	//-- Draw keypoints (pouze posledniho obr)
		drawKeypoints(obrazek, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT); //zakresleni klicovych bodu  
	//namedWindow("SURF-Keypoints", CV_WINDOW_AUTOSIZE);
	//imshow("SURF-Keypoints", img_keypoints); // vykresli obrazek s klicovvymi body
	//waitKey(30);
	return nelezeno = 0;
}

int nalezeniVyzBodShift(Mat img, vector<vector<KeyPoint>> &klice)
{

	Mat frame;
	vector<KeyPoint> keypoints;
	Mat output;

		Ptr<FeatureDetector> feature_detector = FeatureDetector::create("SIFT");
		feature_detector->detect(img, keypoints);

		klice.push_back(keypoints);
	
		drawKeypoints(img, keypoints, output, Scalar::all(-1));

	namedWindow("SHIFT-method", CV_WINDOW_AUTOSIZE);
	imshow("SHIFT-method", output);
	waitKey(30);

	return 0;
}

int vypocetDeskriptoru(Mat img, vector<KeyPoint> keypoints, Mat &descriptors)
{
	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	//Mat descript;
	extractor.compute(img, keypoints, descriptors);
	//descriptors.push_back(descript);
	
	return 0;
}

int slouceniSOdpovidajiciHloubku(vector< DMatch > &matches, Mat descriptors1, Mat descriptors2){
	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	matcher.match(descriptors1, descriptors2, matches);
	
	return 0;
}

double minMaxVzdalenostKlici(Mat descriptors_1, vector< DMatch > matches){
	//-- Quick calculation of max and min distances between keypoints

	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	return min_dist;
}

vector< DMatch > sparovaniSpravnychKlicovychBodu(Mat descriptors_1, vector< DMatch > matches, double min){
	vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}
	
	return  good_matches;
}

vector<DMatch>  loweUpraveneSparovani(Mat descriptorsL, Mat descriptorsR){

	std::vector<std::vector<cv::DMatch>> matches;
	BFMatcher matcher;
	//FlannBasedMatcher matcher;
	matcher.knnMatch(descriptorsL, descriptorsR, matches, 2);  // Find two nearest matches
	vector<cv::DMatch> good_matches;

	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.7; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
			//printf("%f \n", matches[i][0].distance);
		}
	}
	
	return good_matches;
}

vector<DMatch> paroveKlicoveBody(String fotkaL, String fotkaP, vector<vector<vector<KeyPoint>>> &All_keypoints){

	vector<vector<KeyPoint>> keypoints;
	Mat descriptorsL;
	Mat descriptorsR;
	//-- 2 fotky
	Mat img_1 = imread(fotkaL, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread(fotkaP, CV_LOAD_IMAGE_GRAYSCALE);
	/*
	Mat img_1 = imread(nazvyFotek[0], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread(nazvyFotek[1], CV_LOAD_IMAGE_GRAYSCALE);
	*/
	//-- Step 1: Detect the keypoints using SURF/Shift Detector
	//nalezeniVyzBodShift(img_1, keypoints);
	//nalezeniVyzBodShift(img_2, keypoints);
	nalezeniVyzBodSurf(img_1, keypoints);
	nalezeniVyzBodSurf(img_2, keypoints);
	All_keypoints.push_back(keypoints);
	//-- Step 2: Calculate descriptors (feature vectors)
	vypocetDeskriptoru(img_1, keypoints[0], descriptorsL);
	vypocetDeskriptoru(img_2, keypoints[1], descriptorsR);
	/*
	vector< DMatch > matches;
	vector< DMatch > good_matches;
	FlannBasedMatcher matcher;
	matcher.match(descriptorsL, descriptorsR, matches);
	*/
	// lowe, z matches dostaneme good matches
	//-------------------------------------------------------------
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	vector<DMatch> good_matches = loweUpraveneSparovani(descriptorsL, descriptorsR);
	//-------------------------------------------------------------//vykresleni dvojice obr s parovanymi body
	Mat img_matches;
	drawMatches(img_1, keypoints[0], img_2, keypoints[1], good_matches, img_matches);
	//-- Show detected matches
	//imshow("Matches", img_matches);

	waitKey(30);

	return good_matches;
}

vector<vector<Point2f>> parovaniDodu(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){
	vector<cv::Point2f> points1, points2;
	for (int i = 0; i < matches.size(); i++){
		float qx = keypoints1[matches[i].queryIdx].pt.x;
		float qy = keypoints1[matches[i].queryIdx].pt.y;
		points1.push_back(cv::Point2f(qx, qy));
		float tx = keypoints2[matches[i].trainIdx].pt.x;
		float ty = keypoints2[matches[i].trainIdx].pt.y;
		points2.push_back(cv::Point2f(tx, ty));
	}

	vector<vector<Point2f>> dvojiceBodu;
	dvojiceBodu.push_back(points1);
	dvojiceBodu.push_back(points2);

	return dvojiceBodu;
}

Mat readXmlCameraMatrix(string nazevSouboruXmlKal)
{
	Mat  cameraMatrix;
	FileStorage fr;
	fr.open(nazevSouboruXmlKal, FileStorage::READ);
	fr["cameraMatrix"] >> cameraMatrix;
	fr.release();
	cout << "nacteni kalibracni matice" << endl;
	return  cameraMatrix;
}

Mat ffMatrix(vector<vector<Point2f>> dvojiceBodu2D){
	Mat matrixFF = findFundamentalMat(Mat(dvojiceBodu2D[0]), Mat(dvojiceBodu2D[1]));

	return matrixFF;
}

vector<Mat> rozkladFMnaRotaciTranslaci(Mat ffmatice, Mat calibrateCamera){
	Mat E = calibrateCamera*ffmatice*calibrateCamera;
	vector<Mat> rotTran;
	

	SVD svd(E, SVD::MODIFY_A);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;

	Matx33d W(0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	Mat R1 = svd_u * Mat(W).t() * svd_vt; //or svd_u * Mat(W) * svd_vt; 
	Mat t1 = svd_u.col(2); //or -svd_u.col(2)

	Mat R2 = svd_u * Mat(W) * svd_vt; //or svd_u * Mat(W) * svd_vt; 
	Mat t2 = -svd_u.col(2); //or -svd_u.col(2)



	rotTran.push_back(R1);
	rotTran.push_back(t1);
	
	return rotTran;
}

vector<Point3f> body3DpomociRT(vector<Mat> rotTran, vector<vector<Point2f>> dvojiceBodu){
	float pixel = 0.026;
	Point3f bod3D;
	vector<Point3f> body3D;
	//------------------------------------------------------------------------------
	for (int i = 0; i < dvojiceBodu[0].size(); i++){ //cyklus pres dvojice bodu
		float yo = dvojiceBodu[1][i].y*pixel; // y`*pixel
		Mat Y; //bod Y z prvniho borazu
		Y.push_back(dvojiceBodu[0][i].x*pixel);// x*pixel
		Y.push_back(dvojiceBodu[0][i].y*pixel);// y*pixel
		Y.push_back(1.0);// z //doplnek
		Mat citatelJmenovatel = (rotTran[0].col(1) - yo*rotTran[0].col(2)); //Rotace(druhy radek) - y` * /Rotace(treti radek) 
		float a = citatelJmenovatel.dot(rotTran[1].t()); // citatel = citatelJmenovatel * Y
		float b = citatelJmenovatel.dot(Y.t());
		float hloubka = a / b; //vypocet z-ove souradnice bodu Y
		bod3D.x = dvojiceBodu[0][i].x*pixel;
		bod3D.y = dvojiceBodu[0][i].y*pixel;
		bod3D.z = hloubka;
		body3D.push_back(bod3D);
	}
	return body3D;
}

int main()
{
	vector<vector<vector<KeyPoint>>> All_keypoints;
	vector<Mat> All_ffmatice;
	String adresa = "../soubory/fotoKostka/";;
	vector<String> nazvyFotek;
	nalezeniNazvuFotek(adresa, nazvyFotek, 2);
	vector<vector<DMatch>> all_matches;
	string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceData0317.xml";
	Mat cameraMatrix;
	vector<String> nazvySouboru;
	cameraMatrix = readXmlCameraMatrix(nazevSouboruXmlKal);


	for (int i = 0; i < nazvyFotek.size() - 1; i++){
		// matches
		vector<DMatch> good_matches = paroveKlicoveBody(nazvyFotek[i], nazvyFotek[i + 1], All_keypoints);
		all_matches.push_back(good_matches);
		
	}	
	cout << "matchovani bodu dokonceno v poctu" << all_matches.size() << endl;

	for (int i = 0; i < all_matches.size(); i++){
		vector<vector<Point2f>> dvojiceBodu2D = parovaniDodu(all_matches[i], All_keypoints[i][0], All_keypoints[i][1]);
		cout << "vytvoreni dvojic 2D bodu" << endl;
		Mat ffmatice = ffMatrix(dvojiceBodu2D);//find Fundamental Matatrix
		cout << "vytvorena fundamentalni matice" << endl;

		
		// rozklad na rotaci a translaci
		vector<Mat> rotTran = rozkladFMnaRotaciTranslaci(ffmatice, cameraMatrix);
		cout << "vytvorena rotacni a translacni matice z fundamental matrix" << endl;
		//vytvoreni 3D bodu

	}
	system("PAUSE");
	return 0;
}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     