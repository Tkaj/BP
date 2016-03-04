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

int nalezeniNazvuFotek(String adresa, vector<String> &nazvyFotek)
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
			system("PAUSE");
			break;
		}
		cout << "existuje " << endl;
		citac++;

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
	//-------------------------------------------------------------
	Mat img_matches;
	drawMatches(img_1, keypoints[0], img_2, keypoints[1], good_matches, img_matches);
	//-- Show detected matches
	imshow("Matches", img_matches);

	waitKey(30);

	return good_matches;
}

Mat ffMatrix(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){

	vector<cv::Point2f> points1, points2;
	for (int i = 0; i < matches.size(); i++){
		float qx = keypoints1[matches[i].queryIdx].pt.x;
		float qy = keypoints1[matches[i].queryIdx].pt.y;
		points1.push_back(cv::Point2f(qx, qy));
		float tx = keypoints2[matches[i].trainIdx].pt.x;
		float ty = keypoints2[matches[i].trainIdx].pt.y;
		points2.push_back(cv::Point2f(tx, ty));
	}
	Mat matrixFF = findFundamentalMat(Mat(points1), Mat(points2));

	return matrixFF;
}

/*
void saveData(vector<DMatch> good_matches, Mat ffmatice, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,int kolikaty){
	String jmenoXml = "keyMatchffMatrix" + kolikaty;
	jmenoXml = jmenoXml + ".xml";
	FileStorage fs(jmenoXml, FileStorage::WRITE);
	fs << "good_matches" << good_matches;
	fs << "keypoints1" << keypoints1;
	fs << "keypoints2" << keypoints2;

	fs.release();
}
*/

int main()
{
	vector<vector<vector<KeyPoint>>> All_keypoints;
	vector<Mat> All_ffmatice;
	String adresa = "../soubory/fotoKostka/";;
	vector<String> nazvyFotek;
	nalezeniNazvuFotek(adresa, nazvyFotek);
	vector<vector<DMatch>> all_matches;

	for (int i = 0; i < nazvyFotek.size() - 1; i++){
		// matches
		vector<DMatch> good_matches = paroveKlicoveBody(nazvyFotek[i], nazvyFotek[i + 1], All_keypoints);
		all_matches.push_back(good_matches);
	}	

	for (int i = 0; i < all_matches.size(); i++){
		//find Fundamental Matatrix
		Mat ffmatice = ffMatrix(all_matches[i], All_keypoints[i][0], All_keypoints[i][1]);
		All_ffmatice.push_back(ffmatice);
		cout << ffmatice << endl;
		//saveData(good_matches, ffmatice, All_keypoints[i][0], All_keypoints[i][1], i);
	}
	system("PAUSE");
	return 0;
}




