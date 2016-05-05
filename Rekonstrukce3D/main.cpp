/*#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"*/
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

using namespace std;
using namespace cv;

int nalezNazVsechFot(char* adresa, vector<String> &nazvyFotek,int max) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(adresa)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			
			
			String filename = ent->d_name;
			if ((filename.find(".JPG") != string::npos) || (filename.find(".jpg") != string::npos)){
				nazvyFotek.push_back(adresa + filename);

				if (max <= nazvyFotek.size()){ return 0; }
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		cout << "nenalezen adresaz" << endl;
		return 1;
	}

	if (nazvyFotek.size() == 0){ //pokud je pole NazvuFotek prazdne opust fukci
		return 1;
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

vector<DMatch> paroveKlicoveBody(String fotkaL, String fotkaP, vector<vector<vector<KeyPoint>>> &all_keypoints){

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
	all_keypoints.push_back(keypoints);
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
/**
*Parovani dvojic odpovidajicich bodu ze dvou fotek 
@param matches struktura nese informaci o indexech bodu v keypoints1, keypoints2, které patri k sobe
@param keypoints1 nalezene body z prvni fotky
@param keypoints2 nalezene body z druhe fotky
@return [0 vpripade prvni fotky 1 vpripade druhe fotky][index konkretniho bodu]
*/
vector<vector<Point2d>> getPairsOfPoints(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){ //dvojice klicovych bodu to dvojice souradniv bodu 
	vector<cv::Point2d> points1, points2;
	//1 foto //testovacich 8 bodu pro spravnou velikost
	points1.push_back(Point2d(2404, 1582));
	points1.push_back(Point2d(2412, 1704));
	points1.push_back(Point2d(2413, 1923.5));
	points1.push_back(Point2d(2414, 2143));
	points1.push_back(Point2d(2824, 1588));
	points1.push_back(Point2d(2836, 1649.5));
	points1.push_back(Point2d(2848, 1711));
	points1.push_back(Point2d(2847, 2132));
	
	//2 foto //testovacich 8 bodu pro spravnou velikost
	points2.push_back(Point2d(2547, 1560));
	points2.push_back(Point2d(2335, 1668));
	points2.push_back(Point2d(2332, 1879));
	points2.push_back(Point2d(2329, 2090));
	points2.push_back(Point2d(2911, 1629));
	points2.push_back(Point2d(2803, 1685));
	points2.push_back(Point2d(2695, 1741));
	points2.push_back(Point2d(2693, 2167));
	for (int i = 0; i <matches.size(); i++)
	{
		// queryIdx is the "left" image
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		// trainIdx is the "right" image
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	vector<vector<Point2d>> dvojiceBodu;
	dvojiceBodu.push_back(points1);
	dvojiceBodu.push_back(points2);

	return dvojiceBodu;
} //

/**
*metoda nacita matice cameraMatrix distCoeff a imageSize
*@param nazevSouboruXmlKal cely nazev souboru i s cestou
*@param cameraMatrix do promene cameraMatrix nacte ze souboru cameraMatrix
*@param distCoeff do promene distCoeff nacte ze souboru distCoeff
*@param imageSize do promene imageSize nacte ze souboru imageSize 
*/
void readXmlCameraMatrix(string nazevSouboruXmlKal, Mat& cameraMatrix, Mat& distCoeff, Size& imageSize)
{
	Mat cam, dis;
	Size imSi;
	FileStorage fr;
	fr.open(nazevSouboruXmlKal, FileStorage::READ);
	fr["cameraMatrix"] >> cam;
	fr["imageSize"] >> imSi;
	fr["distCoeff"] >> dis;
	fr.release();
	cameraMatrix = cam;
	distCoeff = dis;
	imageSize = imSi;
}
/**
*metoda pocita esencialni matici
*@param F fundamentalni matice
*@param cameraMatrix kalibracni matice
*@return E esencialni matice
*/
Mat calculationEssentialMat(Mat F, Mat cameraMatrix){
	Mat E = cameraMatrix.t()*Mat(F)*cameraMatrix;
	return E;
}

/**
*vypocet 3D projekcniho bodu
*@param u rozsireny bod z prvni projekcni roviny (x,y,1)
*@param P1 projekcni matice z prvni projekcni roviny
*@param u1 rozsireny bod z druhe projekcni roviny (x,y,1)
*@param P2 projekcni matice z druhe projekcni roviny
*@return vypocteny 3D projekcni  bod
*/
Mat_<double> linearLSTriangulation(Point3d u, Matx34d P1, Point3d u1, Matx34d P2)
{
	//build A matrix
	Matx43d A(u.x*P1(2, 0) - P1(0, 0), u.x*P1(2, 1) - P1(0, 1), u.x*P1(2, 2) - P1(0, 2),
		u.y*P1(2, 0) - P1(1, 0), u.y*P1(2, 1) - P1(1, 1), u.y*P1(2, 2) - P1(1, 2),
		u1.x*P2(2, 0) - P2(0, 0), u1.x*P2(2, 1) - P2(0, 1), u1.x*P2(2, 2) - P2(0, 2),
		u1.y*P2(2, 0) - P2(1, 0), u1.y*P2(2, 1) - P2(1, 1), u1.y*P2(2, 2) - P2(1, 2)
		);
	//build B vector
	Matx41d B(-(u.x*P1(2, 3) - P1(0, 3)),
		-(u.y*P1(2, 3) - P1(1, 3)),
		-(u1.x*P2(2, 3) - P2(0, 3)),
		-(u1.y*P2(2, 3) - P2(1, 3)));
	//solve for X
	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);
	return X;
	//Point3d(X(0), X(1), X(2)) by mel byt 3D bod
}
/**
*metoda pocitajici 3D projekcni body dvou obrazu vyuzivajici metodu linearLSTriangulation
*@param x1K rozsireny bod z prvni projekcni roviny (x,y,1)*K^(-1)
*@param P1 projekcni matice z prvni projekcni roviny
*@param x2K rozsireny bod z druhe projekcni roviny (x,y,1)*K^(-1)
*@param P2 projekcni matice z druhe projekcni roviny

*/
void calculation3DProjectionPoints(vector<Point3d> x1K, Matx34d P1, vector<Point3d> x2K, Matx34d P2, vector<Point3d> &points3D){
	for (int k = 0; k < x1K.size(); k++){
		Mat_<double> X = linearLSTriangulation(x1K[k], P1, x2K[k], P2); //vypocet 3D bodu
		points3D.push_back(Point3d(X(0), X(1), X(2)));
	}
}

/**
*metoda rozsiruje bodu o 1 a prenasobuje ho incerznima kalibracni matici (x,y,1)*K^(-1)
*@param x pole 2D bodu (x,y)
*@param cameraMatrix matice kalibrace (K) 
*@param xK do promene xK ulozi rozsireny vektor (x,y,1)*K^(-1)
*/
void extensionsMultipliedInvK(vector<Point2d> x, vector<Point3d>& xK,Mat cameraMatrix){
	
	for (int j = 0; j < x.size(); j++){
		Point2d kp1 = x[j];
		Point3d u(kp1.x, kp1.y, 1.0);
		Mat_<double> um = cameraMatrix.inv()*Mat_<double>(u);
		u = Point3d(um.at<double>(0, 0), um.at<double>(1, 0), um.at<double>(2, 0));
		xK.push_back(u);
	}
}
/**
*metoda rozklada esencialni matici pomoci SVD rozkladu a ulozi 2 rotace a 2 translace (pouze 1 dvojice rota a translace je spravna)
*@param E esencialni matice obsahuje informaci o prechodu mezi dvemi rovinymi projekcemi 
*@param R1 do promene R1 ulozi vypoctenou rotaci 
*@param R2 do promene R2 ulozi vypoctenou rotaci
*@param t  do promene t ulozi vypoctenou translaci (druha translace ma pouze zmenene znamenko)
*/
void decomposeFMToRotTrans(Mat E, Mat_<double>& R1, Mat_<double>& R2, Mat_<double>& t){
	SVD svd(E, SVD::FULL_UV);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;

	Matx33d Sigma(1, 0, 0,
		0, 1, 0,
		0, 0, 0);
	Matx33d W(0, -1, 0,
		1, 0, 0,
		0, 0, 1);
	R1 = svd.u * Mat(W) * svd.vt;
	t = svd.u.col(2);
	R2 = svd.u * Mat(W).t() * svd.vt;

	/*double pom = t(0, 0);
	t(0, 0) = t(1, 0);
	t(1, 0) = t(2, 0);
	t(2, 0) = pom;*/

}

/**
*metoda vypocte projekcni matice pro 2 rovynna zobrazeni
*@param R rotace mezi prvni a druhym zobrazenim
*@param t translace  mezi prvni a druhym zobrazenim
*@param F fundamentalni matice
*@param P1  do promene P1 ulozi vypoctenou projekcni matici pro prvni rovinu zobrazeni
*@param P2  do promene P2 ulozi vypoctenou projekcni matici pro druhou rovinu zobrazeni
*/
void getProjectionMatrix(Mat_<double>  R, Mat_<double>  t, Mat_<double> F, Matx34d &P1, Matx34d &P2){
	P1 = Matx34d(1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0);

	Matx33d tas(0, -t(2, 0), t(1, 0),
				 t(2 ,0), 0, -t(0,0),
			   -t(1,0), t(0,0), 0);
	
	Mat_<double> tasF = Mat(tas).t()*F;
	P2 = Matx34d(tasF(0, 0), tasF(0, 1), tasF(0, 2), t(0),
				 tasF(1, 0), tasF(1, 1), tasF(1, 2), t(1),
				 tasF(2, 0), tasF(2, 1), tasF(2, 2), t(2));
	
	/*P2= Matx34d (R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2));*/
}
/**
*metoda ktera prepocitava projekcni body a eukleidovske
*@param bodyP projekcni body
*@param pPoinsts korespondujici projekcni body s eukleidovskymi body ePoints
*@param ePoinsts korespondujici eukleidovske body s projekcnimibody pPoints
*@param bodyH do promene bodyH ulozi vypoctene eukleidovske body
*/
void getEukleidenPointsFromProjection(vector<Point3d> bodyP, vector<Point3d> pPoinsts, vector<Point3d>  ePoints, vector<Point3d> &bodyH){

	//Od projekèních(X3p) 3D bodù rovnou k eukleidovským X3e pomocí homografie
	//Kontrolni body bkp - projekeni(prvnich osm), - k nim odpovídající eukleidovské a vyroba matice H

	//zvolenych 5 projekcnich a 5eukleidovskych bodu
	vector<Point3d> uPB5, uEB5;
	uPB5.push_back(bodyP[0]); 
	uPB5.push_back(bodyP[2]); 
	uPB5.push_back(bodyP[3]); 
	uPB5.push_back(bodyP[4]);
	uPB5.push_back(bodyP[5]);

	uEB5.push_back(Point3d(1, 1, 9));
	uEB5.push_back(Point3d(1, 5, 1));
	uEB5.push_back(Point3d(1, 9, 1));
	uEB5.push_back(Point3d(9, 1, 9));
	uEB5.push_back(Point3d(9, 1, 5));

	//vypocet matice A(15,16) 
	//vypocet matice A je:
	//kazdé 3 radky jsou pro jeden bod
	Mat A = Mat::zeros(15, 16, CV_64FC1);
	for (int i = 0; i < 5; i++){
		//1+3i radek matice A
		A.at<double>(0 + 3 * i, 0) = -uPB5[i].x;
		A.at<double>(0 + 3 * i, 1) = -uPB5[i].y;
		A.at<double>(0 + 3 * i, 2) = -uPB5[i].z;
		A.at<double>(0 + 3 * i, 3) = -1;
		A.at<double>(0 + 3 * i, 12) = uPB5[i].x * uEB5[i].x;
		A.at<double>(0 + 3 * i, 13) = uPB5[i].y * uEB5[i].x;
		A.at<double>(0 + 3 * i, 14) = uPB5[i].z * uEB5[i].x;
		A.at<double>(0 + 3 * i, 15) = 1 * uEB5[i].x;
		//2+3i radek matice A
		A.at<double>(1 + 3 * i, 4) = -uPB5[i].x;
		A.at<double>(1 + 3 * i, 5) = -uPB5[i].y;
		A.at<double>(1 + 3 * i, 6) = -uPB5[i].z;
		A.at<double>(1 + 3 * i, 7) = -1;
		A.at<double>(1 + 3 * i, 12) = uPB5[i].x* uEB5[i].y;
		A.at<double>(1 + 3 * i, 13) = uPB5[i].y*uEB5[i].y;
		A.at<double>(1 + 3 * i, 14) = uPB5[i].z*uEB5[i].y;
		A.at<double>(1 + 3 * i, 15) = 1 * uEB5[i].y;
		//3+3i radek matice A
		A.at<double>(2 + 3 * i, 8) = -uPB5[i].x;
		A.at<double>(2 + 3 * i, 9) = -uPB5[i].y;
		A.at<double>(2 + 3 * i, 10) = -uPB5[i].z;
		A.at<double>(2 + 3 * i, 11) = -1;
		A.at<double>(2 + 3 * i, 12) = uPB5[i].x* uEB5[i].z;
		A.at<double>(2 + 3 * i, 13) = uPB5[i].y*uEB5[i].z;
		A.at<double>(2 + 3 * i, 14) = uPB5[i].z*uEB5[i].z;
		A.at<double>(2 + 3 * i, 15) = 1 * uEB5[i].z;
	}
	//SVD rozklad matice A
	SVD svd(A, SVD::FULL_UV);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;
	
	//matice homogeni
	Matx44d H(svd_vt.at<double>(15, 0), svd_vt.at<double>(15, 1), svd_vt.at<double>(15, 2), svd_vt.at<double>(15, 3),
		svd_vt.at<double>(15, 4), svd_vt.at<double>(15, 5), svd_vt.at<double>(15, 6), svd_vt.at<double>(15, 7),
		svd_vt.at<double>(15, 8), svd_vt.at<double>(15, 9), svd_vt.at<double>(15, 10), svd_vt.at<double>(15, 11),
		svd_vt.at<double>(15, 12), svd_vt.at<double>(15, 13), svd_vt.at<double>(15, 14), svd_vt.at<double>(15, 15));
	 

	double konstaTomiczkova = 10000 / 15;
	//H = H * konstaTomiczkova; 
	
	for (int i = 0; i < bodyP.size(); i++){
		Matx41d hB4d;
		hB4d(0, 0) = bodyP[i].x;
		hB4d(1, 0) = bodyP[i].y; 
		hB4d(2, 0) = bodyP[i].z; 
		hB4d(3, 0) = 1.0;

		Matx41d bod4D;
		bod4D = H*hB4d;
	
		Point3d bH;
		bH.x = bod4D(0, 0) / bod4D(3, 0);
		bH.y = bod4D(1, 0) / bod4D(3, 0);
		bH.z = bod4D(2, 0) / bod4D(3, 0);

	
		bodyH.push_back(bH);
	}

}

/**
*metoda pocitajici projekcni 3D body 
*@param pairPoints2D pole parovych 2D bodu
*@param cameraMatrix kalibracni matice
*@param points3D do promene points3D uklada vypoctene 3D projekcni body
*/
void reconstraction3DObjectOfTwoPicturesMatchesPoints(vector<vector<Point2d>>pairPoints2D, Mat cameraMatrix, vector<Point3d> &points3D){
	Mat F; //fundamentalni matice
	Mat E; //esencialni matice
	Mat_<double> R1, R2, t;// rotace R1,R2 translace t
	vector<Point3d> x1K, x2K; // body rozsirene a pranasobene incerznima K (x,y,1)*K^(-1)
	Matx34d P1, P2; //projekcni matice

	F = findFundamentalMat(pairPoints2D[0], pairPoints2D[1], FM_RANSAC, 0.1, 0.99);//(x1,y1),(x2,y2) => fundamentalni matice
	E = calculationEssentialMat(F, cameraMatrix);//esencialni matice
	decomposeFMToRotTrans(E, R1, R2, t); //F => R,t 

	extensionsMultipliedInvK(pairPoints2D[0], x1K, cameraMatrix);  //(x1,y1,1)*Kinv
	extensionsMultipliedInvK(pairPoints2D[1], x2K, cameraMatrix); //(x2,y2,1)*Kinv

	getProjectionMatrix(R2, t, F, P1, P2);//tvorba projekcnich matice P1 P2

	calculation3DProjectionPoints(x1K, P1, x2K, P2, points3D);
}
/**
*metoda uklada vektor 3D bodu do souboru xml
*@param body vektor 3D bodu
*@param jmeno je retezec znaku ktery bode obsazen v nazvu souboru
*@param castaXml cesta kam se ma soubor ulozit
*/
void save3DPoints(vector<Point3d> body, String jmeno,String cestaXml){
	FileStorage fr;
	fr.open(cestaXml+"body3D" + jmeno + ".xml", FileStorage::WRITE);
	fr << jmeno << "[";
	for (int n = 0; n < body.size() - 1; n++)
	{
		fr << body[n].x << "," << body[n].y << "," << body[n].z << ";";

	}
	fr << "];";
	fr.release();
}
 
int main()
{
	//-------------------promenne---------------------------------------------------
	vector<vector<vector<KeyPoint>>> all_keypoints;
	char* adresaChar = "../soubory/kostka_mereni/";
	//char* adresaChar = "../soubory/sachovnice/";
	String cestaXml = "../soubory/body3D/";
	String cesta2Dbody = "../soubory/";
	vector<String> nazvyFotek;

	int maxPocetFotek = 2;

	if (nalezNazVsechFot(adresaChar, nazvyFotek, maxPocetFotek) != 1){
		cout << "nalezeni " << nazvyFotek.size() << " fotek" << endl;
	}
	else{
		cout << "zadna fotka v adresari nebyla nenalezena" << endl;
		system("PAUSE");
		return 5;
	}
	vector<vector<DMatch>> all_matches;
	string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceData0317.xml";
	//string nazevSouboruXmlKal = "../soubory/xmlSoubory/Kalibrace201604141626.xml";
	Mat cameraMatrix, distCoeffs;

	
	vector<Point3d> points3D; // vypoctene 3D body
	vector<Point3d> bodyH; //body povynasobeni matici homografie => eukleidovske body
	vector<Point3d> pPoinsts, ePoints; // umele odpovidajici projekcni eukleidovske body
	vector<vector<Point2d>> pairPoints2D; // pary dvojic bodu ve dvou obrazech

	Size  sizeImage;
	//vector<String> nazvySouboru;
	readXmlCameraMatrix(nazevSouboruXmlKal, cameraMatrix, distCoeffs, sizeImage);

	//-------------------hledani-bodu-a-jejich-korespondencnich-bodu---------------------------------------------------
	cout << "pocet nazvu forek " << nazvyFotek.size() << endl;
	for (int i = 0; i < nazvyFotek.size() - 1; i++)
	{
		// matches
		vector<DMatch> good_matches = paroveKlicoveBody(nazvyFotek[i], nazvyFotek[i + 1], all_keypoints);
		all_matches.push_back(good_matches);
	}
	int imageMatchesCount = all_matches.size(); //pocet paru fotek
	cout << "matchovani bodu dokonceno v poctu" << imageMatchesCount << endl;
	//---------------------3D-rekonstrukce----------------------------------------------------------------------------
	for (int i = 0; i < imageMatchesCount; i++)
	{
		
		cout << i + 1 << ". z " << imageMatchesCount << " dvojic fotek" << endl;
		pairPoints2D = getPairsOfPoints(all_matches[i], all_keypoints[i][0], all_keypoints[i][1]);
		cout << i + 1 << ": vytvoreni dvojic 2D bodu" << endl;

		if (pairPoints2D[0].size() != pairPoints2D[1].size()){ //pokud pocet parovych bodu prvni a druhe fotky naodpovida ukonci program
			cout << i + 1 << ": pocet parovych bodu prvni a druhe fotky neni stejny " << endl;
			return 5;
		}
		reconstraction3DObjectOfTwoPicturesMatchesPoints(pairPoints2D, cameraMatrix, points3D);

		getEukleidenPointsFromProjection(points3D,pPoinsts,ePoints, bodyH);

		save3DPoints(points3D, "kostka", cestaXml);
		save3DPoints(bodyH, "kostka_po_homografii", cestaXml);
	}
	system("PAUSE");
	return 0;
}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     