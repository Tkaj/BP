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



using namespace std;
using namespace cv;

int nalezeniNazvuFotek(String adresa, vector<String> &nazvyFotek, int maxPocet)
{
	String nazevSouboru;
	Mat frame; // obrazek
	int pocatek = 20;
		int citac = pocatek; // citac pro vstupni soubory
		
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
		nazvyFotek.push_back(nazevSouboru);

		if (maxPocet + pocatek <= citac){
			cout << "Omezeni maxPoctu fotek: " << maxPocet << endl;
			break;
		}
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

vector<vector<Point2d>> parovaniDodu(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){
	vector<cv::Point2d> points1, points2;
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
}

vector<Mat> readXmlCameraMatrix(string nazevSouboruXmlKal)
{
	Mat  cameraMatrix, distCoeff;
	vector<Mat> calibrateDistor;
	FileStorage fr;
	fr.open(nazevSouboruXmlKal, FileStorage::READ);
	fr["cameraMatrix"] >> cameraMatrix;
	fr["distCoeff"] >> distCoeff;
	fr.release();
	cout << "nacteni kalibracni matice" << endl;
	calibrateDistor.push_back(cameraMatrix);
	calibrateDistor.push_back(distCoeff);
	return  calibrateDistor;
}

Mat overenifundMatice(Mat ffMatrix, vector<vector<Point2d>> dvojiceBodu){
	Mat nula;
	Mat all_nula;
	Point3f prvni;
	Point3f druhy;
	float max = 0;
	for (int a = 0; a < dvojiceBodu[0].size(); a++){
		prvni.x = dvojiceBodu[0][a].x;
		prvni.y = dvojiceBodu[0][a].y;
		prvni.z = 1;
		Mat prvnim = Mat(prvni);
		prvnim.convertTo(prvnim, CV_64F);
		druhy.x = dvojiceBodu[1][a].x;
		druhy.y = dvojiceBodu[1][a].y;
		druhy.z = 1;
		Mat druhym = Mat(druhy);
		druhym.convertTo(druhym, CV_64F);

		nula = Mat(prvnim).t()*ffMatrix*Mat(druhym);
		if (max < nula.at<float>(0, 0)){
			max = nula.at<float>(0, 0);
		}
		all_nula.push_back(nula);

	}
	return all_nula;
}

Mat_<double> LinearLSTriangulation(
	Point3d u,//homogenous image point (u,v,1)
	Matx34d P,//camera 1 matrix
	Point3d u1,//homogenous image point in 2nd camera
	Matx34d P1//camera 2 matrix
	)
{
	//build A matrix
	Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
	//build B vector
	Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));
	//solve for X
	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);
	return X;
	//Point3d(X(0), X(1), X(2)) by mel byt 3D bod
}

vector<Mat_<double>> rozkladFMnaRotaciTranslaci(Mat F, Mat K){
	//Mat E = K.t()*F*K;

	Matx33d F_(0, -2, 1000,
		2, 0, -500,
		-1000, 500, 0);

	Matx33d K_(500, 0, 250,
		0, 500, 250,
		0, 0, 1);

	Mat E = Mat(K_).t()*Mat(F_)*Mat(K_);
	/*
	Mat M = Mat::zeros(4, 5, CV_64F);
	M.col(0).row(0) = 1;
	M.col(4).row(0) = 2;
	M.col(2).row(1) = 3;
	M.col(1).row(3) = 2;
	cout << "M " << M << endl;
	system("PAUSE");
	*/
	SVD svd(E, SVD::FULL_UV);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;
	cout << "u " << svd_u << endl;
	cout << "vt " << svd_vt << endl;
	cout << "w " << svd_w << endl;
	/*
	Mat WWW = Mat::zeros(4, 5, CV_64F);
	WWW.col(0).row(0) = svd_w.at<double>(0, 0);
	WWW.col(1).row(1) = svd_w.at<double>(1, 0);
	WWW.col(2).row(2) = svd_w.at<double>(2, 0);
	WWW.col(3).row(3) = svd_w.at<double>(3, 0);

	Mat pinvA = svd_u*WWW*svd_vt;
	cout << "M " << pinvA << endl;

	system("PAUSE");
	*/
	//Mat testE= svd_u*svd_w*svd_vt;
	// cout << "test esencialni matice: " << testE << endl;
	Matx33d W(0, -1, 0,
		1, 0, 0,
		0, 0, 1);
	Matx31d T_(0, 1, 2);
	Mat_<double> R1 = svd.u * Mat(W) * svd.vt;
	Mat_<double> t1 = svd.u.col(2);
	Mat_<double> R2 = svd.u * Mat(W).t() * svd.vt;



	double pom = t1.at<double>(0, 0);
	t1(0, 0) = t1(1, 0);
	t1(1, 0) = t1(2, 0);
	t1(2, 0) = pom;

	t1(0, 0) = 0.4472;
	t1(1, 0) = 0.8944;
	t1(2, 0) = 0.0018;
	cout << "t1: " << t1 << endl;

	Matx33d tas(0, -t1(2), t1(1), // antisimetricka t2
		t1(2), 0, -t1(0),
		-t1(1), t1(0), 0);

	cout << "tas: " << tas << endl;

	Mat_<double> tasTF = Mat(tas).t()*Mat(F_);
	cout << "tasTF " << tasTF << endl;
	Matx34d P2p(tasTF(0, 0), tasTF(0, 1), tasTF(0, 2), t1(0, 0),
			tasTF(1, 0), tasTF(1, 1), tasTF(1, 2), t1(1, 0),
			tasTF(2, 0), tasTF(2, 1), tasTF(2, 2), t1(2, 0));
	cout << "P2p " << P2p << endl;

	Matx34d P2a(1, 0, 0, 500,
		0, 1, 0, 1000,
		0, 0, 1, 2);

	
	Mat K_T_ = Mat(K_)*Mat(T_);
	double normaKT = norm(K_T_);  // ||KT||
	cout << "normaKT " << normaKT << endl;
	double normaKT2inv = 1 /(normaKT*normaKT);
	cout << "1/normaKT^2 " << normaKT2inv << endl;
	Mat_<double> t1NormaKT = -1*Mat(t1.t()) / normaKT;
	cout << "-t1.t()/normaKT" << t1NormaKT << endl;
	Matx44d Hpinv(1, 0, 0, 0, //testovaci Homogeni matice
			   0, 1, 0, 0,
			   0, 0, 1, 0,
			   t1NormaKT(0, 0), t1NormaKT(0, 1), t1NormaKT(0, 2), normaKT2inv);

	cout << "HomografiMatrix  " << Mat(Hpinv) << endl;
	Mat P2p_vyp = Mat(P2a)*Mat(Hpinv);
	cout << "P2p_vyp " << P2p_vyp << endl;
	Mat P2a_vyp = Mat(P2p)*(Mat(Hpinv).inv);

	cout << "P2a_vyp " << P2a_vyp << endl;

	
	Matx31d vlastC;
	eigen(Mat(F_).t()*Mat(tas), vlastC); //vypocet vlastnich cisel F.t()*tas
	cout << "vlastC: " << vlastC << endl;

	Mat T1 = Mat(K_).inv()*t1*vlastC(0, 0); // vypocet spravne velikosti translace
	Mat T2 = Mat(K_).inv()*t1*vlastC(1, 0); // vypocet spravne velikosti translace
	Mat T3 = Mat(K_).inv()*t1*vlastC(2, 0); // vypocet spravne velikosti translace
	cout <<"R1 "<< R1 << endl;
	cout << "R2 " << R2 << endl;
	cout << "t1 " << t1 << endl;
	cout << "T1 " << T1 << endl;
	cout << "T2 " << T2 << endl;
	cout << "T3 " << T3 << endl;
	system("PAUSE");
	vector<Mat_<double>> rotTran;
	rotTran.push_back(R1);
	rotTran.push_back(R2);
	
	rotTran.push_back(T1);
	rotTran.push_back(-T1);

	return rotTran;
}

vector<Point3d> body3DpomociRT(Mat rot, Mat tran, vector<vector<Point2d>> dvojiceBodu, Mat K){
	vector<Point3d> body3D;
	//------------------------------------------------------------------------------
	for (int i = 0; i < dvojiceBodu[0].size(); i++){ //cyklus pres dvojice bodu
		Point3d Y;
		Y.x = dvojiceBodu[0][i].x;
		Y.y = dvojiceBodu[0][i].y;
		Y.z = 1.0;
		Point3d Yo;
		Yo.x = dvojiceBodu[1][i].x;
		Yo.y = dvojiceBodu[1][i].y;
		Yo.z = 1.0;		
		//---------------------
		Mat Ymat = K.inv()*Mat(Y);//pretypovani 
		Ymat.convertTo(Ymat, CV_64F);
		//---------------------------
		Mat Yomat = K.inv()*Mat(Yo);//pretypovani 
		Yomat.convertTo(Ymat, CV_64F);
		//------------------------------vzorecek pro dopocet z-ove souradnice (hloubka)
		Mat cinitel = (rot.row(1) - Yomat.at<double>(1,0)*rot.row(2)).t(); //Rotace(druhy radek) - y` * /Rotace(treti radek) 
		float a = cinitel.dot(tran); // citatel = citatelJmenovatel * Y
		float b = cinitel.dot(Ymat);
		float hloubka = a / b; //vypocet z-ove souradnice bodu Y
		Ymat = Ymat * hloubka;
		Y.x = Ymat.at<double>(0,0);
		Y.y = Ymat.at<double>(1,0);
		Y.z = Ymat.at<double>(2,0);
		body3D.push_back(Y);
	}
	return body3D;
}

void vypocetProjekcnichMatic(Mat_<double>  R, Mat_<double>  t, Mat K, Mat &P1, Mat &P2){
	Matx34d Pk1(1.0, 0.0, 0.0, 0.0,
		       0.0, 1.0, 0.0, 0.0,
			   0.0, 0.0, 1.0, 0.0);
	Matx34d Pk2(R(0, 0), R(0, 1), R(0, 2), t(0),
			   R(1, 0), R(1, 1), R(1, 2), t(1),
			   R(2, 0), R(2, 1), R(2, 2), t(2));
	P1 = K*Mat(Pk1);
	P2 = K*Mat(Pk2);
}

void save3Dbody(vector<Point3d> body, String jmeno){
	
	FileStorage fr;
	fr.open("body3D_" + jmeno + ".xml", FileStorage::WRITE);

	fr << "A" << "[";
	for (int n = 0; n < body.size(); n++)
	{
		fr << body[n].x << "," << body[n].y << "," << body[n].z << ";";
		
	}

	fr << "]";
	fr.release();	
}
 
int main()
{
	vector<vector<vector<KeyPoint>>> All_keypoints;
	vector<Mat> All_ffmatice;
	String adresa = "../soubory/fotoKostka/";
	
	vector<String> nazvyFotek;
	int maxPocetFotek = 2;
	nalezeniNazvuFotek(adresa, nazvyFotek, maxPocetFotek);
	vector<vector<DMatch>> all_matches;
	vector<vector<Mat>> all_rotTran;
	vector<vector<Point3f>> all_body3D;
	string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceData0317.xml";
	Mat cameraMatrix;
	Mat distCoeff;
	vector<String> nazvySouboru;
	vector<Mat> calibrateDistor = readXmlCameraMatrix(nazevSouboruXmlKal);
	cameraMatrix = calibrateDistor[0];
	distCoeff = calibrateDistor[1];
	
	cout << "pocet nazvu forek" << nazvyFotek.size() << endl;

	for (int i = 0; i < nazvyFotek.size() - 1; i++)
	{
		// matches
		vector<DMatch> good_matches = paroveKlicoveBody(nazvyFotek[i], nazvyFotek[i + 1], All_keypoints);
		all_matches.push_back(good_matches);
		
	}	
	
	int pocetParufotek = all_matches.size();
	cout << "matchovani bodu dokonceno v poctu" << pocetParufotek << endl;
	 
	for (int i = 0; i < pocetParufotek; i++)
	{
		cout << i << "kalibracni matice: " <<endl << cameraMatrix << endl;
		vector<vector<Point2d>> dvojiceBodu2D = parovaniDodu(all_matches[i], All_keypoints[i][0], All_keypoints[i][1]);
		cout << i << ": vytvoreni dvojic 2D bodu" << endl;
		vector<uchar> status;
		Mat F = findFundamentalMat(dvojiceBodu2D[0], dvojiceBodu2D[1], FM_RANSAC, 0.1, 0.99, status);//fundamentalni matice
		/*Mat H1, H2;
		Size imageSize = Size(5184, 3456);
		stereoRectifyUncalibrated(dvojiceBodu2D[0], dvojiceBodu2D[1], F, imageSize, H1, H2);*/
		cout << i << ": fundamentalni matice" << F << endl;
		
		vector<Mat_<double>> rot12Tran12 = rozkladFMnaRotaciTranslaci(Mat(F), Mat(cameraMatrix));
		cout << i << ": rot12Tran34" << endl;
	
		vector<vector<Point3d>> dvojiceBodu3D;
		
		vector<Point3d> body3D = body3DpomociRT(rot12Tran12[1], -rot12Tran12[2], dvojiceBodu2D, cameraMatrix);//vytvoreni 3D bodu kalibrovanych////////
		
		String jmeno = to_string(1) + "_" + to_string(2) + "000"; //jmeno xml
		save3Dbody(body3D, jmeno); //ukladani xml
		
		
		Mat P1, P2;
		vypocetProjekcnichMatic(rot12Tran12[1], rot12Tran12[2], cameraMatrix, P1, P2);//tvorba projekcnich matice P1 P2
		Mat body4D;
		triangulatePoints(P1, P2, dvojiceBodu2D[0], dvojiceBodu2D[1], body4D);//triangulace bodu
		
		int pocetBod4D = body4D.cols;
		body4D = body4D.t();
		vector<Point3d> body3Dz4D;
		for (int k = 0; k < pocetBod4D; k++)
		{
			float delitel = body4D.at<double>(k, 3);
			Point3f bod3Dz4D;
			bod3Dz4D.x = body4D.at<double>(k, 0) / delitel;
			bod3Dz4D.y = body4D.at<double>(k, 1) / delitel;
			bod3Dz4D.z = body4D.at<double>(k, 2) / delitel;
			body3Dz4D.push_back(bod3Dz4D);
		}
		jmeno = to_string(2) + "_" + to_string(1)+"_4D";
		save3Dbody(body3Dz4D, jmeno);
		
	
	}
	system("PAUSE");
	return 0;
}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     