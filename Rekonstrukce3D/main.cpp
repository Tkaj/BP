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

vector<vector<Point2d>> parovaniBodu(vector<DMatch> matches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2){ //dvojice klicovych bodu to dvojice souradniv bodu 
	vector<cv::Point2d> points1, points2;
	//1 foto //testovacich 6 bodu pro spravnou velikost
	/*points1.push_back(Point2d(516, 281));
	points1.push_back(Point2d(516, 311));
	points1.push_back(Point2d(516, 403));

	points1.push_back(Point2d(608, 283));
	points1.push_back(Point2d(614, 311));
	points1.push_back(Point2d(614, 403));
	//2 foto //testovacich 6 bodu pro spravnou velikost
	points2.push_back(Point2d(525, 279));
	points2.push_back(Point2d(509, 309));
	points2.push_back(Point2d(509, 401));

	points2.push_back(Point2d(617, 285));
	points2.push_back(Point2d(604, 313));
	points2.push_back(Point2d(608, 408));*/

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
//prevradi Y souradnici tak aby smerovala nahoru
void prevraceniYsouradnice(vector<Point2d>& body,Size sizeImage){
	for (int j = 0; j < body.size(); j++){
		body[j].y = sizeImage.height - body[j].y;
	}
}

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

Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
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

void rozsizeniaPrenasobeniInvK(vector<Point2d> x, Mat cameraMatrix, vector<Point3d>& xK){
	
	for (int j = 0; j < x.size(); j++){
		Point2d kp1 = x[j];
		Point3d u(kp1.x, kp1.y, 1.0);
		Mat_<double> um = cameraMatrix.inv() *Mat_<double>(u);
		u = Point3d(um.at<double>(0, 0), um.at<double>(1, 0), um.at<double>(2, 0));
		xK.push_back(u);
	}
}

void rozkladFMnaRotaciTranslaci(Mat F, Mat K, Mat_<double>& R1, Mat_<double>& R2, Mat_<double>& t){
	Mat E = K.t()*F*K;

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

	double pom = t(0, 0);
	t(0, 0) = t(1, 0);
	t(1, 0) = t(2, 0);
	t(2, 0) = pom;

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

void vypocetProjekcnichMatic(Mat_<double>  R, Mat_<double>  t, Mat_<double> F, Matx34d &P1, Matx34d &P2){
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

void save3Dbody(vector<Point3d> body, String jmeno,String cestaXml){
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
	vector<vector<vector<KeyPoint>>> All_keypoints;
	char* adresaChar = "../soubory/fotoKostka/";
	//char* adresaChar = "../soubory/sachovnice/";
	String cestaXml = "../soubory/body3D/";
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
	Mat K, distCoeffs;
	Size  sizeImage;
	//vector<String> nazvySouboru;
	readXmlCameraMatrix(nazevSouboruXmlKal, K, distCoeffs, sizeImage);
	
	//-------------------hledani-bodu-a-jejich-korespondencnich-bodu---------------------------------------------------
	cout << "pocet nazvu forek " << nazvyFotek.size() << endl;
	for (int i = 0; i < nazvyFotek.size() - 1; i++)
	{
		// matches
		vector<DMatch> good_matches = paroveKlicoveBody(nazvyFotek[i], nazvyFotek[i + 1], All_keypoints);
		all_matches.push_back(good_matches);
	}
	//---------------------3D-rekonstrukce----------------------------------------------------------------------------
	int pocetParufotek = all_matches.size();
	cout << "matchovani bodu dokonceno v poctu" << pocetParufotek << endl;
	 
	for (int i = 0; i < pocetParufotek; i++)
	{
		cout << i + 1 << ". z " << pocetParufotek << " dvojic fotek"<< endl;
		vector<vector<Point2d>> dvojiceBodu2D = parovaniBodu(all_matches[i], All_keypoints[i][0], All_keypoints[i][1]);//parovani dvojic
		cout << i+1 << ": vytvoreni dvojic 2D bodu" << endl;

		//prevraceniYsouradnice(dvojiceBodu2D[0], sizeImage);// (x1,sizeImage.height - y1)
		//prevraceniYsouradnice(dvojiceBodu2D[1], sizeImage);// (x2,sizeImage.height - y2)
		cout << i + 1 << ": prevraceni Y souradnice" << endl;

		vector<uchar> status;
		Mat F = findFundamentalMat(dvojiceBodu2D[0], dvojiceBodu2D[1], FM_RANSAC, 0.1, 0.99, status);//(x1,y1),(x2,y2) => fundamentalni matice
		cout << i + 1 << ": vypoctena F matrix" << endl;
		
		Mat_<double> R1, R2, t;
		rozkladFMnaRotaciTranslaci(Mat(F), Mat(K), R1, R2, t); //F => R,t 
		cout << i + 1 << ": R1, R2, t" << endl;
		cout << i + 1 << ": R1" << R1 << endl;
		cout << i + 1 << ": t" << t << endl;
		vector<Point3d> x1K, x2K;
		rozsizeniaPrenasobeniInvK(dvojiceBodu2D[0], K, x1K);  //(x1,y1,1)*Kinv
		rozsizeniaPrenasobeniInvK(dvojiceBodu2D[1], K, x2K); //(x2,y2,1)*Kinv
		cout << i + 1 << ": x*K" << endl;
		
		Matx34d P1, P2;
		vypocetProjekcnichMatic(R2, t ,F ,P1 , P2);//tvorba projekcnich matice P1 P2
		cout << i + 1 << ": P1, P2" << endl;
		cout << i + 1 << ": t" << t << endl;
		cout << i + 1 << ": P1" << P1 << endl;
		cout << i + 1 << ": P2" << P2 << endl;
		vector<Point3d> body3D;
		for (int k = 0; k < x1K.size(); k++){
			Mat_<double> X = LinearLSTriangulation(x1K[k], P1, x2K[k], P2); //vypocet 3D bodu
			body3D.push_back(Point3d(X(0), X(1), X(2)));
		}
		cout << i + 1 << ": 3D body" << endl;

		String jmeno = "kostka_plus";
		save3Dbody(body3D, jmeno,cestaXml);		
	}
	system("PAUSE");
	return 0;
}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     