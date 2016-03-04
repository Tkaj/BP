#include "opencv2/opencv.hpp"	//video

using namespace cv;
using namespace std;

int nalezeniNazvuFotek(String adresa, vector<String> &nazvyFotek)
{
	String nazevSouboru;
	Mat frame; // obrazek
	int citac = 0; // citac pro vstupni soubory
	while (true) {
		
		nazevSouboru = adresa + "FCG" + to_string(citac) + ".png";
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

bool chessboardKontrol(Size boardSize, vector<vector<Point2f>> &allCenters, vector<String> nazvyFotek)
{
	cout << "tvorba sachovnic" <<endl;
	vector<Point2f> centers; // inicializace pro body ze SACHOVNICE
	Mat frame; // obrazek

	int pocet = nazvyFotek.size();
	//-------------------------------------------------------------------------------------
	for (int i = 0; i < pocet; i++)
	{
		frame = imread(nazvyFotek[i], CV_LOAD_IMAGE_GRAYSCALE); // nacteni fotky ve stupni sedi
		// findChessboardCorners funkce ktera vraci bool - proto muze byt pouzita v ifu na vyhodnocivani
		// CALIB_CB_FAST_CHECK - udela rychlou kontrolu jestli tam sachovnice o dane velikosti je
		// CALIB_CB_ADAPTIVE_THRESH - pouzije adaptivni prahovani na hledani vrcholu sachovnice
		cout << ":" << i;
		if (findChessboardCorners(frame, boardSize, centers, CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH)){
			//drawChessboardCorners(frame, boardSize, centers, true);
			//imshow("Image View", frame);
			cout << "nalezena sachovnice fotky:" << i << endl;
			cvWaitKey(33);
			allCenters.push_back(centers);
		}
	}
	return true;
}

vector<vector<Point3f>> tvorbaUmelychBodu(Size boardSize, float borderLength, int nImg)
{
	vector<vector<Point3f>> umeleBody;
	for (int k = 0; k < nImg; ++k) {
		vector<Point3f> umelyB;

		for (int i = 0; i < boardSize.height; i++) {
			for (int j = 0; j < boardSize.width; j++) {
				umelyB.push_back(Point3f((1.0f + i) * borderLength, (1.0f + j) * borderLength, 0.0f));
			}
		}

		umeleBody.push_back(umelyB);
	}
	return  umeleBody;
}

bool saveData(vector<vector<Point3f>> objPoints, vector<vector<Point2f>> imgPoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, vector<Mat> rvecs, vector<Mat> tvecs, vector<String> &nazvySouboru)
{

	// nalezeni nazvu 

	int cislo = 0;
	string jmenoXml;
	/*
	while (true)
	{
	jmenoXml.clear();
	jmenoXml.append("kalibraceSave");
	jmenoXml.append(std::to_string(cislo));
	jmenoXml.append(".xml");

	FileStorage fe;
	if (!fe.open(jmenoXml, FileStorage::READ)) break;
	cislo++;
	}*/
	jmenoXml = "kalibraceDataTest.xml";

	// ----------------------zapis kalibrace do xml
	int size = objPoints.size();
	//------------------------------------------------pretipovat vectorToMat
	cv::vector < Point3f > components = objPoints[0];
	cv::Mat objP(components, true);
	//-------------------------------------------------pretipovat end

	FileStorage fs(jmenoXml, FileStorage::WRITE);
	fs << "pocet" << size;
	fs << "cameraMatrix" << cameraMatrix;
	fs << "imageSize" << imageSize;	
	fs << "distCoeff" << distCoeffs;
	fs << "objPoint" << objP;
	fs << "obrazky" << "[";
	for (int i = 0; i != size; i++)
	{
		cout << "ulkaldani: " << nazvySouboru[i] << endl;
		fs << "{:";
		fs << "nazvySouboru" << nazvySouboru[i];
		fs << "imgPoints" << imgPoints[i];
		fs << "rvecs" << rvecs[i];
		fs << "tvecs" << tvecs[i];
		fs << "}";
	}
	fs << "]";
	cout << "nahrani kalibrace dokoncena " << endl;
	fs.release();
	return true;
}

bool calibrate(vector<vector<Point3f>> objPoints, vector<vector<Point2f>> imgPoints, Size imageSize, vector<String> &nazvySouboru)
{
	Mat distCoeffs;
	Mat cameraMatrix;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	
	double reprErr = calibrateCamera(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
	cout << "Reprojected error value: " << reprErr << endl;
	system("PAUSE");
	int pocet = objPoints.size();
	for(int i = 0; i < pocet; i++)
	{
		projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imgPoints[i]);
	}
	if (saveData(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, nazvySouboru)){
		system("PAUSE");
		return true;
	}
	cout << "nezapsalo " << endl;
	system("PAUSE");
	return false;
}

int main()
{
	//String adresa = "../soubory/kalibrace/";
	String adresa = "../soubory/fotoCameraGrey/";
	vector<String> nazvyFotek;
	vector<vector<Point2f> > image_points; // inicializace pro body ze SACHOVNIC 2D
	Size boardSize = Size(7, 5);  // velikost sachovnice
	Size imageSize = Size(5184, 3456); // velikost vstupnich kalibracnich obrazku (640, 480)
	vector<vector<Point3f>> obj_Points; // object points 3D
	//vector<String> nazvySouboru; // nazvy Souboru fotek
	//----------------------------------------------------------------------------------
	nalezeniNazvuFotek(adresa, nazvyFotek);

	if (chessboardKontrol(boardSize, image_points, nazvyFotek)){

		// Vytvori 3D synteticke body
		obj_Points = tvorbaUmelychBodu(boardSize, 3.0f, image_points.size());
		cout << endl << "tvorba umelych bodu (sachovnic) dokoncena " << endl;
		
		// provede kalibraci a ulozi data
		if (calibrate(obj_Points, image_points, imageSize, nazvyFotek)) {
			cout << endl << "Všechno je hotove." << endl;
			
			return 0;
		}
	}
	
	return 1;
}

