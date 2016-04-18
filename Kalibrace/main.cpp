#include "opencv2/opencv.hpp"	//video
#include <dirent.h>
#include <ctime>
#include <iostream>

using namespace cv;
using namespace std;

const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);

	strftime(buf, sizeof(buf), "%Y%m%d%H%M", &tstruct);

	return buf;
}

int nalezNazVsechFot(char* adresa, vector<String> &nazvyFotek){

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(adresa)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			String filename = ent->d_name;
			if ((filename.find(".JPG") != string::npos) || (filename.find(".jpg") != string::npos)){
				nazvyFotek.push_back(adresa + filename);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return 1;
	}

	if (nazvyFotek.size() == 0){ //pokud je pole NazvuFotek prazdne opust fukci
		return 1;
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
		cout << "fotka " << i+1 <<": ";
		if (findChessboardCorners(frame, boardSize, centers, CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH)){
			//drawChessboardCorners(frame, boardSize, centers, true);
			//imshow("Image View", frame);
			cout << "sachovnice nalezena" << endl;
			cvWaitKey(33);
			allCenters.push_back(centers);
		}
		else{
			cout << "sachovnice NEnalezena" << endl;
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

bool saveData(Size imageSize, Mat cameraMatrix, Mat distCoeffs,string jmenoXml)
{
	int cislo = 0;
	/*
	while (true)
	{
	j

	FileStorage fe;
	if (!fe.open(jmenoXml, FileStorage::READ)) break;
	cislo++;
	}*/

	

	//char *intStr = itoa(datuny) ;
	//string str = string(intStr);
	jmenoXml = jmenoXml + "Kalibrace" + currentDateTime() + ".xml";
	// ----------------------zapis kalibrace do xml
	//int size = objPoints.size();
	//------------------------------------------------pretipovat vectorToMat
	/*cv::vector < Point3f > components = objPoints[0];
	cv::Mat objP(components, true);*/
	//-------------------------------------------------pretipovat end

	FileStorage fs(jmenoXml, FileStorage::WRITE);
	//fs << "pocet" << size;
	fs << "cameraMatrix" << cameraMatrix;
	fs << "imageSize" << imageSize;	
	fs << "distCoeff" << distCoeffs;
	/*fs << "objPoint" << objP;
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
	fs << "]";*/
	cout << "nahrani kalibrace dokoncena " << endl;
	fs.release();
	return true;
}

bool calibrate(vector<vector<Point3f>> objPoints, vector<vector<Point2f>> imgPoints, Size imageSize, string nazevXml)
{
	Mat distCoeffs;
	Mat cameraMatrix;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	
	double reprErr = calibrateCamera(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
	cout << "Reprojected error value: " << reprErr << endl;
	//system("PAUSE");
	int pocet = objPoints.size();
	for(int i = 0; i < pocet; i++)
	{
		projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imgPoints[i]);
	}
	if (saveData(imageSize, cameraMatrix, distCoeffs, nazevXml)){
		//system("PAUSE");
		return true;
	}
	cout << "nezapsalo " << endl;
	//system("PAUSE");
	return false;
}

int main()
{
	char* adresaChar = "../soubory/sachovnice/";
	String jmenoXml = "../soubory/xmlSoubory/";
	//String adresa = "../soubory/fotoCameraGrey/";

	vector<String> nazvyFotek;
	if (nalezNazVsechFot(adresaChar, nazvyFotek) != 1){
		cout << "nalezeni " << nazvyFotek.size() << " fotek" << endl;
	}
	else{
		cout << "zadna fotka v adresari nebyla nenalezena" << endl;
		system("PAUSE");
		return 5;
	}
	vector<vector<Point2f> > image_points; // inicializace pro body ze SACHOVNIC 2D
	Size boardSize = Size(7, 5);  // velikost sachovnice
	Size imageSize = Size(1920, 1080);//Size(5184, 3456); // velikost vstupnich kalibracnich obrazku (640, 480)
	vector<vector<Point3f>> obj_Points; // object points 3D
	//vector<String> nazvySouboru; // nazvy Souboru fotek
	//----------------------------------------------------------------------------------
	
	cout << "zpusteni hledani bodu na sachovnici" << endl;
	if (chessboardKontrol(boardSize, image_points, nazvyFotek)){
		cout << "vsechny body nalezeny " << endl;
		// Vytvori 3D synteticke body
		obj_Points = tvorbaUmelychBodu(boardSize, 3.0f, image_points.size());
		cout << "tvorba umelych bodu (sachovnic) dokoncena " << endl;
		
		// provede kalibraci a ulozi data
		if (calibrate(obj_Points, image_points, imageSize, jmenoXml)) {
			cout << endl << "kalibrace dokonceny." << endl;
			
			return 0;
		}
	}
	else{
		cout << "nenalezeny zadne body sachovnice " << endl;
		return 5;
	}
	
	return 1;
}

