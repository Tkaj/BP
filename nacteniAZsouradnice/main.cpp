#include "opencv2/opencv.hpp"	//video
#include <vector>



using namespace cv;
using namespace std;




bool readXmlvector(string nazevSouboruXmlKal, vector<vector<Point3f>> *objPoints, vector<vector<Point2f>> *imgPoints, Size *imageSize, Mat *cameraMatrix, Mat *distCoeffs, vector<Mat> *rvecs, vector<Mat> *tvecs, vector<String> *nazvySouboru)
{


	int pocet;
	Mat camera;
	Size velikost;
	Mat diCo;
	String nazvySoubor;
	vector<vector<Point2f>> imgPoint;
	Mat objPoint;
	Mat rvec;
	Mat tvec;
	vector<Mat> rvecss;
	vector<Mat> tvecss;


	cout << "oteviram..." << endl;

	FileStorage fr;
	fr.open(nazevSouboruXmlKal, FileStorage::READ);

	fr["pocet"] >> pocet;
	fr["cameraMatrix"] >> camera;
	fr["imageSize"] >> velikost;
	fr["distCoeff"] >> diCo;
	fr["objPoint"] >> objPoint;

	//cout << "obj-konec: " << objPoint<< "\n";

	///////////////////////////////obrazky start
		FileNode pictures = fr["obrazky"];
		FileNodeIterator it = pictures.begin(), it_end = pictures.end();
		// iterate through a sequence using FileNodeIterator
		for (int idx = 0 ; it != it_end; ++it, idx++)
		{
			//cout << "obrazek #" << idx << ": " << "\n";
				//cout << (string)(*it)["nazvySouboru"];
				

			//////////////////////////////imgpoint start
			vector<Point2f> pointip_obr;
			FileNode ip = (*it)["imgPoints"];
			for (FileNodeIterator it_ip = ip.begin(); it_ip != ip.end(); ++it_ip)
			{
				Point2f pointip;
				pointip.x = *it_ip; 
				++it_ip;
				pointip.y = *it_ip; 

				pointip_obr.push_back(pointip);
			}
			//cout << "img point: "<< pointip_obr << "\n";
			imgPoint.push_back(pointip_obr);

			///////////////////////////////imgpoint end
		
			(*it)["rvecs"] >> rvec;
			rvecss.push_back(rvec);
			
			(*it)["tvecs"] >> tvec;
			tvecss.push_back(tvec);
		}
		
	//cout << pocet << endl;
	/*cout << cameraMatrix << endl;
	cout << velikost << endl;
	cout << diCo << endl;
	cout << nazvySoubor << endl;
	cout << rvecss << endl;
	cout << tvecss << endl;*/
	///////////////////////////////obrazky end
	fr.release();


	Mat R1;
	Mat R2;
	Mat P1;
	Mat P2;
	Mat Q;

	stereoRectify(camera, diCo, camera, diCo, Size(640, 480), rvec, tvec, R1, R2, P1, P2, Q);
	reprojectImageTo3D(frame, _3dImage, Q);
	return true;
}


//Poèítá rektifikace transformuje na jedné hlavì kalibrovaného stereo kamerou.
Mat rektifikaceTransformujeStereoKamerou(Mat cameraMatrix1, Mat cameraMatrix2, Mat distCoeffs1, Mat distCoeffs2, Size imageSize, Mat rvecs, Mat tvecs){
	Mat R1;
	Mat R2;
	Mat P1;
	Mat P2;
	Mat Q;
	stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, rvecs, tvecs, R1, R2, P1, P2, Q);
	return Q;
}

//Reprojects nepomìr obrazu do 3D prostoru.
vector<double> triDobraz(Mat frame, InputArray Q){
	vector<double> _3dImage;
	reprojectImageTo3D(frame, _3dImage, Q);
	return _3dImage;
}


int main()
{
	//string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceSave0.xml";
	//string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceSaveKopieUprava.xml";
	string nazevSouboruXmlKal = "../soubory/xmlSoubory/kalibraceDataTest.xml";
	
	vector<vector<Point3f>> objPoints;
	vector<vector<Point2f>> imgPoints;
	Size imageSize;
	Mat cameraMatrix;
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	vector<String> nazvySouboru;

	cout << "spusteno nacteni" << endl;
	system("PAUSE");
	
	if (readXmlvector(nazevSouboruXmlKal, &objPoints, &imgPoints, &imageSize, &cameraMatrix, &distCoeffs, &rvecs, &tvecs, &nazvySouboru))
	{

	cout << "nacteno " << endl;
	system("PAUSE");
	return 0;
	}


	cout << "nenacteno" << endl;
	system("PAUSE");
	
	return 1;

	//----------------------------------------------------------------------------------


}