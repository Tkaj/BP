#include "opencv2/opencv.hpp"	//video
#include <stdio.h>
#include <string>
#include "fileExist.h"

using namespace cv;
using namespace std;

int main()
{
	Size boardSize = Size(7, 5);
	vector<Point2f> centers;
	namedWindow("video", 1);
	VideoCapture cap(0); // open the default camera
	int vyskedek = 1;

	cv::Mat workFoto;

	std::cout << "zmaèknutim klavesi ESC opustit program " << std::endl
		<< "zmaèknutim klavesi f vyfotim a uložím fotografii" << std::endl;
	//-------------------------------ZJISTOVANI JMENA A ADRASY-----------------------------//

	std::string adresar = "FCG"; // jmeno adresare pro fotky //FCG /FCG
	int cislofotky = 0; // inicializace promenne pro zjisteni cisla fotky
	// zjistovani cisla fotky pro naslede ukladani
	bool hledani = false;
	std::string fotoName;

	while (true)
	{
		fotoName = adresar;
		fotoName.append(std::to_string(cislofotky));
		fotoName.append(".png");
		hledani = fileExists(fotoName);
		
		if (!hledani) break;
		cislofotky++;
	}

	std::cout << "aktualni pocet fotek ve slozce je: " << cislofotky << "\n";
	//-------------------------------FOCENI A UKLADANI-----------------------------//

	unsigned int counter = cislofotky; // nacitaci pro 0, 1, 2, 3... pro tvorbu nazvu souboru
	Mat finFoto; //inicializace pro vystupmi sedotonovou fotku
	Mat frame; //inicializace pro nacteny obrazek z kamery 
	string nazev; //nazev nahravaneho souboru fotky

	while (true)
	{
		cap >> frame; // nacteny novy obrazek z kamery		
		if (findChessboardCorners(frame, boardSize, centers, CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH)){

			drawChessboardCorners(frame, boardSize, centers, true);
			imshow("Image View", frame);
			cvWaitKey(33);
		}

		workFoto = frame;  // obrazek z kamery ulozeni do pracovni promenne workFoto
		imshow("video", workFoto); //v okne nacteny aktualni workFoto

		char key = waitKey(30);	// cekej na klavesu

		if (key >= 0){			// cekej na klavesu 
			if (key == 27)	// pokud je to "esc" tak ukonci program, jinak se fotka ulozi 
			{
				std::cout << "konec foceni " << std::endl;
				break;
			}
			
			else
			{
				// jmeno souboru fcg + cislo poradi

				nazev.clear();
				nazev.append(adresar);
				//nazev.append("/FCG"); // jmeno adresare pro fotky //FCG
				nazev.append(std::to_string(counter));
				nazev.append(".png");

				counter++; // nacitani pro ciselne pojmenovani dalsiho souboru
				
				cvtColor(workFoto, finFoto, CV_BGR2GRAY, 0); // prevedeni do sedotunu
			
				imwrite(nazev, finFoto); // zapisovani souboru na adresu NAZEV
				
				std::cout << "ulozena fotka: " << nazev << "\n";
			}
			
		}
	}
	return 0;
}

