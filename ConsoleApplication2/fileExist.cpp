#include <string>
#include <fstream>

bool fileExists(const std::string& name) {
	std::ifstream file(name.c_str());
	if (file.good()) {
		file.close();
		return true;
	}
	else {
		file.close();
		return false;
	}
}

