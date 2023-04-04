///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <unistd.h>
#include <set>
#include <dirent.h>

#include "opencv4/opencv2/video/tracking.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

typedef std::vector<std::string> stringvec;

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void readDirectory(const std::string &name, stringvec &v)
{
	DIR *dirp = opendir(name.c_str());
	if (dirp == NULL)
	{
		printf("%s [IMAGE_PATH_ERROR] %s\n","\033[31m","\033[39m");
	}
	struct dirent *dp;

	while ((dp = readdir(dirp)) != NULL)
	{
		std::string str = std::string(dp->d_name);
		if (str.find(".png") != std::string::npos || str.find(".jpg") != std::string::npos){
			v.push_back(dp->d_name);
		}
	}
	closedir(dirp);
}