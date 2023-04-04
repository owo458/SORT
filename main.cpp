#include "Hungarian.h"
#include "KalmanTracker.h"
#include "main.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	std::string rawPath;
	std::string dataPath;
	std::string resultPath;

	unsigned int startFrame = 0;
	unsigned int isStop = 0;
	float imageWidth = 1;
	
	if (argc < 2){
		printf("%sCHECK ReadMe.md or ./bin/ld --help %s\n", "\033[31m", "\033[39m");
		return 0;
	}
	else{
		for (int i = 1; i < argc; i += 2){
			
			std::string str = std::string(argv[i]);
			
			if(str.compare("--raw") == 0)
			{
				rawPath = std::string(argv[i+1]);
			}
			else if(str.compare("--detData") == 0)
			{
				dataPath = std::string(argv[i+1]);
			}
			else if(str.compare("--result") == 0)
			{
				resultPath = std::string(argv[i+1]);
			}
			else if(str.compare("--start") == 0)
			{
				startFrame = atoi(argv[i+1]);
			}
			else
			{
				printf("%sERROR!! ./bin/ld --help%s\n", "\033[31m", "\033[39m");	
				return -1;
			}	
		}
	}

	printf("%sSTART      FRAME  : %u %s\n", "\033[32m", startFrame, "\033[39m");
	printf("%sRAW        PATH   : %s %s\n", "\033[32m", rawPath.c_str(), "\033[39m");
	printf("%sDETDATA    PATH   : %s %s\n", "\033[32m", dataPath.c_str(), "\033[39m");
	printf("%sRESULT     PATH   : %s %s\n", "\033[32m", resultPath.c_str(), "\033[39m");
	
	/* randomly generate colors, only for display */ 
	int colorMapSize = 20;
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[colorMapSize];
	for (int i = 0; i < colorMapSize; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	/* Load image */
	unsigned int index = startFrame;
	stringvec imgList;	
	readDirectory(rawPath, imgList);
	sort(imgList.begin(), imgList.end());

	/* Load detection data */
	ifstream detectionFile;
	detectionFile.open(dataPath);

	if (!detectionFile.is_open())
	{
		printf("%s [DATA_PATH_ERROR] %s\n","\033[31m","\033[39m");
		return 0;
	}

	string detLine;
	istringstream ss;
	vector<TrackingBox> detData;
	char ch;
	float tpx, tpy, tpw, tph;

	while (getline(detectionFile, detLine) )
	{
		TrackingBox tb;

		ss.str(detLine);
		ss >> tb.frame >> ch >> tb.id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
		ss.str("");

		tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
		detData.push_back(tb);
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detData) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	vector<vector<TrackingBox>> detFrameData;
	vector<TrackingBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detData)
			if (tb.frame == fi + 1) // frame num starts from 1
				tempVec.push_back(tb);
		detFrameData.push_back(tempVec);
		tempVec.clear();
	}

	// 3. update across frames
	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;
	vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

	/* Result file Save */
	ofstream resultsFile;
	string resFileName = resultPath +"test.txt";
	resultsFile.open(resFileName);

	if (!resultsFile.is_open())
	{
		printf("%s [RRESULT_PATH_ERROR] %s\n","\033[31m","\033[39m");
		return 0 ;
	}

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	int idx = 0;

	/* Tracking variables */
	vector<TrackingBox> detectionData;

	/* Main loop */
	for (int fi = 0; fi < maxFrame; fi++)
	//while(1)
	{
		cv::Mat raw;
		raw = imread(rawPath + imgList[idx], IMREAD_COLOR);
		//imshow("test!!!",raw);

		/* 처음이라서 Tracking이 동작하지 않음 */
		if (trackers.size() == 0)
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detFrameData[fi].size(); i++)
			{
				KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
				trackers.push_back(trk);
			}
			// output the first frame detections
			for (unsigned int id = 0; id < detFrameData[fi].size(); id++)
			{
				TrackingBox tb = detFrameData[fi][id];
				resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			}
			continue;
		}

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = detFrameData[fi].size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
				cv::rectangle(raw, detFrameData[fi][j].box, Scalar(255,0,0), 2, 8, 0);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detFrameData[fi][detIdx].box);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
			trackers.push_back(tracker);
		}

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}

		for (auto tb : frameTrackingResult)
			resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;

		//ostringstream oss;
		//oss << rawPath << setw(6) << setfill('0') << fi + 1;
		//Mat img = imread(oss.str() + ".jpg");
		if (raw.empty())
			continue;
		
		for (auto tb : frameTrackingResult)
			cv::rectangle(raw, tb.box, randColor[tb.id % colorMapSize], 2, 8, 0);
		imshow("test", raw);
		waitKey(0);
		idx++;
	}

	resultsFile.close();
	
	// destroyAllWindows();

	//vector<string> sequences = { "PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2" };
	// for (auto seq : sequences)
	// 	TestSORT(seq, false);

	//AlgorithmSORT(rawPath, dataPath,resultPath,true);

	
	// Note: time counted here is of tracking procedure, while the running speed bottleneck is opening and parsing detectionFile.
	// cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or " << ((double)total_frames / (double)total_time) << " FPS" << endl;

	return 0;
}