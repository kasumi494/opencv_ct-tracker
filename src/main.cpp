#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tracker.h"

typedef std::list< std::string > ListOfFilenames;

ListOfFilenames loadListOfFilenames (const std::string& fName)
{
    ListOfFilenames imageFilenameList;
    std::ifstream input (fName.c_str(), std::ios_base::in);

    if (input)
    {
        std::string line;
        while (getline (input, line))
            imageFilenameList.push_back (line);

        input.close();
    }

    return imageFilenameList;
}

int main (int argc, char* argv[])
{
    if (argc != 6)
    {
        std::cout << "Usage: <image list> <x> <y> <width> <height>" << std::endl;
        std::cout << "- image list - text file containing list of image file names" << std::endl;
        std::cout << "- x, y - coordinates of top-left corner of object's bounding rectangle (integer)" << std::endl;
        std::cout << "- width, height - size of object's bounding rectangle (integer)" << std::endl;
        return -1;
    }

    cv::Rect objectPosition (
        atoi( argv[ 2 ] ), // X coordinate
        atoi( argv[ 3 ] ), // Y coordinate
        atoi( argv[ 4 ] ) - atoi( argv[ 2 ] ), // width
        atoi( argv[ 5 ] ) - atoi( argv[ 3 ] ) // height
    );

    ListOfFilenames imageFilenameList = loadListOfFilenames (argv[1]);

    if (imageFilenameList.empty())
    {
        std::cerr << "Can't load image list from file " << argv[1] << std::endl;
        return -1;
    }

    Tracker tracker;
    cv::Mat startImage = cv::imread (imageFilenameList.front(), 0);

    if (!startImage.data)
    {
        std::cerr << "Can't load first image" << std::endl;
        return -1;
    }

    tracker.Init (startImage, objectPosition);
    cv::Mat gray;
    double sum_time = 0.0;
    int count = 0;
    char fps[20];

    while (!imageFilenameList.empty())
    {
        cv::Mat currentImage = cv::imread (imageFilenameList.front());
        imageFilenameList.pop_front();

        if (currentImage.data)
        {
            cvtColor (currentImage, gray, CV_RGB2GRAY);

            double t = static_cast <double> (cv::getTickCount ());
            objectPosition = tracker.TrackObject (gray);
            t = static_cast <double> (cv::getTickCount () - t) / cv::getTickFrequency();

            sum_time += t;

            if (sum_time >= 1.0)
            {
                sprintf  (fps, "FPS: %d", count);

                sum_time = .0;
                count = 0;
            }

            count++;
            std::cout << objectPosition.x << ", " << objectPosition.y << ", ";
            std::cout << objectPosition.width  + objectPosition.x << ", "
                      << objectPosition.height + objectPosition.y << std::endl;

            tracker.DrawObject (currentImage);
            cv::putText (currentImage, fps, cv::Point (10, currentImage.rows - 10),
                         cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar (0, 255, 255));

            imshow ("Tracker", currentImage);
            if (cvWaitKey(2) == 'q') {	break; }
        }
    }

    return 0;
}
