#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using std::cout;
using std::vector;
using std::string;
using namespace std;
using namespace cv;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

Ptr<Feature2D> featureDetector(string type) {

    if (type == "brisk") {
        Ptr<BRISK> brisk = BRISK::create();
        return brisk;
    }
    if (type == "akaze") {
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->setThreshold(akaze_thresh);
        return akaze;
    }
    else{// orb by default
        Ptr<ORB> orb = ORB::create();
        return orb;
    }
}

namespace example {
    class Tracker
    {
    public:
        Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
            detector(_detector),
            matcher(_matcher)
        {}

        void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
        Mat process(const Mat frame, Stats& stats);
        Ptr<Feature2D> getDetector() {
            return detector;
        }
    protected:
        Ptr<Feature2D> detector;
        Ptr<DescriptorMatcher> matcher;
        Mat first_frame, first_desc;
        vector<KeyPoint> first_kp;
        vector<Point2f> object_bb;
    };

    void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
    {
        cv::Point* ptMask = new cv::Point[ bb.size()];
        const Point* ptContain = { &ptMask[0] };
        int iSize = static_cast<int>(bb.size());
        for (size_t i = 0; i < bb.size(); i++) {
            ptMask[i].x = static_cast<int>(bb[i].x);
            ptMask[i].y = static_cast<int>(bb[i].y);
        }
        first_frame = frame.clone();
        cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
        detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
        stats.keypoints = (int)first_kp.size();
        drawBoundingBox(first_frame, bb);
        putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
        object_bb = bb;
        delete[] ptMask;
    }

    Mat Tracker::process(const Mat frame, Stats& stats)
    { 
        TickMeter tm;
        vector<KeyPoint> kp;
        Mat desc;

        tm.start();
        detector->detectAndCompute(frame, noArray(), kp, desc);
        stats.keypoints = (int)kp.size();

        vector< vector<DMatch> > matches;
        vector<KeyPoint> matched1, matched2;
        matcher->knnMatch(first_desc, desc, matches, 2);
        for (unsigned i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
                matched1.push_back(first_kp[matches[i][0].queryIdx]);
                matched2.push_back(kp[matches[i][0].trainIdx]);
            }
        }
        stats.matches = (int)matched1.size();

        Mat inlier_mask, homography;
        vector<KeyPoint> inliers1, inliers2;
        vector<DMatch> inlier_matches;
        if (matched1.size() >= 4) {
            homography = findHomography(Points(matched1), Points(matched2),
                RANSAC, ransac_thresh, inlier_mask);
        }
        tm.stop();
        stats.fps = 1. / tm.getTimeSec();

        if (matched1.size() < 4 || homography.empty()) {
            Mat res;
            hconcat(first_frame, frame, res);
            stats.inliers = 0;
            stats.ratio = 0;
            return res;
        }
        for (unsigned i = 0; i < matched1.size(); i++) {
            if (inlier_mask.at<uchar>(i)) {
                int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);
                inlier_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }
        stats.inliers = (int)inliers1.size();
        stats.ratio = stats.inliers * 1.0 / stats.matches;

        vector<Point2f> new_bb;
        perspectiveTransform(object_bb, new_bb, homography);
        Mat frame_with_bb = frame.clone();
        if (stats.inliers >= bb_min_inliers) {
            drawBoundingBox(frame_with_bb, new_bb);
        }
        Mat res;
        drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
            inlier_matches, res,
            Scalar(255, 0, 0), Scalar(255, 0, 0));
        return res;
    }
}

const char* keys =
"{@input_path | 0           |input path can be a camera id, like 0,1,2 or a video filename}"
"{ input1     | orb         | Feature Detector. }"
"{ input2     | blur        | Gaussian Blur }";

int main(int argc, char** argv){

    cout << "Type: FeatureDescriptor, blur ";
    CommandLineParser parser(argc, argv, keys);
    string input_path = parser.get<string>(0);
 

    string desc_type(parser.get<cv::String>("input1"));
    string blur(parser.get<cv::String>("input2"));
    string video_name = desc_type + " " + blur;
    VideoCapture video_in;

    if ((isdigit(input_path[0]) && input_path.size() == 1))
    {
        int camera_no = input_path[0] - '0';
        video_in.open(camera_no);
    }
    else {
        video_in.open(video_name);
    }

    if (!video_in.isOpened()) {
        cerr << "Couldn't open " << video_name << endl;
        return 1;
    }

    Stats stats, detector_stats;
    Ptr<Feature2D> detector = featureDetector(desc_type);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    example::Tracker tracker(detector, matcher);

    Mat frame, frame_blurred;
    String window_name_of_video_blurred_with_5x5_kernel = "Video Blurred with 5 x 5 Gaussian Kernel";
    //namedWindow(window_name_of_video_blurred_with_5x5_kernel, WINDOW_NORMAL);
    namedWindow(video_name, WINDOW_NORMAL);
    cout << "\nPress any key to stop the video and select a bounding box" << endl;

    while (waitKey(1) < 1){
        video_in >> frame;
        cv::resizeWindow(video_name, frame.size());
        if (blur == "blur") {
            GaussianBlur(frame, frame_blurred, Size(5, 5), 0);
            frame = frame_blurred;
        }

        imshow(video_name, frame);
    }
    vector<Point2f> bb;
    cv::Rect uBox = cv::selectROI(video_name, frame);
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

    tracker.setFirstFrame(frame, bb, desc_type, stats);
    Stats draw_stats;
    Mat res;
    int i = 0;
    for (;;) {
        i++;
        bool update_stats = (i % stats_update_period == 0);
        if (blur == "blur") {
            GaussianBlur(frame, frame_blurred, Size(5, 5), 1);
            frame = frame_blurred;
        }
        video_in >> frame;
        // stop the program if no more images
        if (frame.empty()) break;

        res = tracker.process(frame, stats);
        detector_stats += stats;
        if (update_stats) {
            draw_stats = stats;
        }
        drawStatistics(res, draw_stats);
        cv::imshow(video_name, res);

        if (waitKey(1) == 27) break; //quit on ESC button
    }
    stats /= i - 1;
    printStatistics(desc_type, stats);
    return 0;
}