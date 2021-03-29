#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using std::cout;
using std::endl;

const char* keys =
"{ help h |                  | Print help message. }"
"{ input1 | box.png          | Path to input image 1. }"
"{ input2 | box_in_scene.png | Path to input image 2. }";

int main(int argc, char* argv[]){
    CommandLineParser parser(argc, argv, keys);
    Mat img_object = imread(samples::findFile(parser.get<String>("input1")), IMREAD_GRAYSCALE);
    Mat img_scene = imread(samples::findFile(parser.get<String>("input2")), IMREAD_GRAYSCALE);
    if (img_object.empty() || img_scene.empty()){
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using AKAZE and ORB Detector
    Ptr<AKAZE> detector = AKAZE::create();
    //TODO add a selector to determine the desired detector
    //Ptr<ORB> detector = ORB::create();
    //Ptr<BRISK> detector = BRISK::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors1);
    detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors2);
    //-- Step 2: Matching descriptor vectors with a brute force matcher
    //TODO test diferent matchers 
    BFMatcher matcher(NORM_HAMMING);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    //-- Step 3: Filter matches using the Lowe's ratio test
    // TODO 1 cross check test and 2 geometric test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    //without filter
    //Mat img_matches;
    //drawMatches(img1, keypoints_object, img2, keypoints_scene, knn_matches, img_matches);
    //with lowe's filter
    Mat img_good_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_good_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++){
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    Mat H = findHomography(obj, scene, RANSAC);
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)img_object.cols, 0);
    obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
    obj_corners[3] = Point2f(0, (float)img_object.rows);
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_good_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
        scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_good_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
        scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_good_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
        scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_good_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
        scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);


    //-- Show detected matches
   //imshow("No_filter", img_matches);
    imshow("Filter", img_good_matches);
    waitKey();
    return 0;
}