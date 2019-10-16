#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{

    //The index of every point {e.g Head is 0, Neck is 1, etc}
    map<string, int> points_idx;
    points_idx.insert(pair<string, int>("Head", 0));
    points_idx.insert(pair<string, int>("Neck", 1));
    points_idx.insert(pair<string, int>("RShoulder", 2));
    points_idx.insert(pair<string, int>("RElbow", 3));
    points_idx.insert(pair<string, int>("RWrist", 4));
    points_idx.insert(pair<string, int>("LShoulder", 5));
    points_idx.insert(pair<string, int>("LElbow", 6));
    points_idx.insert(pair<string, int>("LWrist", 7));
    points_idx.insert(pair<string, int>("RHip", 8));
    points_idx.insert(pair<string, int>("RKnee", 9));
    points_idx.insert(pair<string, int>("RAnkle", 10));
    points_idx.insert(pair<string, int>("LHip", 11));
    points_idx.insert(pair<string, int>("LKnee", 12));
    points_idx.insert(pair<string, int>("LAnkle", 13));
    points_idx.insert(pair<string, int>("Chest", 14));
    points_idx.insert(pair<string, int>("Background", 15));

    //Vector containing the links necessary to build the skeleton
    vector<vector<string>> posePairs;
    posePairs.push_back({"Head", "Neck"});
    posePairs.push_back({"Neck", "RShoulder"});
    posePairs.push_back({"RShoulder", "RElbow"});
    posePairs.push_back({"RElbow", "RWrist"});
    posePairs.push_back({"Neck", "LShoulder"});
    posePairs.push_back({"LShoulder", "LElbow"});
    posePairs.push_back({"LElbow", "LWrist"});
    posePairs.push_back({"Neck", "Chest"});
    posePairs.push_back({"Chest", "RHip"});
    posePairs.push_back({"RHip", "RKnee"});
    posePairs.push_back({"RHip", "RKnee"});
    posePairs.push_back({"RKnee", "RAnkle"});
    posePairs.push_back({"Chest", "LHip"});
    posePairs.push_back({"LHip", "LKnee"});
    posePairs.push_back({"LKnee", "LAnkle"});

    // Load our sample image
    Mat image;
    image = imread(samples::findFile("pose.jpg"));

    KeypointsModel model(samples::findFile("openpose_pose_mpi.prototxt"), samples::findFile("openpose_pose_mpi.caffemodel")); // create our model

    // Define the transformations that we need to apply to our image
    Size size{368, 368}; // resize

    // Whether to swap Red and Blue channels since OpenCV loads images in BGR
    bool swapRB = false;

    // rescale the image (image * scale)
    double scale = 1.0/255;
    Scalar mean = Scalar(0, 0, 0); // mean to subtract (e.g: 103.939, 116.779, 123.68)

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    // Network Forward pass
    std::vector<Point> points;
    points = model.estimate(image, 0.5);

    //Draw the estimated points and their index
    for (int i=0; i < points.size(); i++){
        circle(image, Point((int)points[i].x, (int)points[i].y), 8, Scalar(0,255,255), -1);
        putText(image, format("%d", i), Point((int)points[i].x, (int)points[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 1);

    }

    //Display Keypoints
    imshow("Keypoints", image);

    //Draw skeleton
    for (int n = 0; n < posePairs.size(); n++)
    {
        // lookup 2 connected body/hand parts
        Point2f partA = points[points_idx.find(posePairs[n][0])->second];
        Point2f partB = points[points_idx.find(posePairs[n][1])->second];

        if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
            continue;

        line(image, partA, partB, Scalar(0,255,255), 8);
        circle(image, partA, 8, Scalar(0,0,255), -1);
        circle(image, partB, 8, Scalar(0,0,255), -1);
    }

    //Display Keypoints and Skeleton
    imshow("Skeleton", image);
    waitKey(0);

    return 0;
}
