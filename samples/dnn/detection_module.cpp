#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace dnn;

int main()
{

    // Load our sample image
    Mat image = imread(samples::findFile("dog416.png"));

    //Load our model
    std::string weights_file = samples::findFile("yolov3.weights");
    std::string config_file = samples::findFile("yolov3.cfg");
    DetectionModel model(weights_file, config_file);

    // Define the transformations that we need to apply to our image
    Size size{416, 416}; // resize

    // Whether to swap Red and Blue channels since OpenCV loads images in BGR
    bool swapRB = true;

    // rescale the image (image * scale)
    double scale = 1.0 / 255.0;

    // Discard predictions under confThreshold confidence
    float confThreshold = 0.8;
    // Discard boxes with an IOU less than nmsThreshold
    double nmsThreshold = 0.0;

    Scalar mean = Scalar(127.5, 127.5, 127.5); // mean to subtract to each channel

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    // Vectors to store the predictions
    std::vector<Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    // Forward Pass
    model.detect(image, classIds, confidences, boxes, confThreshold, nmsThreshold);

    // Iterate over every predicted box and draw them on the image with the predicted class and confidence on top
    std::vector<Rect2d> boxesDouble(boxes.size());
    for (int i = 0; i < boxes.size(); i++) {
        boxesDouble[i] = boxes[i];
        rectangle(image, boxesDouble[i], Scalar(0, 0, 255), 1, 8, 0);
        std::string text = std::to_string(classIds[i]) + ": " + std::to_string(confidences[i]);
        putText(image, text, Point(boxes[i].x, boxes[i].y), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,0,0), 2);
    }

    // Show results
    imshow("Detections", image);
    waitKey(0);

    return 0;
}
