#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{

    // Load our sample image
    Mat image = imread(samples::findFile("ladybug.jpg"));

    ClassificationModel model(samples::findFile("alexnet.onnx")); // create our model

    // Define the transformations that we need to apply to our image
    Size size{227, 227}; // resize


    // Whether to swap Red and Blue channels since OpenCV loads images in BGR
    bool swapRB = false;

    // rescale the image (image * scale)
    double scale = 1.0;
    Scalar mean = Scalar(103.939, 116.779, 123.68); // mean to subtract to each channel

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    // Network Forward pass
    std::pair<int, float> prediction = model.classify(image);

    std::string pred = "Prediction " + std::to_string(prediction.first);
    std::string confidence = " Confidence " + std::to_string(prediction.second);

    // It shows 301 which is the class for ladybug in ImageNet
    imshow(pred + confidence, image);
    waitKey(0);

    return 0;
}
