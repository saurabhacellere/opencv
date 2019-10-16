#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{

    // Load our sample image
    Mat image;
    image = imread(samples::findFile("style_sample.jpg"));

    GenerationModel model(samples::findFile("fast_style.onnx")); // create our model

    // Define the transformations that we need to apply to our image
    Size size{512, 512}; // resize

    // Whether to swap Red and Blue channels since OpenCV loads images in BGR
    bool swapRB = true;

    // rescale the image (image * scale)
    double scale = 1.0;
    Scalar mean = Scalar(); // mean to subtract (e.g: 103.939, 116.779, 123.68)

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    // Network Forward pass
    Mat out;
    model.generate(image, out);

    //Display Image
    imshow("styled_image", out);
    waitKey(0);

    return 0;
}
