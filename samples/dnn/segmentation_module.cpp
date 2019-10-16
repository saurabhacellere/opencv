#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/imgproc.hpp"

#include "common.hpp"

using namespace cv;
using namespace dnn;

int main()
{

    // Load our sample image
    Mat image = imread(samples::findFile("boat.jpg"));


    //Load our Model
    SegmentationModel model(samples::findFile("fcn8s-heavy-pascal.prototxt"),
                            samples::findFile("fcn8s-heavy-pascal.caffemodel"));

    // Define the transformations that we need to apply to our image
    Size size{512, 512}; // resize

    // Whether to swap Red and Blue channels since OpenCV loads images in BGR
    bool swapRB = true;

    // rescale the image (image * scale)
    double scale = 1.0;

    Scalar mean = Scalar(127.5, 127.5, 127.5); // mean to subtract to each channel

    // Set the transformations we want to apply
    model.setInputParams(scale, size, mean, swapRB);

    //Number of classes
    int num_classes = 19;

    //Forward pass
    Mat mask;
    model.segment(image, mask);

    // Generate colors.
    const int rows = mask.size[0];
    const int cols = mask.size[1];

    std::vector<Vec3b> colors;
    colors.push_back(Vec3b());
    for (int i = 1; i < num_classes; ++i)
    {
        Vec3b color;
        for (int j = 0; j < 3; ++j)
            color[j] = (colors[i - 1][j] + rand() % 256) / 2;
        colors.push_back(color);
    }

    //Generate image with segmented colors
    Mat segm;
    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = mask.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }

    // Blend original and segmented image
    resize(segm, segm, image.size(), 0, 0, INTER_NEAREST);
    addWeighted(image, 0.1, segm, 0.9, 0.0, image);
    imshow("Segmented Image", segm);
    imshow("Blended Image", image);
    waitKey(0);

    return 0;
}
