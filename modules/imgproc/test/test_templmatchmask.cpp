/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2019, Julius Durst, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CompareNaiveImpl : public TestWithParam<int>
{
protected:
    // Member functions inherited from ::testing::Test
    void SetUp() override;

    // Data members
    RNG rng_;
    // Matrices for test calculations (always CV_32)
    Mat img_;
    Mat templ_;
    Mat mask_;
    Mat templ_masked_;
    Mat img_roi_masked_;
    // Matrices for call to matchTemplate (have test type)
    Mat img_testtype_;
    Mat templ_testtype_;
    Mat mask_testtype_;
    Mat result_;

    // Constants
    static const Size IMG_SIZE;
    static const Size TEMPL_SIZE;
    static const Point TEST_POINT;
};

const Size  CompareNaiveImpl::IMG_SIZE(160, 100); // Arbitraryly chosen value
const Size  CompareNaiveImpl::TEMPL_SIZE(21, 13); // Arbitraryly chosen value
const Point CompareNaiveImpl::TEST_POINT(8, 9);   // Arbitraryly chosen value

void CompareNaiveImpl::SetUp()
{
    int type = GetParam();

    // Matrices are created with the depth to test (for the call to matchTemplate()), but are also
    // converted to CV_32 for the test calculations, because matchTemplate() also only operates on
    // and returns CV_32.
    img_testtype_.create(IMG_SIZE, type);
    templ_testtype_.create(TEMPL_SIZE, type);
    mask_testtype_.create(TEMPL_SIZE, type);

    rng_.fill(img_testtype_, RNG::UNIFORM, 0, 10);
    rng_.fill(templ_testtype_, RNG::UNIFORM, 0, 10);
    rng_.fill(mask_testtype_, RNG::UNIFORM, 0, 5);

    img_testtype_.convertTo(img_, CV_32F);
    templ_testtype_.convertTo(templ_, CV_32F);
    mask_testtype_.convertTo(mask_, CV_32F);
    if (CV_MAT_DEPTH(type) == CV_8U)
    {
        mask_ /= 255.0;
    }

    Rect roi(TEST_POINT, TEMPL_SIZE);
    img_roi_masked_ = img_(roi).mul(mask_);
    templ_masked_ = templ_.mul(mask_);
}

TEST_P(CompareNaiveImpl, accuracy_SQDIFF)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_SQDIFF, mask_testtype_);
    // Naive implementation for one point
    Mat temp = img_roi_masked_ - templ_masked_;
    Scalar temp_s = sum(temp.mul(temp));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

TEST_P(CompareNaiveImpl, accuracy_SQDIFF_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_SQDIFF_NORMED, mask_testtype_);
    // Naive implementation for one point
    Mat temp = img_roi_masked_ - templ_masked_;
    Scalar temp_s = sum(temp.mul(temp));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templ_masked_.mul(templ_masked_));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(img_roi_masked_.mul(img_roi_masked_));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

TEST_P(CompareNaiveImpl, accuracy_CCORR)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCORR, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(templ_masked_.mul(img_roi_masked_));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

TEST_P(CompareNaiveImpl, accuracy_CCORR_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCORR_NORMED, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(templ_masked_.mul(img_roi_masked_));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templ_masked_.mul(templ_masked_));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(img_roi_masked_.mul(img_roi_masked_));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

TEST_P(CompareNaiveImpl, accuracy_CCOEFF)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCOEFF, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(mask_);
    for (int i = 0; i < 4; i++)
    {
        if (temp_s[i] != 0.0)
            temp_s[i] = 1.0 / temp_s[i];
        else
            temp_s[i] = 1.0;
    }
    Mat temp = mask_.clone(); temp = temp_s; // Workaround to multiply Mat by Scalar
    Mat temp2 = mask_.clone(); temp2 = sum(templ_masked_); // Workaround to multiply Mat by Scalar
    Mat templx = templ_masked_ - mask_.mul(temp).mul(temp2);
    temp2 = sum(img_roi_masked_); // Workaround to multiply Mat by Scalar
    Mat imgx = img_roi_masked_ - mask_.mul(temp).mul(temp2);
    temp_s = sum(templx.mul(imgx));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

TEST_P(CompareNaiveImpl, accuracy_CCOEFF_NORMED)
{
    matchTemplate(img_testtype_, templ_testtype_, result_, CV_TM_CCOEFF_NORMED, mask_testtype_);
    // Naive implementation for one point
    Scalar temp_s = sum(mask_);
    for (int i = 0; i < 4; i++)
    {
        if (temp_s[i] != 0.0)
            temp_s[i] = 1.0 / temp_s[i];
        else
            temp_s[i] = 1.0;
    }
    Mat temp = mask_.clone(); temp = temp_s; // Workaround to multiply Mat by Scalar
    Mat temp2 = mask_.clone(); temp2 = sum(templ_masked_); // Workaround to multiply Mat by Scalar
    Mat templx = templ_masked_ - mask_.mul(temp).mul(temp2);
    temp2 = sum(img_roi_masked_); // Workaround to multiply Mat by Scalar
    Mat imgx = img_roi_masked_ - mask_.mul(temp).mul(temp2);
    temp_s = sum(templx.mul(imgx));
    double val = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];

    // Normalization
    temp_s = sum(templx.mul(templx));
    double norm = temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    temp_s = sum(imgx.mul(imgx));
    norm *= temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
    norm = sqrt(norm);
    val /= norm;

    EXPECT_NEAR(val, result_.at<float>(TEST_POINT), 10.0*abs(val)*FLT_EPSILON);
}

INSTANTIATE_TEST_CASE_P(Imgproc_MatchTemplateMasked, CompareNaiveImpl,
                        Values(CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3));


class CompareWithAndWithoutMask : public TestWithParam<std::tuple<int,int>>
{
protected:
    // Member functions inherited from ::testing::Test
    void SetUp() override;

    // Data members
    RNG rng_;
    Mat img_;
    Mat templ_;
    Mat mask_;
    Mat result_withoutmask_;
    Mat result_withmask_;

    // Constants
    static const Size IMG_SIZE;
    static const Size TEMPL_SIZE;
};

const Size  CompareWithAndWithoutMask::IMG_SIZE(160, 100); // Arbitraryly chosen value
const Size  CompareWithAndWithoutMask::TEMPL_SIZE(21, 13); // Arbitraryly chosen value

void CompareWithAndWithoutMask::SetUp()
{
    int type = std::get<0>(GetParam());

    img_.create(IMG_SIZE, type);
    templ_.create(TEMPL_SIZE, type);
    mask_.create(TEMPL_SIZE, type);

    rng_.fill(img_, RNG::UNIFORM, 0, 100);
    rng_.fill(templ_, RNG::UNIFORM, 0, 100);
    mask_ = Scalar(1, 1, 1, 1);

    if (CV_MAT_DEPTH(type) == CV_8U)
    {
        mask_ *= 255;
    }
}

TEST_P(CompareWithAndWithoutMask, compatibility_correctness)
{
    int method = std::get<1>(GetParam());

    matchTemplate(img_, templ_, result_withmask_, method, mask_);
    matchTemplate(img_, templ_, result_withoutmask_, method);

    // Get maximum result for relative error calculation
    double min, max;
    minMaxLoc(abs(result_withmask_), &min, &max);

    // Get maximum of absolute diff for comparison
    double mindiff, maxdiff;
    minMaxLoc(abs(result_withmask_ - result_withoutmask_), &mindiff, &maxdiff);

    EXPECT_LT(maxdiff, max*100.0*FLT_EPSILON);
}


INSTANTIATE_TEST_CASE_P(Imgproc_MatchTemplateMasked, CompareWithAndWithoutMask,
    Combine(
        Values(CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3),
        Values(CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCORR, CV_TM_CCORR_NORMED,
               CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED)));

}} // namespace
