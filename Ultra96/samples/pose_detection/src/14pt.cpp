/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include "14pt.h"

namespace deephi {

/**
 * Draw line on an image
 */
void drawline(Mat& img,
    Point2f point1, Point2f point2,
    Scalar colour, int thickness,
    float scale_w, float scale_h) {
    if ((point1.x > scale_w || point1.y > scale_h) && (point2.x > scale_w || point2.y > scale_h))
    {
        line(img, point1, point2, colour, thickness);
    }
}

/**
 * Draw lines on the image
 */
void draw_img(Mat& img, vector<float>& results, float scale_w, float scale_h) {
    float mark = 5.f;
    float mark_w = mark * scale_w;
    float mark_h = mark * scale_h;
    vector<Point2f> pois(14);

    for (size_t i = 0; i < pois.size(); ++i) {
        pois[i].x = results[i * 2] * scale_w;
        pois[i].y = results[i * 2 + 1] * scale_h;
    }

    for (size_t i = 0; i < pois.size(); ++i) {
        circle(img, pois[i], 3, Scalar::all(255));
    }

    drawline(img, pois[0], pois[1], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[1], pois[2], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[6], pois[7], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[7], pois[8], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[4], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[4], pois[5], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[9], pois[10], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[10], pois[11], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[12], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[0], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[0], pois[6], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[9], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[6], pois[9], Scalar(0, 255, 255), 2, mark_w, mark_h);
}

/**
 * convert output data format
 */
void dpuOutputIn2F32(DPUTask* task, const char* nodeName, float* buffer, int size) {
    int8_t* outputAddr = dpuGetOutputTensorAddress(task, nodeName);
    float scale = dpuGetOutputTensorScale(task, nodeName);

    for (int idx = 0; idx < size; idx++) {
        buffer[idx] = outputAddr[idx] * scale;
    }
}

/**
 * do average pooling calculation
 */
void CPUCalcAvgPool(DPUTask* conv, DPUTask* fc) {
    assert(conv && fc);
    DPUTensor* outTensor = dpuGetOutputTensor(conv, PT_CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(outTensor);
    int outWidth = dpuGetTensorWidth(outTensor);
    int outChannel = dpuGetTensorChannel(outTensor);
    int tensorSize = dpuGetTensorSize(outTensor);

    float* outBuffer = new float[tensorSize];
    dpuGetOutputTensorInHWCFP32(conv, PT_CONV_OUTPUT_NODE, outBuffer, tensorSize);

    int8_t* fcInput = dpuGetInputTensorAddress(fc, PT_FC_NODE);
    float scaleFC = dpuGetInputTensorScale(fc, PT_FC_NODE);
    int length = outHeight * outWidth;
    float avg = static_cast<float>(length);

    for (int i = 0; i < outChannel; i++) {
        float sum = 0.0f;
        for (int j = 0; j < length; j++) {
            sum += outBuffer[outChannel * j + i];
        }
        fcInput[i] = static_cast<int8_t>(sum / avg * scaleFC);
    }

    delete[] outBuffer;
}

/**
 * construction  of GestureDetect
 *      initialize the DPU Kernels
 */
GestureDetect::GestureDetect() {
}

/**
 * destroy the DPU Kernels and Tasks
 */
GestureDetect::~GestureDetect() {
}

/**
 * @brief Init - initialize the 14pt model
 */
void GestureDetect::Init() {
    kernel_conv_PT = dpuLoadKernel(PT_KRENEL_CONV);
    kernel_fc_PT = dpuLoadKernel(PT_KRENEL_FC);

    task_conv_PT = dpuCreateTask(kernel_conv_PT, 0);
    task_fc_PT = dpuCreateTask(kernel_fc_PT, 0);
}

/**
 * @brief Finalize - release resource
 */
void GestureDetect::Finalize() {
    if(task_conv_PT) {
        dpuDestroyTask(task_conv_PT);
    }

    if(kernel_conv_PT) {
        dpuDestroyKernel(kernel_conv_PT);
    }

    if(task_fc_PT) {
        dpuDestroyTask(task_fc_PT);
    }

    if(kernel_fc_PT) {
        dpuDestroyKernel(kernel_fc_PT);
    }
}

/**
 *  @brief Run - run detection algorithm
 */
void GestureDetect::Run(cv::Mat& img) {
    vector<float> results(28);
    float mean[3] = {104, 117, 123};
    int width = dpuGetInputTensorWidth(task_conv_PT, PT_CONV_INPUT_NODE);
    int height = dpuGetInputTensorHeight(task_conv_PT, PT_CONV_INPUT_NODE);

    dpuSetInputImage(task_conv_PT,PT_CONV_INPUT_NODE,img,mean);

    dpuRunTask(task_conv_PT);
    CPUCalcAvgPool(task_conv_PT, task_fc_PT);
    dpuRunTask(task_fc_PT);

    int channel = dpuGetOutputTensorChannel(task_fc_PT, PT_FC_NODE);

    dpuOutputIn2F32(task_fc_PT, PT_FC_NODE, results.data(), channel);

    float scale_w = (float)img.cols / (float)width;
    float scale_h = (float)img.rows / (float)height;

    draw_img(img, results, scale_w, scale_h);
}

}
