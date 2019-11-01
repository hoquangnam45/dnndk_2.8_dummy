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

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

/* 3.16GOP times calculation for GoogLeNet CONV */
#define GOOGLENET_WORKLOAD_CONV (3.16f)
/* (2.048/1000)GOP times calculation for GoogLeNet FC */
#define GOOGLENET_WORKLOAD_FC (2.048f / 1000)

#define KRENEL_CONV "inception_v1_0"
#define KERNEL_FC "inception_v1_2"

#define TASK_CONV_INPUT "conv1_7x7_s2"
#define TASK_CONV_OUTPUT "inception_5b_output"
#define TASK_FC_INPUT "loss3_classifier"
#define TASK_FC_OUTPUT "loss3_classifier"

const string baseImagePath = "../common/image500_640_480/";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(std::string const &path, std::vector<std::string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            std::string name = entry->d_name;
            std::string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(std::string const &path, std::vector<std::string> &kinds) {
    kinds.clear();
    std::fstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    std::string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
}


/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, std::vector<std::string> &vkinds) {
    assert(d && size > 0 && k > 0);
    std::priority_queue<std::pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(std::pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        std::pair<float, int> ki = q.top();
        fprintf(stdout, "top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
        vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Compute average pooling on CPU
 *
 * @param conv - pointer to GoogLeNet CONV Task
 * @param fc - pointer to GoogLeNet CONV Task
 *
 * @return none
 */
void CPUCalcAvgPool(DPUTask *conv, DPUTask *fc) {
    assert(conv && fc);

    DPUTensor *conv_out_tensor = dpuGetOutputTensor(conv, TASK_CONV_OUTPUT);

    /* Get size of the output Tensor */
    int tensorSize = dpuGetTensorSize(conv_out_tensor);
    /* Get height dimension of the output Tensor */
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    /* Get width dimension of the output Tensor */
    int outWidth = dpuGetTensorWidth(conv_out_tensor);

    /**
    * Get the channels of the last inception to compute the output Node's
    * actual output channel.
    */
    int outChannel = dpuGetTensorChannel(conv_out_tensor);

    /* Allocate the memory and store conv's output after conversion */
    float *outBuffer = new float[tensorSize];
    dpuGetOutputTensorInHWCFP32(conv, TASK_CONV_OUTPUT, outBuffer, tensorSize);

    /* Get the input address to the first Node of FC Task */
    int8_t *fcInput = dpuGetInputTensorAddress(fc, TASK_FC_INPUT);
    /* Get scale value for the first input Node of FC task */
    float scaleFC = dpuGetInputTensorScale(fc, TASK_FC_INPUT);
    int length = outHeight * outWidth;
    float avg = static_cast<float>(length);
    for (int i = 0; i < outChannel; i++) {
        float sum = 0.0f;
        for (int j = 0; j < length; j++) {
            sum += outBuffer[outChannel * j + i];
        }

        /* Compute average and set into the first input Node of FC Task */
        fcInput[i] = static_cast<int8_t>(sum / avg * scaleFC);
    }

    delete[] outBuffer;
}

/**
 * @brief Run DPU CONV Task and FC Task for GoogLeNet
 *
 * @param taskConv - pointer to GoogLeNet CONV Task
 * @param taskFC - pointer to GoogLeNet FC Task
 *
 * @return none
 */
void runGoogLeNet(DPUTask *taskConv, DPUTask *taskFC) {
    assert(taskConv && taskFC);

    /* Mean value for GoogLeNet specified in Caffe prototxt */
    vector<string> kinds, images;

    /*Load all image names */
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images existing under " << baseImagePath << endl;
        return;
    }

    /*Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
    /* Get channel count of the output Tensor for FC Task  */
    int channel = dpuGetOutputTensorChannel(taskFC, TASK_FC_OUTPUT);
    float *softmax = new float[channel];
    float *result = new float[channel];
    for (auto &image_name : images) {
        cout << "\nLoad image : " << image_name << endl;
        /* Load image and Set image into DPU Task for GoogLeNet */
        Mat image = imread(baseImagePath + image_name);
        dpuSetInputImage2(taskConv, TASK_CONV_INPUT, image);

        /* Run GoogLeNet Task */
        cout << "\nRun GoogLeNet CONV ..." << endl;
        dpuRunTask(taskConv);

        /* Get DPU execution time (in us) of CONV Task */
        int64_t timeProf = dpuGetTaskProfile(taskConv);
        cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";
        float convProf = (GOOGLENET_WORKLOAD_CONV / timeProf) * 1000000.0f;
        cout << "  DPU CONV Performance: " << convProf << "GOPS\n";

        /* Compute average pooling on CPU */
        CPUCalcAvgPool(taskConv, taskFC);

        /* Run GoogLeNet FC layers */
        cout << "Run GoogLeNet FC Part ..." << endl;

        /* Launch the running of GoogLeNet FC Task */
        dpuRunTask(taskFC);
        /* Get DPU execution time (in us) for FC Task */
        timeProf = dpuGetTaskProfile(taskFC);
        cout << "  DPU FC Execution time: " << (timeProf * 1.0f) << "us\n";
        float fcProf = (GOOGLENET_WORKLOAD_FC / timeProf) * 1000000.0f;
        cout << "  DPU FC Performance: " << fcProf << "GOPS\n";
        DPUTensor *outTensor = dpuGetOutputTensor(taskFC, TASK_FC_OUTPUT);
        int8_t *outAddr = dpuGetTensorAddress(outTensor);
        float convScale=dpuGetOutputTensorScale(taskFC, TASK_FC_OUTPUT,  0);
        int size = dpuGetOutputTensorSize(taskFC, TASK_FC_OUTPUT);
        /* Calculate softmax on CPU and display TOP-5 classification results */
        dpuRunSoftmax(outAddr, softmax, channel, size/channel ,convScale );

        TopK(softmax, channel, 5, kinds);

        /* Display the image */
        cv::imshow("Image", image);
        cv::waitKey(1);
    }

    delete[] softmax;
    delete[] result;
}

/**
 * @brief Entry for running GoogLeNet neural network
 *
 * @note DNNDK APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char *argv[]) {
    /* DPU Kernels/Tasks for running GoogLeNet */
    DPUKernel *kernelConv;
    DPUKernel *kernelFC;
    DPUTask *taskConv;
    DPUTask *taskFC;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Create DPU Kernels for CONV & FC Nodes in GoogLeNet */
    kernelConv = dpuLoadKernel(KRENEL_CONV);
    kernelFC = dpuLoadKernel(KERNEL_FC);

    /* Create DPU Tasks for CONV & FC Nodes in GoogLeNet */
    taskConv = dpuCreateTask(kernelConv, 0);
    taskFC = dpuCreateTask(kernelFC, 0);

    /* Run CONV & FC Kernels for GoogLeNet */
    runGoogLeNet(taskConv, taskFC);

    /* Destroy DPU Tasks & free resources */
    dpuDestroyTask(taskConv);
    dpuDestroyTask(taskFC);

    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(kernelConv);
    dpuDestroyKernel(kernelFC);

    /* Dettach from DPU driver & release resources */
    dpuClose();

    return 0;
}
