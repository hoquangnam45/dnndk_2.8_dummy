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
#define GOOGLENET_WORKLOAD (3.16f)

#define KRENEL_GoogLeNet "inception_v1_0"
/* Input Node for Kernel GoogLeNet */
#define INPUT_NODE "conv1_7x7_s2"
/* Output Node for GoogLeNet */
#define OUTPUT_NODE "loss3_classifier"

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
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
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
 * @brief Run DPU Task for GoogLeNet
 *
 * @param taskGoogLeNet - pointer to GoogLeNet Task
 *
 * @return none
 */
void runGoogLeNet(DPUTask *taskGoogLeNet) {
    assert(taskGoogLeNet);

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
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file words.txt." << endl;
        return;
    }

    /* Get channel count of the output Tensor for GoogLeNet Task  */
    int channel = dpuGetOutputTensorChannel(taskGoogLeNet, OUTPUT_NODE);
    float *softmax = new float[channel];
    float *result = new float[channel];

    for (auto &image_name : images) {
        cout << "\nLoad image : " << image_name << endl;
        /* Load image and Set image into DPU Task for GoogLeNet */
        Mat image = imread(baseImagePath + image_name);
        dpuSetInputImage2(taskGoogLeNet, INPUT_NODE, image);

        /* Run GoogLeNet Task */
        cout << "\nRun DPU Task for GoogLeNet ..." << endl;
        dpuRunTask(taskGoogLeNet);

        /* Get DPU execution time (in us) of DPU Task */
        long long timeProf = dpuGetTaskProfile(taskGoogLeNet);
        cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        float prof = (GOOGLENET_WORKLOAD / timeProf) * 1000000.0f;
        cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Get FC result and convert from INT8 to FP32 format */
        dpuGetOutputTensorInHWCFP32(taskGoogLeNet, OUTPUT_NODE, result, channel);

        /* Calculate softmax on CPU and display TOP-5 classification results */
        CPUCalcSoftmax(result, channel, softmax);
        TopK(softmax, channel, 5, kinds);

        /* Display the image */
        cv::imshow("Classification of inception_v1", image);
        cv::waitKey(1);
    }

    delete[] softmax;
    delete[] result;
}

/**
 * @brief Entry for running GoogLeNet neural network

 *
 */
int main(int argc, char *argv[]) {
    /* DPU Kernels/Tasks for running GoogLeNet */
    DPUKernel *kernelGoogLeNet;
    DPUTask *taskGoogLeNet;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Create DPU Kernels for GoogLeNet */
    kernelGoogLeNet = dpuLoadKernel(KRENEL_GoogLeNet);

    /* Create DPU Tasks for GoogLeNet */
    taskGoogLeNet = dpuCreateTask(kernelGoogLeNet, 0);

    /* Run GoogLeNet Task */
    runGoogLeNet(taskGoogLeNet);

    /* Destroy DPU Tasks & free resources */
    dpuDestroyTask(taskGoogLeNet);

    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(kernelGoogLeNet);

    /* Dettach from DPU driver & release resources */
    dpuClose();

    return 0;
}
