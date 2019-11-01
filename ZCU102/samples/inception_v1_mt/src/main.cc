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
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;
#define RESNET50_WORKLOAD_CONV (7.71f)
#define RESNET50_WORKLOAD_FC (4.0f / 1000)

/* 3.16GOP times calculation for GoogLeNet CONV */
#define GOOGLENET_WORKLOAD_CONV (3.16f)
/* (2.048/1000)GOP times calculation for GoogLeNet FC */
#define GOOGLENET_WORKLOAD_FC (2.048f / 1000)

#define KRENEL_CONV "inception_v1_0"
#define KERNEL_FC "inception_v1_2"

#define CONV_INPUT_NODE "conv1_7x7_s2"
#define CONV_OUTPUT_NODE "inception_5b_output"
#define FC_INPUT_NODE "loss3_classifier"
#define FC_OUTPUT_NODE "loss3_classifier"

#define IMAGE_COUNT 1000

const string baseImagePath = "./image/";

#ifdef SHOWTIME
#define _T(func)                                                          \
{                                                                         \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "us" << endl;                 \
}
#else
#define _T(func) func;
#endif

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
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
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kind file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
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
void TopK(const float *d, int size, int k, vector<string> &vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("[Top]%d prob = %-8f  name = %s\n", i, d[ki.second], vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Compute average pooling on CPU
 *
 * @param conv - pointer to ResNet50 CONV Task
 * @param fc - pointer to ResNet50 FC Task
 *
 * @return none
 */
void CPUCalcAvgPool(DPUTask *conv, DPUTask *fc) {
    assert(conv && fc);

    /* Get output Tensor to the last Node of ResNet50 CONV Task */
    DPUTensor *outTensor = dpuGetOutputTensor(conv, CONV_OUTPUT_NODE);
    /* Get size, height, width and channel of the output Tensor */
    int tensorSize = dpuGetTensorSize(outTensor);
    int outHeight = dpuGetTensorHeight(outTensor);
    int outWidth = dpuGetTensorWidth(outTensor);
    int outChannel = dpuGetTensorChannel(outTensor);

    /* allocate memory buffer */
    int8_t *outBuffer = new int8_t[tensorSize];

    /* Get the input address to the first Node of FC Task */
    int8_t *fcInput = dpuGetInputTensorAddress(fc, FC_INPUT_NODE);

    /* Copy the last Node's output from DPU memory buffer to CPU memory buffer */
    memcpy(outBuffer, dpuGetTensorAddress(outTensor), tensorSize);
    int length = outHeight * outWidth;

    int sum;
    for (int i = 0; i < outChannel; i++) {
        sum = 0;
        for (int j = 0; j < length; j++) {
            sum += outBuffer[outChannel * j + i];
        }
        /* compute average and set into the first input Node of FC Task */
        fcInput[i] = (int8_t)((float)sum / (float)length * 4);
    }

    delete[] outBuffer;
}

/**
 * @brief Run CONV Task and FC Task for ResNet50
 *
 * @param taskConv - pointer to ResNet50 CONV Task
 * @param taskFC - pointer to ResNet50 FC Task
 *
 * @return none
 */
void runGoogLeNet(DPUTask *taskConv, DPUTask *taskFC, Mat img) {
    assert(taskConv && taskFC);

    int channel = dpuGetOutputTensorChannel(taskFC, FC_OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];
    _T(dpuSetInputImage2(taskConv, CONV_INPUT_NODE, img));

    _T(dpuRunTask(taskConv));
    //cout << "[DPU]Conv takes " << dpuGetTaskProfile(taskConv) << " us\n";
    _T(CPUCalcAvgPool(taskConv, taskFC));

    _T(dpuRunTask(taskFC));
    //cout << "[DPU]FC   takes " << dpuGetTaskProfile(taskFC) << " us\n";
    DPUTensor *outTensor = dpuGetOutputTensor(taskFC, FC_OUTPUT_NODE);
    int8_t *outAddr = dpuGetTensorAddress(outTensor);
    float convScale=dpuGetOutputTensorScale(taskFC, FC_OUTPUT_NODE,  0);
    int size = dpuGetOutputTensorSize(taskFC, FC_OUTPUT_NODE);
    _T(dpuRunSoftmax(outAddr, softmax, channel, size/channel ,convScale ));
    //_T(TopK(softmax, channel, 5, kinds));

    delete[] softmax;
    delete[] FCResult;
}

/*
 * @brief  - Entry of face detection using Densebox
 *
 * @param kernel - point to DPU Kernel
 */
void classifyEntry(DPUKernel *kernelconv, DPUKernel *kernelfc) {
    vector<string> kinds, images;
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images exist in " << baseImagePath << endl;
        return;
    } else {
        cout << "total image : " << IMAGE_COUNT << endl;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: Not words exist in words.txt." << endl;
        return;
    }

    thread workers[threadnum];

    Mat img = imread(baseImagePath + images.at(0));
    auto _start = system_clock::now();

    for (auto i = 0; i < threadnum; i++) {
        workers[i] = thread([&,i]() {
            // Create DPU Tasks from DPU Kernel
            DPUTask *taskconv = dpuCreateTask(kernelconv, 0);
            DPUTask *taskfc = dpuCreateTask(kernelfc, 0);

            for(unsigned int ind = i  ;ind < IMAGE_COUNT;ind+=threadnum) {
                // Process the image using DenseBox model
                runGoogLeNet(taskconv, taskfc, img);

            }

            // Destroy DPU Tasks & free resources
            dpuDestroyTask(taskconv);
            dpuDestroyTask(taskfc);
        });
    }

    // Release thread resources.
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }


    auto _end = system_clock::now();
    auto duration = (duration_cast<microseconds>(_end - _start)).count();
    cout << "[Time]" << duration << "us" << endl;
    cout << "[FPS]" << IMAGE_COUNT*1000000.0/duration  << endl;
}

/**
 * @brief Entry for running GoogLeNet neural network
 *
 */
int main(int argc ,char** argv) {
    DPUKernel *kernelConv;
    DPUKernel *kernelFC;

    if(argc == 2)
    {
        threadnum = stoi(argv[1]);
    }
    else {
        cerr << "please input thread number!" <<  endl;
        exit(-1);
    }

    dpuOpen();
    kernelConv = dpuLoadKernel(KRENEL_CONV);
    kernelFC = dpuLoadKernel(KERNEL_FC);

    classifyEntry(kernelConv, kernelFC);

    dpuDestroyKernel(kernelConv);
    dpuDestroyKernel(kernelFC);
    dpuClose();

    return 0;
}

