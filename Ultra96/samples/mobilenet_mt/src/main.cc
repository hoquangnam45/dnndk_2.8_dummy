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

/* DPU Kernel name for MobileNet */
#define KRENEL_MOBILNET "mobilenet_relu6"
/* Input Node for Kernel MobileNet */
#define CONV_INPUT_NODE "conv1"
/* Output Node for Kernel MobileNet */
#define OUTPUT_NODE "fc7"

#define IMAGE_COUNT 1000

const string baseImagePath = "./image/";

/*#define SHOWTIME*/
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
 * @param path - path of the kinds file
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
 * @brief Run DPU Task for MobileNet
 *
 * @param taskMobilenet - pointer to MobileNet Task
 * @param img - The mat to be process
 *
 * @return none
 */
void runMobilenet(DPUTask *taskMobilenet, Mat &img) {
    assert(taskMobilenet);

    /* Get channel count of the output Tensor for MobileNet Task  */
    int channel = dpuGetOutputTensorChannel(taskMobilenet, OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];

    vector<float> mean{104, 117, 123};
    float scale = 0.00390625;
    dpuSetInputImageWithScale(taskMobilenet, CONV_INPUT_NODE, img, mean.data(), scale);

    /* Launch RetNet50 Task */
    _T(dpuRunTask(taskMobilenet));

    /* Calculate softmax on CPU and display TOP-5 classification results */
    _T(dpuGetOutputTensorInHWCFP32(taskMobilenet, OUTPUT_NODE, FCResult, channel));
    _T(CPUCalcSoftmax(FCResult, channel, softmax));

    delete[] softmax;
    delete[] FCResult;
}

/*
 * @brief  - Entry of classify using Mobilenet
 *
 * @param kernelMobilenet - point to DPU Kernel of Mobilenet
 */
void classifyEntry(DPUKernel *kernelMobilenet) {
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
            /* Create DPU Tasks from DPU Kernel */
            DPUTask *taskMobilenet = dpuCreateTask(kernelMobilenet, 0);

            for(unsigned int ind = i  ;ind < IMAGE_COUNT;ind+=threadnum) {
                /* Run MobileNet Task */
                runMobilenet(taskMobilenet, img);
            }

            /* Destroy DPU Tasks & free resources */
            dpuDestroyTask(taskMobilenet);
        });
    }

    /* Release thread resources. */
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }

    auto _end = system_clock::now();
    auto duration = (duration_cast<microseconds>(_end - _start)).count();

    cout << "[Time]" << duration << "us" << endl;
    cout << "[FPS]" << IMAGE_COUNT*1000000.0/duration  << endl;
}

/**
 * @brief Entry for running MobileNet neural network
 *
 * @note DNNDK APIs prefixed with "dpu" are used to easily program &
 *       deploy MobileNet on DPU platform.
 *
 */
int main(int argc ,char** argv) {
    DPUKernel *kernelMobilenet;

    if(argc == 2)
        threadnum = stoi(argv[1]);
    else {
        cout << "please input thread number!" << endl;
        exit(-1);
    }

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Create DPU Task for MobileNet */
    kernelMobilenet = dpuLoadKernel(KRENEL_MOBILNET);

    /* Entry of classify using Mobilenet */
    classifyEntry(kernelMobilenet);

    /* Destroy DPU Task & free resources */
    dpuDestroyKernel(kernelMobilenet);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
