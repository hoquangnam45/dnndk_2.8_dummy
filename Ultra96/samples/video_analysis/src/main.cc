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

#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <chrono>

// Header file OpenCV for image processing
#include <opencv2/opencv.hpp>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

#include "ssd_detector.h"
#include "prior_boxes.h"

using namespace std;
using namespace cv;
using namespace deephi;

// DPU Kernel name for SSD Convolution layers
#define KERNEL_CONV "ssd"
// DPU node name for input and output
#define CONV_INPUT_NODE "conv1_1"
#define CONV_OUTPUT_NODE_LOC "mbox_loc"
#define CONV_OUTPUT_NODE_CONF "mbox_conf"

// detection params
const float NMS_THRESHOLD = 0.45;
const float CONF_THRESHOLD = 0.5;
const int TOP_K = 400;
const int KEEP_TOP_K = 200;
int num_classes = 4;
const int TNUM = 6;

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
array<bool, TNUM> is_running;
bool is_displaying = true;

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

queue<pair<int, Mat>> read_queue;                                               // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue;  // display queue
mutex mtx_read_queue;                                                           // mutex of read queue
mutex mtx_display_queue;                                                        // mutex of display queue
int read_index = 0;                                                             // frame index of input video
int display_index = 0;                                                          // frame index to display

/**
 * @brief Create prior boxes for feature maps of one scale
 *
 * @note Each prior box is represented as a vector: c-x, c-y, width, height,
 *       variences.
 *
 * @param priors - the result of prior boxes
 *
 * @return none
 */
void CreatePriors(vector<shared_ptr<vector<float>>> *priors) {
    vector<float> variances{0.1, 0.1, 0.2, 0.2};
    vector<PriorBoxes> prior_boxes;

    // vehicle detect
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 60, 45, variances, {15.0, 30}, {33.0, 60}, {2}, 0.5, 8.0, 8.0});
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 30, 23, variances, {66.0}, {127.0}, {2, 3}, 0.5, 16, 16});
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 15, 12, variances, {127.0}, {188.0}, {2, 3}, 0.5, 32, 32});
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 8, 6, variances, {188.0}, {249.0}, {2, 3}, 0.5, 64, 64});
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 6, 4, variances, {249.0}, {310.0}, {2}, 0.5, 100, 100});
    prior_boxes.emplace_back(PriorBoxes{
          480, 360, 4, 2, variances, {310.0}, {372.0}, {2}, 0.5, 300, 300});

    int num_priors = 0;
    for (auto &p : prior_boxes) {
        num_priors += p.priors().size();
    }

    priors->clear();
    priors->reserve(num_priors);
    for (auto i = 0U; i < prior_boxes.size(); ++i) {
        priors->insert(priors->end(), prior_boxes[i].priors().begin(),
            prior_boxes[i].priors().end());
    }
}

/**
 * @brief Run DPU and ARM Tasks for SSD, and put image into display queue
 *
 * @param task_conv - pointer to SSD CONV Task
 * @param running - status flag of RunSSD thread
 * @param priors - pointer to prior box
 *
 * @return none
 */
void RunSSD(DPUTask *task_conv, bool &running, vector<shared_ptr<vector<float>>> &priors) {
    // Initializations
    int8_t* loc =
        (int8_t*)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_LOC);
    int8_t* conf =
        (int8_t*)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_CONF);
    float loc_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_LOC);
    float conf_scale =
        dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_CONF);
    int size = dpuGetOutputTensorSize(task_conv, CONV_OUTPUT_NODE_CONF);

    float* conf_softmax = new float[size];

    // Run detection for images in read queue
    while (running) {
        // Get an image from read queue
        int index;
        Mat img;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            mtx_read_queue.unlock();
            if (is_reading) {
                continue;
            } else {
                running = false;
                break;
            }
        } else {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }

        // Set image and run CONV Task
        dpuSetInputImage2(task_conv, (char *)CONV_INPUT_NODE, img);
        dpuRunTask(task_conv);

        // Run the calculation for softmax
        dpuRunSoftmax(conf, conf_softmax, 4, size/4, conf_scale);

        // Post-process after DPU running
        MultiDetObjects results;
        vector<float> th_conf(num_classes, CONF_THRESHOLD);
        SSDdetector* detector_ = new SSDdetector(num_classes, SSDdetector::CodeType::CENTER_SIZE, false,
                  KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors, loc_scale);
        detector_->Detect(loc, conf_softmax, &results);

        for (size_t i = 0; i < results.size(); ++i) {
            int label = get<0>(results[i]);
            int xmin = get<2>(results[i]).x * img.cols;
            int ymin = get<2>(results[i]).y * img.rows;
            int xmax = xmin + (get<2>(results[i]).width) * img.cols;
            int ymax = ymin + (get<2>(results[i]).height) * img.rows;
            xmin = std::min(std::max(xmin, 0), img.cols);
            xmax = std::min(std::max(xmax, 0), img.cols);
            ymin = std::min(std::max(ymin, 0), img.rows);
            ymax = std::min(std::max(ymax, 0), img.rows);

            if(label == 1) {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1,
                          1, 0);
            } else if (label == 2) {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
                1, 0);
            } else if (label == 3) {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0), 1,
                         1, 0);
            }
        }

        // Put image into display queue
        mtx_display_queue.lock();
        display_queue.push(make_pair(index, img));
        mtx_display_queue.unlock();
    }

    delete[] conf_softmax;
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool &is_reading) {
    while (is_reading) {
        Mat img;
        if (read_queue.size() < 30) {
            if (!video.read(img)) {
                cout << "Video end." << endl;
                is_reading = false;
                break;
            }
            mtx_read_queue.lock();
            read_queue.push(make_pair(read_index++, img));
            mtx_read_queue.unlock();
        } else {
            usleep(20);
        }
    }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool &is_displaying) {
    Mat image(360, 480, CV_8UC3);
    imshow("Video Analysis@Deephi DPU", image);
    while (is_displaying) {
        mtx_display_queue.lock();
        if (display_queue.empty()) {
            if (any_of(is_running.begin(), is_running.end(), [](bool cond){return cond;})) {
                mtx_display_queue.unlock();
                usleep(20);
            } else {
                is_displaying = false;
                break;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
            imshow("Video Analysis@Deephi DPU", display_queue.top().second);
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
            if (waitKey(1) == 'q') {
                is_reading = false;
                for(int i = 0; i < TNUM; ++i) {
                    is_running[i] = false;
                }
                is_displaying = false;
                break;
            }
        } else {
            mtx_display_queue.unlock();
        }
    }
}

/**
 * @brief Entry for running SSD neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 * @return 0 on success, or error message dispalyed in case of failure.
 */
int main(int argc, char** argv) {
    // Check args
    if (argc != 2) {
        cout << "Usage of video analysis demo: ./video_analysis video_file[string]" << endl;
        cout << "\tvideo_file: file path to the input video file" << endl;
        return -1;
    }

    // DPU Kernel for running SSD
    DPUKernel *kernel_conv;

    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Create DPU Kernel and Task for CONV Nodes in SSD
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    vector<DPUTask*> task_conv(TNUM);

    vector<shared_ptr<vector<float>>> priors;
    CreatePriors(&priors);

    // Initializations
    string file_name = argv[1];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }

    // Run DPU Tasks for SSD
    vector<thread> threads(TNUM);
    is_running.fill(true);
    for(int i = 0; i < TNUM; ++i) {
        task_conv[i] = dpuCreateTask(kernel_conv, 0);
        threads[i] = thread(RunSSD, ref(task_conv[i]), ref(is_running[i]), ref(priors));
    }
    threads.push_back(thread(Read, ref(is_reading)));
    threads.push_back(thread(Display, ref(is_displaying)));

    for (int i = 0; i < 2+TNUM; ++i) {
        threads[i].join();
    }

    // Destroy DPU Tasks and Kernel and free resources
    for(int i = 0; i < TNUM; ++i) {
        dpuDestroyTask(task_conv[i]);
    }
    dpuDestroyKernel(kernel_conv);

    // Detach from DPU driver and release resources
    dpuClose();

    video.release();
    return 0;
}
