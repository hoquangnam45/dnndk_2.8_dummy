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
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

#include "14pt.h"
#include "ssd.h"

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace deephi;

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_displaying = true;

queue<pair<int, Mat>> read_queue;                                               // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue;  // display queue
mutex mtx_read_queue;                                                           // mutex of read queue
mutex mtx_display_queue;                                                        // mutex of display queue
int read_index = 0;                                                             // frame index of input video
int display_index = 0;                                                          // frame index to display

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runGestureDetect(bool &is_running) {
    SSD ssd;
    GestureDetect gesture;
    ssd.Init("ssd_person");
    gesture.Init();

    // Run detection for images in read queue
    while (is_running) {
        // Get an image from read queue
        int index;
        Mat img;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            mtx_read_queue.unlock();
            if (is_reading) {
                continue;
            } else {
                is_running = false;
                break;
            }
        } else {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }

        // detect persons using ssd
        vector<tuple<int, float, cv::Rect_<float>>> results;
        ssd.Run(img, &results);

        // detect joint point of each person
        for (size_t i = 0; i < results.size(); ++i) {
            int xmin = get<2>(results[i]).x * img.cols;
            int ymin = get<2>(results[i]).y * img.rows;
            int xmax = xmin + (get<2>(results[i]).width) * img.cols;
            int ymax = ymin + (get<2>(results[i]).height) * img.rows;
            xmin = min(max(xmin, 0), img.cols);
            xmax = min(max(xmax, 0), img.cols);
            ymin = min(max(ymin, 0), img.rows);
            ymax = min(max(ymax, 0), img.rows);

            Rect roi = Rect(Point(xmin, ymin), Point(xmax, ymax));
            Mat sub_img = img(roi);
            gesture.Run(sub_img);
        }

        // Put image into display queue
        mtx_display_queue.lock();
        display_queue.push(make_pair(index, img));
        mtx_display_queue.unlock();
    }

    ssd.Finalize();
    gesture.Finalize();
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
                cout << "Finish reading the video." << endl;
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
    while (is_displaying) {
        mtx_display_queue.lock();
        if (display_queue.empty()) {
            if (is_running_1 ) {
                mtx_display_queue.unlock();
                usleep(20);
            } else {
                mtx_display_queue.unlock();
                is_displaying = false;
                break;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
            imshow("PoseDetection @Deephi DPU", display_queue.top().second);
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
            if (waitKey(1) == 'q') {
                is_reading = false;
                is_running_1 = false;
                is_displaying = false;
                break;
            }
        } else {
            mtx_display_queue.unlock();
        }
    }
}

/**
 * @brief Entry for running pose detection neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv) {
    // Check args
    if (argc != 2) {
        cout << "Usage of pose detection demo: ./pose_detection file_name[string]" << endl;
        cout << "\tfile_name: path to your video file" << endl;
        return -1;
    }

    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Initializations
    string file_name = argv[1];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }

    // Run tasks for SSD
    array<thread, 4> threads = {thread(Read, ref(is_reading)),
                                thread(runGestureDetect, ref(is_running_1)),
                                thread(runGestureDetect, ref(is_running_1)),
                                thread(Display, ref(is_displaying))};

    for (int i = 0; i < 4; ++i) {
        threads[i].join();
    }

    // Detach from DPU driver and release resources
    dpuClose();
    video.release();

    return 0;
}
