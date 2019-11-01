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

#ifndef __SSD_H__
#define __SSD_H__

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <tuple>
#include <chrono>

#include <cmath>
#include <algorithm>
#include <functional>
#include <tuple>
#include <thread>
#include <iostream>
#include <opencv2/core.hpp>

#include <dnndk/dnndk.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

#ifdef MEASURE_PERFORMANCE
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeMeasure]" << left << setw(30) << tmp;                   \
        cout << right << setw(10) << duration << " us" << endl;               \
    }
#else
#define _T(func) func;
#endif

namespace deephi {

using SingleDetObject = tuple<int, float, cv::Rect_<float> >;
using MultiDetObjects = vector<SingleDetObject>;

enum SSD_TYPE {
        VEHICLE, PERSON
    };

/*
 * class SSDdetector: post processing of SSD model
 */
class SSDdetector {

 public:
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  SSDdetector(unsigned int num_classes,
      CodeType code_type, bool variance_encoded_in_target,
      unsigned int keep_top_k,
      const vector<float>& confidence_threshold,
      unsigned int nms_top_k, float nms_threshold, float eta,
      const vector<shared_ptr<vector<float> > >& priors,
      float scale = 1.f, bool clip = false);

  template <typename T>
  void Detect(const T* loc_data, const float* conf_data,
              MultiDetObjects* result);

 protected:

    /*
     * @brief ApplyOneClassNMS - Non-Maximum Suppression(NMS) for one class
     */
  template <typename T>
  void ApplyOneClassNMS(const T (*bboxes)[4], const float* conf_data,
      int label, const vector<pair<float, int> >& score_index_vec,
      vector<int>* indices);

    /*
     * @brief GetOneClassMaxScoreIndex - get one max score index
     */
  void GetOneClassMaxScoreIndex(const float* conf_data, int label,
      vector<pair<float, int> >* score_index_vec);

    /*
     * @brief GetMultiClassMaxScoreIndex - get multiple max score index
     */
  void GetMultiClassMaxScoreIndex(const float* conf_data,
      int start_label, int num_classes,
      vector<vector<pair<float, int> > >* score_index_vec);

    /*
     * @brief GetMultiClassMaxScoreIndexMT - get multiple max score index in multi-thread
     */
  void GetMultiClassMaxScoreIndexMT(const float* conf_data,
      int start_label, int num_classes,
      vector<vector<pair<float, int> > >* score_index_vec,
      int threads = 2);

    /*
     * @brief JaccardOverlap - calculate Jaccard Overlap
     */
  template <typename T>
  float JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
      bool normalized = true);

    /*
     * @brief DecodeBBox - decode bounding box
     */
  template <typename T>
  void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

  map<int, vector<float> > decoded_bboxes_;

  const unsigned int num_classes_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  unsigned int keep_top_k_;
  vector<float> confidence_threshold_;
  float nms_confidence_;
  unsigned int nms_top_k_;
  float nms_threshold_;
  float eta_;

  const vector<shared_ptr<vector<float> > >& priors_;
  float scale_;
  bool clip_;
  int num_priors_;
};

/*
 * class PriorBoxes : helper class for creating prior boxes in ssd
 */
class PriorBoxes {

 public:
  PriorBoxes(
          int image_width,
            int image_height,
      int layer_width,
            int layer_height,
      const vector<float>& variances,
      const vector<float>& min_sizes,
            const vector<float>& max_sizes,
      const vector<float>& aspect_ratios,
            float offset,
      float step_width = 0.f,
            float step_height = 0.f,
      bool flip = true,
            bool clip = false);

  const vector<shared_ptr<vector<float> > >& priors() const {
    return priors_;
  }
    /*
     * @brief Create - Create prior boxes according to SSD type
     *
     * @param vecPriors - the vector of prior boxes
     * @param type - SSD type: vehicle or person
     */
  static void Create(vector<shared_ptr<vector<float> > >& vecPriors, SSD_TYPE type);

 protected:

  void CreatePriors();

  vector<shared_ptr<vector<float> > > priors_;

  pair<int, int> image_dims_;
  pair<int, int> layer_dims_;
  pair<float, float> step_dims_;

  vector<pair<float, float> > boxes_dims_;

  float offset_;
  bool clip_;

  vector<float> variances_;
};

/*
 * class SSD : top class for target detection using SSD
 */
class SSD {

public:
    SSD(){
    }
    ~SSD();

    /*
     * @brief Init - initialize SSD model
     *
     * @param kernel_name - the kernel name which is generated by DNNC
     * @param input_node - name of the input node in SSD model
     * @param output_loc - name of the location node in SSD model
     * @param output_conf - name of the confidence node in SSD model
     * @param th_nms - NMS threshold
     * @param top_k - the IOU threshold
     * @param class_k - the tiling dimension
     */
    void Init(const string& kernel_name,
            const string& input_node = "conv1_1",
            const string& output_loc = "mbox_loc",
            const string& output_conf = "mbox_conf",
            float th_nms = 0.5,
            int top_k = 400,
            int class_k = 200);

    /*
     * @brief Run - target detection using SSD model
     *
     * @param img - the input image
     * @param results - the detected targets
     *
     */
    void Run(Mat& img, MultiDetObjects* results);

    /*
     * @brief Finalize - release resource
     */
    void Finalize();

    /*
     * @brief DrawBoxes - draw detected boxes on the image
     *
     * @param img - the input image
     * @param results - the detected targets
     *
     */
    void DrawBoxes(Mat& img, MultiDetObjects & results);

private:

    vector<shared_ptr<vector<float>>> priors_;

    string kernel_name_;
    string input_node_;
    string output_loc_;
    string output_conf_;

    int num_classes_;
    vector<float> th_conf_;

    float* softmax_data_;
    SSDdetector* detector_;

    DPUKernel* kernel_conv;
    DPUTask* task;

    int8_t* loc;
    int8_t* conf;
    float loc_scale, conf_scale;
    int conf_size;
    SSD_TYPE type_;
};

}

#endif
