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

#ifndef DEEPHI_SSD_DETECTOR_H_
#define DEEPHI_SSD_DETECTOR_H_

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>
#include <tuple>

namespace deephi {

using SingleDetObject = std::tuple<int, float, cv::Rect_<float> >;
using MultiDetObjects = std::vector<SingleDetObject>;

class SSDdetector {
public:
    enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

    SSDdetector(
        unsigned int num_classes,  // int background_label_id,
        CodeType code_type, bool variance_encoded_in_target,
        unsigned int keep_top_k, const std::vector<float>& confidence_threshold,
        unsigned int nms_top_k, float nms_threshold, float eta,
        const std::vector<std::shared_ptr<std::vector<float> > >& priors,
        float scale = 1.f, bool clip = false);

    template <typename T>
    void Detect(const T* loc_data, const float* conf_data,
                MultiDetObjects* result);

    unsigned int num_classes() const { return num_classes_; }
    unsigned int num_priors() const { return priors_.size(); }

protected:
    template <typename T>
    void ApplyOneClassNMS(
        const T (*bboxes)[4], const float* conf_data, int label,
        const std::vector<std::pair<float, int> >& score_index_vec,
        std::vector<int>* indices);

    void GetOneClassMaxScoreIndex(
        const float* conf_data, int label,
        std::vector<std::pair<float, int> >* score_index_vec);

    void GetMultiClassMaxScoreIndex(
        const float* conf_data, int start_label, int num_classes,
        std::vector<std::vector<std::pair<float, int> > >* score_index_vec);

    void GetMultiClassMaxScoreIndexMT(
        const float* conf_data, int start_label, int num_classes,
        std::vector<std::vector<std::pair<float, int> > >* score_index_vec,
        int threads = 2);

    template <typename T>
    float JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                         bool normalized = true);

    template <typename T>
    void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

    std::map<int, std::vector<float> > decoded_bboxes_;

    const unsigned int num_classes_;
    CodeType code_type_;
    bool variance_encoded_in_target_;
    unsigned int keep_top_k_;
    std::vector<float> confidence_threshold_;
    float nms_confidence_;
    unsigned int nms_top_k_;
    float nms_threshold_;
    float eta_;

    const std::vector<std::shared_ptr<std::vector<float> > >& priors_;
    float scale_;

    bool clip_;

    int num_priors_;
};

}

#endif
