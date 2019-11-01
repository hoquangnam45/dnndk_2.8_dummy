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

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <thread>
#include <tuple>
#include "ssd_detector.h"

namespace deephi {

using namespace cv;
using namespace std;

SSDdetector::SSDdetector(unsigned int num_classes,  // int background_label_id,
                         CodeType code_type, bool variance_encoded_in_target,
                         unsigned int keep_top_k,
                         const vector<float>& confidence_threshold,
                         unsigned int nms_top_k, float nms_threshold, float eta,
                         const vector<shared_ptr<vector<float>>>& priors,
                         float scale, bool clip)
    : num_classes_(num_classes),
      code_type_(code_type),
      variance_encoded_in_target_(variance_encoded_in_target),
      keep_top_k_(keep_top_k),
      confidence_threshold_(confidence_threshold),
      nms_top_k_(nms_top_k),
      nms_threshold_(nms_threshold),
      eta_(eta),
      priors_(priors),
      scale_(scale),
      clip_(clip) {
    num_priors_ = priors_.size();
    nms_confidence_ = *std::min_element(confidence_threshold_.begin() + 1,
                                        confidence_threshold_.end());
}

template <typename T>
void SSDdetector::Detect(const T* loc_data, const float* conf_data,
                         MultiDetObjects* result) {
    decoded_bboxes_.clear();
    const T(*bboxes)[4] = (const T(*)[4])loc_data;

    unsigned int num_det = 0;
    vector<vector<int>> indices(num_classes_);
    vector<vector<pair<float, int>>> score_index_vec(num_classes_);

    // Get top_k scores (with corresponding indices).
    GetMultiClassMaxScoreIndexMT(conf_data, 1, num_classes_ - 1,
                                 &score_index_vec);

    for (size_t c = 1; c < num_classes_; ++c) {
        // Perform NMS for one class
        ApplyOneClassNMS(bboxes, conf_data, c, score_index_vec[c],
                         &(indices[c]));

        num_det += indices[c].size();
    }

    if (keep_top_k_ > 0 && num_det > keep_top_k_) {
        vector<tuple<float, int, int>> score_index_tuples;
        for (size_t label = 0; label < num_classes_; ++label) {
            const vector<int>& label_indices = indices[label];
            for (size_t j = 0; j < label_indices.size(); ++j) {
                auto idx = label_indices[j];
                auto score = conf_data[idx * num_classes_ + label];
                score_index_tuples.emplace_back(score, label, idx);
            }
        }

        // Keep top k results per image.
        std::sort(score_index_tuples.begin(), score_index_tuples.end(),
                  [](const tuple<float, int, int>& lhs,
                     const tuple<float, int, int>& rhs) {
                      return get<0>(lhs) > get<0>(rhs);
                  });
        score_index_tuples.resize(keep_top_k_);

        indices.clear();
        indices.resize(num_classes_);
        for (auto& item : score_index_tuples) {
            indices[get<1>(item)].push_back(get<2>(item));
        }

        num_det = keep_top_k_;
    }

    for (size_t label = 1; label < indices.size(); ++label) {
        for (size_t idx : indices[label]) {
            auto score = conf_data[idx * num_classes_ + label];
            if (score < confidence_threshold_[label]) {
                continue;
            }
            auto& bbox = decoded_bboxes_[idx];
            bbox[0] = std::max(std::min(bbox[0], 1.f), 0.f);
            bbox[1] = std::max(std::min(bbox[1], 1.f), 0.f);
            bbox[2] = std::max(std::min(bbox[2], 1.f), 0.f);
            bbox[3] = std::max(std::min(bbox[3], 1.f), 0.f);

            auto box_rect = Rect_<float>(Point2f(bbox[0], bbox[1]),
                                         Point2f(bbox[2], bbox[3]));
            result->emplace_back(label, score, box_rect);
        }
    }

}

template void SSDdetector::Detect(const int* loc_data, const float* conf_data,
                                  MultiDetObjects* result);
template void SSDdetector::Detect(const int8_t* loc_data,
                                  const float* conf_data,
                                  MultiDetObjects* result);

template <typename T>
void SSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {

    // Do nms.
    float adaptive_threshold = nms_threshold_;
    indices->clear();
    unsigned int i = 0;
    while (i < score_index_vec.size()) {
        const int idx = score_index_vec[i].second;
        if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
            DecodeBBox(bboxes, idx, true);
        }

        bool keep = true;
        for (size_t k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes, idx, kept_idx);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        ++i;
        if (keep && eta_ < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta_;
        }
    }
}

template void SSDdetector::ApplyOneClassNMS(
    const int (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);
template void SSDdetector::ApplyOneClassNMS(
    const int8_t (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);

void SSDdetector::GetOneClassMaxScoreIndex(
    const float* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
    conf_data += label;
    for (int i = 0; i < num_priors_; ++i) {
        auto score = *conf_data;
        if (score > nms_confidence_) {
            score_index_vec->emplace_back(score, i);
        }
        conf_data += num_classes_;
    }

    std::stable_sort(
        score_index_vec->begin(), score_index_vec->end(),
        [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
            return lhs.first > rhs.first;
        });

    if (nms_top_k_ < score_index_vec->size()) {
        score_index_vec->resize(nms_top_k_);
    }
}

void SSDdetector::GetMultiClassMaxScoreIndex(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
    for (auto i = start_label; i < start_label + num_classes; ++i) {
        GetOneClassMaxScoreIndex(conf_data, i, &((*score_index_vec)[i]));
    }
}

void SSDdetector::GetMultiClassMaxScoreIndexMT(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec, int threads) {

    int thread_classes = num_classes / threads;
    int last_thread_classes = num_classes % threads + thread_classes;

    vector<std::thread> workers;

    auto c = start_label;
    for (auto i = 0; i < threads - 1; ++i) {
        workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                             conf_data, c, thread_classes, score_index_vec);
        c += thread_classes;
    }
    workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                         conf_data, c, last_thread_classes, score_index_vec);

    for (auto& worker : workers)
        if (worker.joinable()) worker.join();
}

void BBoxSize(vector<float>& bbox, bool normalized) {
    float width = bbox[2] - bbox[0];
    float height = bbox[3] - bbox[1];
    if (width > 0 && height > 0) {
        if (normalized) {
            bbox[4] = width * height;
        } else {
            bbox[4] = (width + 1) * (height + 1);
        }
    } else {
        bbox[4] = 0.f;
    }
}

float IntersectBBoxSize(const vector<float>& bbox1, const vector<float>& bbox2,
                        bool normalized) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
        bbox2[3] < bbox1[1]) {
        // Return 0 if there is no intersection.
        return 0.f;
    }

    vector<float> intersect_bbox(5);
    intersect_bbox[0] = max(bbox1[0], bbox2[0]);
    intersect_bbox[1] = max(bbox1[1], bbox2[1]);
    intersect_bbox[2] = min(bbox1[2], bbox2[2]);
    intersect_bbox[3] = min(bbox1[3], bbox2[3]);
    BBoxSize(intersect_bbox, normalized);
    return intersect_bbox[4];
}

template <typename T>
float SSDdetector::JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                                  bool normalized) {

    const vector<float>& bbox1 = decoded_bboxes_[idx];
    const vector<float>& bbox2 = decoded_bboxes_[kept_idx];
    float intersect_size = IntersectBBoxSize(bbox1, bbox2, normalized);
    return intersect_size <= 0
               ? 0
               : intersect_size / (bbox1[4] + bbox2[4] - intersect_size);
}

template float SSDdetector::JaccardOverlap(const int (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);
template float SSDdetector::JaccardOverlap(const int8_t (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);

template <typename T>
void SSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
    vector<float> bbox(5, 0);
    // scale bboxes
    transform(bboxes[idx], bboxes[idx] + 4, bbox.begin(),
              std::bind2nd(multiplies<float>(), scale_));

    auto& prior_bbox = priors_[idx];

    if (code_type_ == CodeType::CORNER) {
        if (variance_encoded_in_target_) {
            // variance is encoded in target, we simply need to add the offset predictions.
            transform(bbox.begin(), bbox.end(), prior_bbox->begin(),
                      bbox.begin(), plus<float>());
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4,
                      bbox.begin(), multiplies<float>());
            transform(bbox.begin(), bbox.end(), prior_bbox->begin(),
                      bbox.begin(), plus<float>());
        }
    } else if (code_type_ == CodeType::CENTER_SIZE) {
        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target_) {
            // variance is encoded in target, we simply need to retore the offset predictions.
            decode_bbox_center_x =
                bbox[0] * (*prior_bbox)[10] + (*prior_bbox)[8];
            decode_bbox_center_y =
                bbox[1] * (*prior_bbox)[11] + (*prior_bbox)[9];
            decode_bbox_width = exp(bbox[2]) * (*prior_bbox)[10];
            decode_bbox_height = exp(bbox[3]) * (*prior_bbox)[11];
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
                (*prior_bbox)[4] * bbox[0] * (*prior_bbox)[10] +
                (*prior_bbox)[8];
            decode_bbox_center_y =
                (*prior_bbox)[5] * bbox[1] * (*prior_bbox)[11] +
                (*prior_bbox)[9];
            decode_bbox_width =
                exp((*prior_bbox)[6] * bbox[2]) * (*prior_bbox)[10];
            decode_bbox_height =
                exp((*prior_bbox)[7] * bbox[3]) * (*prior_bbox)[11];
        }

        bbox[0] = decode_bbox_center_x - decode_bbox_width / 2.;
        bbox[1] = decode_bbox_center_y - decode_bbox_height / 2.;
        bbox[2] = decode_bbox_center_x + decode_bbox_width / 2.;
        bbox[3] = decode_bbox_center_y + decode_bbox_height / 2.;
    } else if (code_type_ == CodeType::CORNER_SIZE) {
        if (variance_encoded_in_target_) {
            // variance is encoded in target, we simply need to add the offset predictions.
            bbox[0] *= (*prior_bbox)[10];
            bbox[1] *= (*prior_bbox)[11];
            bbox[2] *= (*prior_bbox)[10];
            bbox[3] *= (*prior_bbox)[11];
            transform(bbox.begin(), bbox.end(), prior_bbox->begin(),
                      bbox.begin(), plus<float>());
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            bbox[0] *= (*prior_bbox)[10];
            bbox[1] *= (*prior_bbox)[11];
            bbox[2] *= (*prior_bbox)[10];
            bbox[3] *= (*prior_bbox)[11];
            transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4,
                      bbox.begin(), multiplies<float>());
            transform(bbox.begin(), bbox.end(), prior_bbox->begin(),
                      bbox.begin(), plus<float>());
        }
    } else {
    }

    BBoxSize(bbox, normalized);

    decoded_bboxes_.emplace(idx, std::move(bbox));
}

template void SSDdetector::DecodeBBox(const int (*bboxes)[4], int idx,
                                      bool normalized);
template void SSDdetector::DecodeBBox(const int8_t (*bboxes)[4], int idx,
                                      bool normalized);
}
