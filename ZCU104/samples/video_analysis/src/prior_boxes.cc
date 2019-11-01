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
#include "prior_boxes.h"

namespace deephi {

using std::copy_n;
using std::fill_n;
using std::make_pair;
using std::make_shared;
using std::sqrt;
using std::vector;

PriorBoxes::PriorBoxes(int image_width, int image_height, int layer_width,
                       int layer_height, const vector<float>& variances,
                       const vector<float>& min_sizes,
                       const vector<float>& max_sizes,
                       const vector<float>& aspect_ratios, float offset,
                       float step_width, float step_height, bool flip,
                       bool clip)
    : offset_(offset), clip_(clip) {
    // Store image dimensions and layer dimensions
    image_dims_ = make_pair(image_width, image_height);
    layer_dims_ = make_pair(layer_width, layer_height);

    // Compute step width and height
    if (step_width == 0 || step_height == 0) {
        step_dims_ = make_pair(
            static_cast<float>(image_dims_.first) / layer_dims_.first,
            static_cast<float>(image_dims_.second) / layer_dims_.second);
    } else {
        step_dims_ = make_pair(step_width, step_height);
    }

    // Store box variances
    if (variances.size() == 4) {
        variances_ = variances;
    } else if (variances.size() == 1) {
        variances_.resize(4);
        fill_n(variances_.begin(), 4, variances[0]);
    } else {
        variances_.resize(4);
        fill_n(variances_.begin(), 4, 0.1f);
    }

    // Generate boxes' dimensions
    for (size_t i = 0; i < min_sizes.size(); ++i) {
        // first prior: aspect_ratio = 1, size = min_size
        boxes_dims_.emplace_back(min_sizes[i], min_sizes[i]);
        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
        if (!max_sizes.empty()) {
            boxes_dims_.emplace_back(sqrt(min_sizes[i] * max_sizes[i]),
                                     sqrt(min_sizes[i] * max_sizes[i]));
        }
        // rest of priors
        for (auto ar : aspect_ratios) {
            float w = min_sizes[i] * sqrt(ar);
            float h = min_sizes[i] / sqrt(ar);
            boxes_dims_.emplace_back(w, h);
            if (flip) boxes_dims_.emplace_back(h, w);
        }
    }

    // automatically create priors
    CreatePriors();
}

void PriorBoxes::CreatePriors() {
    for (int h = 0; h < layer_dims_.second; ++h) {
        for (int w = 0; w < layer_dims_.first; ++w) {
            float center_x = (w + offset_) * step_dims_.first;
            float center_y = (h + offset_) * step_dims_.second;
            for (auto& dims : boxes_dims_) {
                auto box = make_shared<vector<float> >(12);
                // xmin, ymin, xmax, ymax
                (*box)[0] = (center_x - dims.first / 2.) / image_dims_.first;
                (*box)[1] = (center_y - dims.second / 2.) / image_dims_.second;
                (*box)[2] = (center_x + dims.first / 2.) / image_dims_.first;
                (*box)[3] = (center_y + dims.second / 2.) / image_dims_.second;

                if (clip_) {
                    for (int i = 0; i < 4; ++i)
                        (*box)[i] = std::min(std::max((*box)[i], 0.f), 1.f);
                }
                // variances
                copy_n(variances_.begin(), 4, box->data() + 4);
                // centers and dimensions
                (*box)[8] = 0.5f * ((*box)[0] + (*box)[2]);
                (*box)[9] = 0.5f * ((*box)[1] + (*box)[3]);
                (*box)[10] = (*box)[2] - (*box)[0];
                (*box)[11] = (*box)[3] - (*box)[1];

                priors_.push_back(std::move(box));
            }
        }
    }
}

}
