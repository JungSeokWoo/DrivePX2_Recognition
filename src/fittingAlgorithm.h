/*
 * fittingAlgorithm.h
 *
 *  Created on: Aug 27, 2018
 *      Author: swjung
 */

#ifndef FITTINGALGORITHM_H_
#define FITTINGALGORITHM_H_

#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include "common_cv.h"

#include <dw/core/Context.h>


using namespace std;
using namespace arma;

class LMSFit{
public:
	LMSFit() {}
    ~LMSFit() {}

    vector<double> FitLine(vector<cv::Point> _pt_list,uint _poly_order);
    vector<float> FitLine(vector<dwVector2f> _pt_list, uint _poly_order);

};

class outlierFilter {
public:
    outlierFilter() {}
    ~outlierFilter() {}

    bool RANSACFilter(vector<cv::Point> _raw_ld_result, uint _poly_order,
    		vector<float>& _out_model_coeff, vector<cv::Point>& _out_inlier_pt_list);
};

#endif /* FITTINGALGORITHM_H_ */
