#ifndef COMMON_CV_H
#define COMMON_CV_H

// For avoiding float16_t declaration confliction
#ifdef __ARM_NEON__
#undef __ARM_NEON__
#endif

#ifdef __ARM_NEON
#undef __ARM_NEON
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

#endif // COMMON_CV_H

