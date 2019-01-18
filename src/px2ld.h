#ifndef PX2LD_H
#define PX2LD_H

#include "px2camlib.h"

#include <dw/dnn/LaneNet.h>
#include <dw/laneperception/LaneDetector.h>

class px2LD{
public:
    px2LD(px2Cam* _px2Cam);
    ~px2LD();

    void Init(float32_t thresVal);

    void DetectLanesByDW(dwImageCUDA* dwLDInputImg,
                         vector<vector<dwVector2f> >& outputLDPtsPerLane,
                         vector<dwVector4f>& outputLDColorPerLane,
                         vector<string>& outputLDPositionNamePerLane,
                         vector<string>& outputLDTypeNamePerLane);

    void DetectLanesByJUNG(float* trtLDInputImg);

    void DetectLanesByHarmony(dwImageCUDA* dwLDInputImg,
                             float* trtLDInputImg);

private:
    dwVector4f GetLaneMarkingColor(dwLanePositionType positionType);

private:
    px2Cam* mPx2Cam;

    cudaStream_t mCudaStream = 0;

    dwLaneNetHandle_t mLaneNet = DW_NULL_HANDLE;
    dwLaneDetectorHandle_t mLaneDetector = DW_NULL_HANDLE;
    dwLaneDetection mLaneDetectionResult;
    float32_t mThresVal = 0.3f;

    const dwImageCUDA* mLDInputImg;

    float32_t mLaneColor[4];
};

#endif // PX2LD_H

