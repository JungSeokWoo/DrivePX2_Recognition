#include "px2ld.h"

px2LD::px2LD(px2Cam *_px2Cam)
{
    mPx2Cam = _px2Cam;
}

px2LD::~px2LD()
{

}

void px2LD::Init(float32_t thresVal)
{
    mThresVal = thresVal;
    dwLaneNetParams laneNetParams{};

    // Initialize LaneNet network
    CHECK_DW_ERROR(dwLaneNet_initDefaultParams(&laneNetParams));
    CHECK_DW_ERROR(dwLaneNet_initialize(&mLaneNet, &laneNetParams, mPx2Cam->GetDwContext()));

    // Initialize LaneDetector from LaneNet
    CHECK_DW_ERROR(dwLaneDetector_initializeFromLaneNet(&mLaneDetector, mLaneNet,
                                                        CAM_IMG_WIDTH, CAM_IMG_HEIGHT, mPx2Cam->GetDwContext()));

    CHECK_DW_ERROR(dwLaneDetector_setCUDAStream(mCudaStream, mLaneDetector));

    CHECK_DW_ERROR(dwLaneDetector_setDetectionThreshold(mThresVal, mLaneDetector));
}

void px2LD::DetectLanesByDW(dwImageCUDA* dwLDInputImg,
                     vector<vector<dwVector2f> >& outputLDPtsPerLane,
                     vector<dwVector4f>& outputLDColorPerLane,
                     vector<string>& outputLDPositionNamePerLane,
                     vector<string>& outputLDTypeNamePerLane)
{
    mLDInputImg = dwLDInputImg;
    CHECK_DW_ERROR(dwLaneDetector_processDeviceAsync(mLDInputImg, mLaneDetector));
    CHECK_DW_ERROR(dwLaneDetector_interpretHost(mLaneDetector));
    CHECK_DW_ERROR(dwLaneDetector_getLaneDetections(&mLaneDetectionResult, mLaneDetector));

    vector<vector<dwVector2f> > ldPtsPerLane;
    vector<dwVector4f> ldColorPerLane;
    vector<string> ldPositionNamePerLane;
    vector<string> ldTypeNamePerLane;
    for(uint32_t laneIdx = 0U; laneIdx < mLaneDetectionResult.numLaneMarkings; ++laneIdx)
    {
        const dwLaneMarking& laneMarking = mLaneDetectionResult.laneMarkings[laneIdx];

        dwLanePositionType lanePos = laneMarking.positionType;

        vector<dwVector2f> lanePts(laneMarking.imagePoints, laneMarking.imagePoints + laneMarking.numPoints);
        ldPtsPerLane.push_back(lanePts);

        dwVector4f laneColorVec = GetLaneMarkingColor(lanePos);
        ldColorPerLane.push_back(laneColorVec);

        string lanePosNameStr;
        switch(lanePos)
        {
        case DW_LANEMARK_POSITION_ADJACENT_LEFT:
            lanePosNameStr = "LEFT-LEFT";
            break;
        case DW_LANEMARK_POSITION_EGO_LEFT:
            lanePosNameStr = "LEFT";
            break;
        case DW_LANEMARK_POSITION_EGO_RIGHT:
            lanePosNameStr = "RIGHT";
            break;
        case DW_LANEMARK_POSITION_ADJACENT_RIGHT:
            lanePosNameStr = "RIGHT-RIGHT";
            break;
        case DW_LANEMARK_POSITION_UNDEFINED:
            lanePosNameStr = "UNKNOWN";
            break;
        default:
            lanePosNameStr = "UNKNOWN";
            break;
        }
        ldPositionNamePerLane.push_back(lanePosNameStr);

        string laneTypeNameStr;
        switch(laneMarking.lineType)
        {
        case DW_LANEMARK_TYPE_SOLID:
            laneTypeNameStr = "SOLID";
            break;
        case DW_LANEMARK_TYPE_DASHED:
            laneTypeNameStr = "DASHED";
            break;
        case DW_LANEMARK_TYPE_ROAD_BOUNDARY:
            laneTypeNameStr = "ROAD_BOUNDARY";
            break;
        case DW_LANEMARK_TYPE_UNDEFINED:
            laneTypeNameStr = "UNKNOWN";
            break;
        default:
            laneTypeNameStr = "UNKNOWN";
            break;
        }
        ldTypeNamePerLane.push_back(laneTypeNameStr);
    }

    outputLDPtsPerLane = ldPtsPerLane;
    outputLDColorPerLane = ldColorPerLane;
    outputLDPositionNamePerLane = ldPositionNamePerLane;
    outputLDTypeNamePerLane = ldTypeNamePerLane;
}

void px2LD::DetectLanesByJUNG(float* trtLDInputImg)
{
    // TBD
}

void px2LD::DetectLanesByHarmony(dwImageCUDA *dwLDInputImg, float *trtLDInputImg)
{
    // TBD
}

dwVector4f px2LD::GetLaneMarkingColor(dwLanePositionType positionType)
{
    dwVector4f laneColorVector;

    const dwVector4f colorCyan{10.0f/255.0f,   230.0f/255.0f,   230.0f/255.0f, 1.0f};
    const dwVector4f colorDarkYellow{180.0f/255.0f, 180.0f/255.0f,  10.0f/255.0f, 1.0f};

    switch (positionType) {
    case DW_LANEMARK_POSITION_ADJACENT_LEFT:
        laneColorVector = colorCyan;
        break;
    case DW_LANEMARK_POSITION_EGO_LEFT:
        laneColorVector = DW_RENDER_ENGINE_COLOR_RED;
        break;
    case DW_LANEMARK_POSITION_EGO_RIGHT:
        laneColorVector = DW_RENDER_ENGINE_COLOR_GREEN;
        break;
    case DW_LANEMARK_POSITION_ADJACENT_RIGHT:
        laneColorVector = DW_RENDER_ENGINE_COLOR_BLUE;
        break;
    default:
        laneColorVector = colorDarkYellow;
        break;
    }

    return laneColorVector;
}
