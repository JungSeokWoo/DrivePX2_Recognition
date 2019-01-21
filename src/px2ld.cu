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

void px2LD::Init(float32_t thresVal, string invRectMapFilePath, string ipmMatrixFilePath)
{
    cv::FileStorage fs(invRectMapFilePath, cv::FileStorage::READ);
    fs["invMap1"] >> mInvMap1;
    fs["invMap2"] >> mInvMap2;
    fs.release();

    cv::FileStorage fs2(ipmMatrixFilePath, cv::FileStorage::READ);
    fs2["ipmMat"] >> mIPMMat;
    fs2.release();

    Init(thresVal);
}

vector<dwVector2f> px2LD::DistortList2RectifiedList(vector<dwVector2f> distortionCoordList)
{
    vector<dwVector2f> rectifiedCoordList;

    for(uint ptIdx = 0U; ptIdx < distortionCoordList.size(); ptIdx++)
    {

        dwVector2f rectifiedCoord = Dist2Rect(distortionCoordList[ptIdx]);

        if((rectifiedCoord.x >= 0) && (rectifiedCoord.y > 0))
        {
            rectifiedCoordList.push_back(rectifiedCoord);
        }
    }

    return rectifiedCoordList;
}

dwVector2f px2LD::Dist2Rect(dwVector2f distortionCoord)
{
    int x = (int)distortionCoord.x;
    int y = (int)distortionCoord.y;

    int oriX = mInvMap1.at<int>(y,x);
    int oriY = mInvMap2.at<int>(y,x);

    dwVector2f rectifiedCoord;
    if((oriX == 0) || (oriY == 0))
    {
        rectifiedCoord.x = -1;
        rectifiedCoord.y = -1;
    }
    else
    {
        rectifiedCoord.x = oriX;
        rectifiedCoord.y = oriY;
    }

    return rectifiedCoord;
}

vector<dwVector2f> px2LD::RectifiedList2TopviewList(vector<dwVector2f> rectifiedCoordList)
{
    vector<dwVector2f> topViewPtList;

    for(uint ptIdx = 0U; ptIdx < rectifiedCoordList.size(); ptIdx++)
    {
        dwVector2f topViewPt = Rect2Topview(rectifiedCoordList[ptIdx]);

        topViewPtList.push_back(topViewPt);
    }

    return topViewPtList;
}


dwVector2f px2LD::Rect2Topview(dwVector2f rectifiedCoord)
{
    double H11 = mIPMMat.at<double>(0, 0);
    double H12 = mIPMMat.at<double>(0, 1);
    double H13 = mIPMMat.at<double>(0, 2);
    double H21 = mIPMMat.at<double>(1, 0);
    double H22 = mIPMMat.at<double>(1, 1);
    double H23 = mIPMMat.at<double>(1, 2);
    double H31 = mIPMMat.at<double>(2, 0);
    double H32 = mIPMMat.at<double>(2, 1);
    double H33 = mIPMMat.at<double>(2, 2);

    float x = rectifiedCoord.x;
    float y = rectifiedCoord.y;

    dwVector2f topViewPt;

    topViewPt.x = (H11*x + H12*y + H13)/(H31*x + H32*y + H33);
    topViewPt.y = (H21*x + H22*y + H23)/(H31*x + H32*y + H33);

    return topViewPt;
}

vector<float> px2LD::TopviewList2Eq(vector<dwVector2f> topviewList)
{
    vector<float> eqCoeffs = laneFitter.FitLine(topviewList, 3);

    return eqCoeffs;
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
