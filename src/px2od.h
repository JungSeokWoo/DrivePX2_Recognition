#ifndef PX2OD_H
#define PX2OD_H

#include "px2camlib.h"

#include <dw/dnn/DriveNet.h>
#include <dw/objectperception/camera/ObjectDetector.h>

class px2OD{
public:
    px2OD(px2Cam* _px2Cam);
    ~px2OD();

    void Init();

    void DetectObjects(dwImageCUDA* dwODInputImg,
                       vector<vector<dwRectf> >& outputODRectPerClass,
                       vector<const float32_t*>& outputODRectColorPerClass,
                       vector<vector<const char*> >& outputODLabelPerClass,
                       vector<vector<float32_t> >& outputODConfidencePerClass,
                       vector<vector<int> >& outputODIDPerClass);


private:
    px2Cam *mPx2Cam;

    cudaStream_t mCudaStream = 0;
    dwDriveNetHandle_t mDriveNet = DW_NULL_HANDLE;
    dwDriveNetParams mDriveNetParams{};
    const dwDriveNetClass* mDriveNetClasses = nullptr;
    uint32_t mNumDriveNetClasses = 0;
    const uint32_t mMaxProposalsPerClass = 1000U;
    const uint32_t mMaxClustersPerClass = 400U;

    // Detector
    dwObjectDetectorParams mDetectorParams{};
    dwObjectDetectorHandle_t mDriveNetDetector = DW_NULL_HANDLE;
    dwRectf mDetectorROI;

    // Clustering
    dwObjectClusteringHandle_t* mObjectClusteringHandles = nullptr;

    // Colors for rendering bounding boxes
    static const uint32_t mMaxODBoxColors = DW_DRIVENET_NUM_CLASSES;
    const float32_t mOdBoxColorList[mMaxODBoxColors][4] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                                          {0.0f, 1.0f, 0.0f, 1.0f},
                                                          {0.0f, 0.0f, 1.0f, 1.0f},
                                                          {1.0f, 0.0f, 1.0f, 1.0f},
                                                          {1.0f, 0.647f, 0.0f, 1.0f}};
    // Labels of each class
    std::vector<std::string> mClassLabels;

    // Vectors of boxes and class label ids
    vector<vector<dwRectf> > mDnnBoxList;
    vector<vector<string> > mDnnLabelList;
    vector<vector<const char*> > mDnnLabelListPtr;

    vector<vector<float32_t> > mDnnConfidence;
    vector<vector<int> > mDnnObjectID;

    const dwImageCUDA* mODInputImg;
    dwImageCUDA* mCamImgDwCuda = nullptr;
    /// The maximum number of output objects for a given bound output.
    static constexpr uint32_t MAX_OBJECT_OUTPUT_COUNT = 1000;
    dwObjectHandleList mDetectorOutput[DW_OBJECT_MAX_CLASSES];
    dwObjectHandleList mClustererOutput[DW_OBJECT_MAX_CLASSES];


    std::unique_ptr<dwObjectHandle_t[]> mDetectorOutputObjects[DW_OBJECT_MAX_CLASSES];
    std::unique_ptr<dwObjectHandle_t[]> mClustererOutputObjects[DW_OBJECT_MAX_CLASSES];

};



#endif // PX2OD_H

