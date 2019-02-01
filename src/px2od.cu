#include "px2od.h"

px2OD::px2OD(px2Cam *_px2Cam)
{
    mPx2Cam = _px2Cam;
}

px2OD::~px2OD()
{

}

void px2OD::Init()
{
    CHECK_DW_ERROR(dwDriveNet_initDefaultParams(&mDriveNetParams));

    mDriveNetParams.maxClustersPerClass = mMaxClustersPerClass;
    mDriveNetParams.maxProposalsPerClass = mMaxProposalsPerClass;
    mDriveNetParams.networkModel = DW_DRIVENET_MODEL_FRONT;
    mDriveNetParams.batchSize = DW_DRIVENET_BATCH_SIZE_1;
    mDriveNetParams.networkPrecision = DW_PRECISION_FP32;

    CHECK_DW_ERROR(dwDriveNet_initialize(&mDriveNet, &mObjectClusteringHandles,
                                         &mDriveNetClasses,
                                         &mNumDriveNetClasses,
                                         &mDriveNetParams, mPx2Cam->GetDwContext()));

    // Initialize Object Detector from DriveNet
    CHECK_DW_ERROR(dwObjectDetector_initDefaultParams(&mDetectorParams));
    mDetectorParams.enableFuseObjects = false;
    mDetectorParams.maxNumImages = 1;

    CHECK_DW_ERROR(dwObjectDetector_initializeFromDriveNet(&mDriveNetDetector, &mDetectorParams,
                                                           mDriveNet, mPx2Cam->GetDwContext()));

    CHECK_DW_ERROR(dwObjectDetector_setCUDAStream(mCudaStream, mDriveNetDetector));


    float32_t driveNetInputAR = 1.0f;
    dwBlobSize driveNetInputBlob;
    CHECK_DW_ERROR(dwDriveNet_getInputBlobsize(&driveNetInputBlob, mDriveNet));

    driveNetInputAR = static_cast<float32_t>(driveNetInputBlob.height) / static_cast<float32_t>(driveNetInputBlob.width);

    dwRect driveNetROI;

    driveNetROI = {0, 0, static_cast<int32_t>(CAM_IMG_WIDTH), static_cast<int32_t>(CAM_IMG_WIDTH*driveNetInputAR)};

    dwTransformation2D driveNetROITrans ={{1.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f}};

    CHECK_DW_ERROR(dwObjectDetector_setROI(0, &driveNetROI, &driveNetROITrans, mDriveNetDetector));

    CHECK_DW_ERROR(dwObjectDetector_getROI(&mDetectorParams.ROIs[0], &mDetectorParams.transformations[0], 0, mDriveNetDetector));

    mDetectorROI.x = mDetectorParams.ROIs[0].x;
    mDetectorROI.y = mDetectorParams.ROIs[0].y;
    mDetectorROI.width = mDetectorParams.ROIs[0].width;
    mDetectorROI.height = mDetectorParams.ROIs[0].height;

    CHECK_DW_ERROR(dwObjectDetector_bindInput(&mODInputImg, 1, mDriveNetDetector));

    for(uint32_t classIdx = 0; classIdx < mNumDriveNetClasses; ++classIdx)
    {
        mDetectorOutputObjects[classIdx].reset(new dwObjectHandle_t[MAX_OBJECT_OUTPUT_COUNT]);
        mClustererOutputObjects[classIdx].reset(new dwObjectHandle_t[MAX_OBJECT_OUTPUT_COUNT]);

        // Initialize each object handle
        for (uint32_t objIdx = 0U; objIdx < MAX_OBJECT_OUTPUT_COUNT; ++objIdx)
        {
            dwObjectData objectData{};
            dwObjectDataCamera objectDataCamera{};
            CHECK_DW_ERROR(dwObject_createCamera(&mDetectorOutputObjects[classIdx][objIdx], &objectData, &objectDataCamera));
            CHECK_DW_ERROR(dwObject_createCamera(&mClustererOutputObjects[classIdx][objIdx], &objectData, &objectDataCamera));
        }

        mDetectorOutput[classIdx].count = 0;
        mDetectorOutput[classIdx].objects = mDetectorOutputObjects[classIdx].get();
        mDetectorOutput[classIdx].maxCount = MAX_OBJECT_OUTPUT_COUNT;
        mClustererOutput[classIdx].count = 0;
        mClustererOutput[classIdx].objects = mClustererOutputObjects[classIdx].get();
        mClustererOutput[classIdx].maxCount = MAX_OBJECT_OUTPUT_COUNT;

        CHECK_DW_ERROR(dwObjectDetector_bindOutput(&mDetectorOutput[classIdx], 0, classIdx, mDriveNetDetector));
        CHECK_DW_ERROR(dwObjectClustering_bindInput(&mDetectorOutput[classIdx], mObjectClusteringHandles[classIdx]));
        CHECK_DW_ERROR(dwObjectClustering_bindOutput(&mClustererOutput[classIdx], mObjectClusteringHandles[classIdx]));
    }

    // Initialize box list
    mDnnBoxList.resize(mNumDriveNetClasses);
    mDnnLabelList.resize(mNumDriveNetClasses);
    mDnnLabelListPtr.resize(mNumDriveNetClasses);
    mDnnConfidence.resize(mNumDriveNetClasses);
    mDnnObjectID.resize(mNumDriveNetClasses);

    // Get which label name for each class id
    mClassLabels.resize(mNumDriveNetClasses);
    for(uint32_t classIdx = 0U; classIdx < mNumDriveNetClasses; ++classIdx)
    {
        const char* classLabel;
        CHECK_DW_ERROR(dwDriveNet_getClassLabel(&classLabel, classIdx, mDriveNet));
        mClassLabels[classIdx] = classLabel;

        // Reserve label and box lists
        mDnnBoxList[classIdx].reserve(mMaxClustersPerClass);
        mDnnLabelList[classIdx].reserve(mMaxClustersPerClass);
        mDnnLabelListPtr[classIdx].reserve(mMaxClustersPerClass);
        mDnnConfidence[classIdx].reserve(mMaxClustersPerClass);
        mDnnObjectID[classIdx].reserve(mMaxClustersPerClass);
    }
}

void px2OD::DetectObjects(dwImageCUDA* dwODInputImg,
                   vector<vector<dwRectf> >& outputODRectPerClass,
                   vector<const float32_t*>& outputODRectColorPerClass,
                   vector<vector<const char*> >& outputODLabelPerClass,
                   vector<vector<float32_t> >& outputODConfidencePerClass,
                   vector<vector<int> >& outputODIDPerClass)
{
    mODInputImg = dwODInputImg;
    CHECK_DW_ERROR(dwObjectDetector_processDeviceAsync(mDriveNetDetector));

    CHECK_DW_ERROR(dwObjectDetector_processHost(mDriveNetDetector));

    for (uint32_t classIdx = 0U; classIdx < mClassLabels.size(); ++classIdx)
    {
        CHECK_DW_ERROR(dwObjectClustering_process(mObjectClusteringHandles[classIdx]));

        // Get outputs of object clustering
        mDnnLabelListPtr[classIdx].clear();
        mDnnLabelList[classIdx].clear();
        mDnnBoxList[classIdx].clear();
        mDnnConfidence[classIdx].clear();
        mDnnObjectID[classIdx].clear();

        dwObjectHandleList clusters = mClustererOutput[classIdx];

        for (uint32_t objIdx = 0U; objIdx < clusters.count; ++objIdx)
        {
            dwObjectHandle_t obj = clusters.objects[objIdx];
            dwObjectDataCamera objCameraData{};
            dwObject_getDataCamera(&objCameraData, 0, obj);
            mDnnBoxList[classIdx].push_back(objCameraData.box2D);
            mDnnConfidence[classIdx].push_back(objCameraData.classConfidence);

            dwObjectData objData{};
            dwObject_getData(&objData, 0, obj);
            mDnnObjectID[classIdx].push_back(objData.id);

            string boxAnnot = mClassLabels[classIdx];
            mDnnLabelList[classIdx].push_back(boxAnnot);
            mDnnLabelListPtr[classIdx].push_back(mDnnLabelList[classIdx].back().c_str());
        }
    }

    outputODRectPerClass = mDnnBoxList;
    outputODRectColorPerClass = vector<const float*>(mOdBoxColorList, mOdBoxColorList + sizeof mOdBoxColorList/ sizeof mOdBoxColorList[0]);
    outputODLabelPerClass = mDnnLabelListPtr;
    outputODConfidencePerClass = mDnnConfidence;
    outputODIDPerClass = mDnnObjectID;
}
