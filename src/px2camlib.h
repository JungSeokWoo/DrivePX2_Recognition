#ifndef PX2CAMLIB_H
#define PX2CAMLIB_H

#include <iostream>
#include <thread>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>
#include <dw/core/NvMedia.h>

// HAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/SensorSerializer.h>
#include <dw/sensors/camera/Camera.h>

// Image
#include <dw/image/ImageStreamer.h>

// Renderer
#include <dw/renderer/Renderer.h>

// DW Framework
#include <framework/ProgramArguments.hpp>
#include <framework/WindowGLFW.hpp>
#include <framework/Checks.hpp>
#include <dw/renderer/RenderEngine.h>

#include <dw/isp/SoftISP.h>

#include "common_cv.h"

#include "img_dev.h"

#define CAM_IMG_WIDTH 1920
#define CAM_IMG_HEIGHT 1208
#define CUDA_PITCH 7680
#define CAM_IMG_SIZE CAM_IMG_WIDTH*CAM_IMG_HEIGHT

using namespace std;

typedef enum { MASTER_TEGRA = 0,
               SLAVE_TEGRA = 1
}dwTegraMode;

typedef enum {GMSL_CAM_YUV = 0,
              GMSL_CAM_RAW = 1,
              H264_FILE = 2,
              RAW_FILE = 3
}dwCamInputMode;

typedef struct {
    dwCamInputMode camInputMode;
    string filePath = "";
}camInputParameters;


typedef struct {
    float resizeRatio = 1.f;
    int roiX = 0;
    int roiY = 0;
    int roiW = CAM_IMG_WIDTH;
    int roiH = CAM_IMG_HEIGHT;
}imgCropParameters;

typedef struct {
    bool onDisplay = true;
    string windowTitle = "";
    int windowWidth = 1280;
    int windowHeight = 720;
}displayParameters;

typedef struct{
    uint64_t timestamp_us = 0;
    float* trtImg;
}trtImgData;

typedef struct{
    uint64_t timestamp_us = 0;
    cv::Mat matImg;
}matImgData;


class px2Cam
{
public:
    px2Cam();
    ~px2Cam();

public:
    bool Init(camInputParameters camInputParams,
              imgCropParameters imgCropParams,
              displayParameters dispParams,
              dwTegraMode tegraMode);

    bool Init(camInputParameters camInputParams,
              imgCropParameters imgCropParams,
              displayParameters dispParams,
              dwTegraMode tegraMode,
              const char* writePath);

    bool UpdateCamImg();
    void RenderCamImg();
    void DrawBoundingBoxes(vector<cv::Rect>  bbRectList, vector<float32_t*> bbColorList, float32_t lineWidth);
    void DrawBoundingBoxesWithLabels(vector<cv::Rect>  bbRectList, vector<float32_t*> bbColorList, vector<const char*> bbLabelList, float32_t lineWidth);
    void DrawBoundingBoxesWithLabelsPerClass(vector<vector<dwRectf> >  bbRectList, vector<const float32_t*> bbColorList, vector<vector<const char*> > bbLabelList, float32_t lineWidth);
    void DrawPoints(vector<cv::Point> ptList, float32_t ptSize, float32_t* ptColor);
    void DrawPolyLine(vector<cv::Point> ptList, float32_t lineWidth, float32_t* lineColor);
    void DrawPolyLineDw(vector<dwVector2f> ptList, float32_t lineWidth, dwVector4f lineColor);
    void DrawText(const char* text, cv::Point textPos, float32_t* textColor);

    void UpdateRendering();

    dwContextHandle_t GetDwContext();
    trtImgData GetTrtImgData();
    matImgData GetCroppedMatImgData();
    matImgData GetOriMatImgData();
    dwImageCUDA* GetDwImageCuda();

    void CoordTrans_Resize2Ori(int xIn, int yIn, int& xOut, int& yOut);
    void CoordTrans_ResizeAndCrop2Ori(float xIn, float yIn, float &xOut, float &yOut);

public:
    ProgramArguments mArguments;

protected:
    void InitGL();
    bool InitSDK();
    bool InitRenderer();
    bool InitSAL();
    bool InitSensors();
    bool InitPipeline();

    void ReleaseModules();


private:
    dwContextHandle_t mContext = DW_NULL_HANDLE;
    dwSALHandle_t mSAL = DW_NULL_HANDLE;
    dwRendererHandle_t mRenderer = DW_NULL_HANDLE;
    dwSensorHandle_t mCamera = DW_NULL_HANDLE;
    dwImageProperties mCamImgProp;
    dwCameraProperties mCamProp;
    dwImageStreamerHandle_t mStreamerCUDA2GL = DW_NULL_HANDLE;
    dwCameraFrameHandle_t mFrameHandle = DW_NULL_HANDLE;
    dwImageHandle_t mFrameCUDAHandle = DW_NULL_HANDLE;
    dwImageHandle_t mFrameGLHandle = DW_NULL_HANDLE;
    dwSensorSerializerHandle_t mSerializer = DW_NULL_HANDLE;

    bool mRecordCamera = false;
    bool mResizeEnable = false;
    dwTegraMode mTegraMode = MASTER_TEGRA;

    float mResizeRatio = 1.f;
    int mResizeWidth = CAM_IMG_WIDTH;
    int mResizeHeight = CAM_IMG_HEIGHT;
    int mROIx;
    int mROIy;
    int mROIw;
    int mROIh;

    WindowBase* mWindow = nullptr;
    dwRenderEngineHandle_t mRenderEngine = DW_NULL_HANDLE;

    uint32_t sibling = 0;
    dwTime_t timeout_us = 40000;

    dwImageCUDA* mCamImgCuda;
    uint8_t* mPitchedImgCudaRGBA;
    uint8_t* mGpuMat_data;
    cv::cuda::GpuMat mGpuMat;
    uint8_t* mGpuMatResized_data;
    cv::cuda::GpuMat mGpuMatResized;
    float* mTrtImg;
    uint8_t* mGpuMatResizedAndCropped_data;
    cv::cuda::GpuMat mGpuMatResizedAndCropped;
    cv::Mat mMatResizedAndCropped;
    cv::Mat mMatOri;

    uint64_t mCamTimestamp = 0;
    trtImgData mCurTrtImgData;
    matImgData mCurCroppedMatImgData;
    matImgData mCurOriMatImgData;

    displayParameters mDispParams;
    dwImageGL* mImgGl;

    camInputParameters mCamInputParams;

    // For Raw
    dwSoftISPHandle_t mISP = DW_NULL_HANDLE;
    uint32_t mISPoutput;
    dwImageCUDA* mCamImgCudaRaw;
    dwImageCUDA* mCamImgCudaRCB;
    dwImageHandle_t mRawImageHandle = DW_NULL_HANDLE;
    dwImageHandle_t mRCBImageHandle = DW_NULL_HANDLE;
    dwImageProperties mRCBImgProp{};

};



#endif // PX2CAMLIB_H

