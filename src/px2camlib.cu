#include "px2camlib.h"

// Convert Cam Img to gpuMat
__global__
void PitchedRGBA2GpuMat(uint8_t* pitchedImgRGBA, uint8_t* imgGpuMat, int width, int height, int cudaPitch)
{
    int xIndex_3ch = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;

    if((xIndex_3ch < 3*width) && (yIndex < height))
    {
        int xIndex = xIndex_3ch%width;
        int c = xIndex_3ch/width;

        int j = yIndex*width + xIndex;
        imgGpuMat[j*3 + 2 - c] = pitchedImgRGBA[cudaPitch*yIndex + c + xIndex*4];
    }
}

// Crop and convert gpuMat original image to Tensor RT and gpuMat
__global__
void GpuMat2Img(uint8_t* imgGpuMatOri, float* imgTrt, uint8_t* imgGpuMat,
                int width, int height,
                int roiX, int roiY, int roiW, int roiH)
{
    int xIndex_3ch = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex_3ch < 3*roiW) && (yIndex < roiH))
    {
        int xIndex = xIndex_3ch%roiW;
        int c = xIndex_3ch/roiW;

        int j = (yIndex)*roiW + (xIndex);
        int j_ori = (yIndex + roiY)*width + (xIndex + roiX);

        imgTrt[c*roiH*roiW + j] = (float)imgGpuMatOri[j_ori*3 + 2 - c]/255.f;
        imgGpuMat[j*3 + 2 - c] = imgGpuMatOri[j_ori*3 + 2 - c];
    }
}

px2Cam::px2Cam()
{
    mArguments = ProgramArguments(
    {           ProgramArguments::Option_t("camera-type", "ar0231-rccb-ae-sf3324"),
                ProgramArguments::Option_t("custom-board", "1"),
                ProgramArguments::Option_t("csi-port", "ab"),
                ProgramArguments::Option_t("write-file", ""),
                ProgramArguments::Option_t("serializer-type", "h264"),
                ProgramArguments::Option_t("serializer-bitrate", "8000000"),
                ProgramArguments::Option_t("serializer-framerate", "30"),
                ProgramArguments::Option_t("slave", "0")
    });
}

px2Cam::~px2Cam()
{
    ReleaseModules();
}

void px2Cam::ReleaseModules()
{
    if(mStreamerCUDA2GL)
    {
        dwImageStreamer_release(&mStreamerCUDA2GL);
    }

    if(mCamera)
    {
        dwSensor_stop(mCamera);
        dwSAL_releaseSensor(&mCamera);
    }

    if(mRenderEngine)
    {
        dwRenderEngine_release(&mRenderEngine);
    }

    if(mRenderer)
    {
        dwRenderer_release(&mRenderer);
    }

    dwSAL_release(&mSAL);
    dwRelease(&mContext);
}

bool px2Cam::Init(imgCropParameters imgCropParams,
                  displayParameters dispParams,
                  dwTegraMode tegraMode,
                  const char* writePath)
{
    mArguments.set("write-file", writePath);
    return Init(imgCropParams, dispParams, tegraMode);
}

bool px2Cam::Init(imgCropParameters imgCropParams,
                  displayParameters dispParams,
                  dwTegraMode tegraMode)
{
    // Set Resize and ROI info

    mRecordCamera = !mArguments.get("write-file").empty();
    mTegraMode = tegraMode;
    mDispParams = dispParams;

    if (abs(imgCropParams.resizeRatio - 1.0) < 0.001)
    {
        mResizeEnable = false;
    }
    else
    {
        mResizeEnable = true;
        mResizeRatio = imgCropParams.resizeRatio;
        mResizeWidth = (int)((float)CAM_IMG_WIDTH*imgCropParams.resizeRatio);
        mResizeHeight = (int)((float)CAM_IMG_HEIGHT*imgCropParams.resizeRatio);
    }

    mROIx = imgCropParams.roiX;
    mROIy = imgCropParams.roiY;
    mROIw = imgCropParams.roiW;
    mROIh = imgCropParams.roiH;

    // Check ROI is inrange of Camera Image
    if(!mResizeEnable)
    {
        int roiBRx = mROIx + mROIw;
        int roiBRy = mROIy + mROIh;

        if((mROIx < 0) && (mROIx > CAM_IMG_WIDTH) ||
                (mROIy < 0) || (mROIy > CAM_IMG_HEIGHT) ||
                (roiBRx < 0) || (roiBRx > CAM_IMG_WIDTH) ||
                (roiBRy < 0) || (roiBRy > CAM_IMG_HEIGHT))
        {
            cout << "ROI is out of range..." << "Camera image resolution : (" << CAM_IMG_WIDTH << "," << CAM_IMG_HEIGHT << ")"
                 << "...But ROI is : " << "(" << mROIx << "," << mROIy << ") ~ (" << roiBRx << "," << roiBRy << ")" <<  endl;
            return false;
        }
    }
    else
    {
        int roiBRx = mROIx + mROIw;
        int roiBRy = mROIy + mROIh;

        int roiTLxOri;
        int roiTLyOri;
        int roiBRxOri;
        int roiBRyOri;

        CoordTrans_Resize2Ori(mROIx, mROIy, roiTLxOri, roiTLyOri);
        CoordTrans_Resize2Ori(roiBRx, roiBRy, roiBRxOri, roiBRyOri);

        if((roiTLxOri < 0) || (roiTLxOri > CAM_IMG_WIDTH) ||
                (roiTLyOri < 0) || (roiTLyOri > CAM_IMG_HEIGHT) ||
                (roiBRxOri < 0) || (roiBRxOri > CAM_IMG_WIDTH) ||
                (roiBRyOri < 0) || (roiBRyOri > CAM_IMG_HEIGHT))
        {
            cout << "ROI is out of range" << "Camera image resolution : (" << CAM_IMG_WIDTH << "," << CAM_IMG_HEIGHT << ")"
                 << "...But ROI is : " << "(" << roiTLxOri << "," << roiTLyOri << ") ~ (" << roiBRxOri << "," << roiBRyOri << ")" <<  endl;
            return false;
        }
    }

    // Initialize Modules
    bool status;
    InitGL();

    status = InitSDK();
    if(!status)
        return status;

    status = InitRenderer();
    if(!status)
        return status;

    status = InitSAL();
    if(!status)
        return status;

    status = InitSensors();
    if(!status)
        return status;

    status = InitPipeline();
    if(!status)
        return status;

    return true;
}

void px2Cam::CoordTrans_Resize2Ori(int xIn, int yIn, int& xOut, int& yOut)
{
    xOut = (int)(xIn/mResizeRatio);
    yOut = (int)(yIn/mResizeRatio);
}

void px2Cam::CoordTrans_ResizeAndCrop2Ori(int xIn, int yIn, int &xOut, int &yOut)
{
    xOut = (int)((xIn + mROIx)/mResizeRatio);
    yOut = (int)((yIn + mROIy)/mResizeRatio);
}

void px2Cam::InitGL()
{
    if(!mWindow)
    {
        mWindow = new WindowGLFW(mDispParams.windowTitle.c_str(), mDispParams.windowWidth, mDispParams.windowHeight, !mDispParams.onDisplay);
    }
    mWindow->makeCurrent();
}

bool px2Cam::InitSDK()
{
    dwStatus status;

    dwContextParameters sdkParams = {};

    sdkParams.eglDisplay = mWindow->getEGLDisplay();

    status = dwInitialize(&mContext, DW_VERSION, &sdkParams);

    if(status == DW_SUCCESS)
    {
        cout << "[DW_INIT_STEP_1] Driveworks init success" << endl;
    }
    else
    {
        cout << "[DW_INIT_STEP_1] Driveworks init fail" << endl;
        return false;
    }

    return true;
}

bool px2Cam::InitRenderer()
{
    dwStatus status;

    status = dwRenderer_initialize(&mRenderer, mContext);

    if(status == DW_SUCCESS)
    {
        cout << "[DW_INIT_STEP_2] Renderer init success" << endl;
    }
    else
    {
        cout << "[DW_INIT_STEP_2] Renderer init fail" << endl;
        return false;
    }

    dwRenderEngineParams renderEngineParams{};
    CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderEngineParams, mWindow->width(), mWindow->height()));

    renderEngineParams.defaultTile.lineWidth = 0.2f;
    renderEngineParams.defaultTile.font = DW_RENDER_ENGINE_FONT_VERDANA_20;

    CHECK_DW_ERROR(dwRenderEngine_initialize(&mRenderEngine, &renderEngineParams, mContext));

    return true;
}

bool px2Cam::InitSAL()
{
    dwStatus status;

    status = dwSAL_initialize(&mSAL, mContext);

    if(status == DW_SUCCESS)
    {
        cout << "[DW_INIT_STEP_3] SAL init success" << endl;
    }
    else
    {
        cout << "[DW_INIT_STEP_3] SAL init fail" << endl;
        return false;
    }

    return true;
}

bool px2Cam::InitSensors()
{
    dwStatus status;

    dwSensorParams sensorParams;
    memset(&sensorParams, 0, sizeof(dwSensorParams));

    std::string parameterString = std::string("output-format=yuv,fifo-size=3");

    parameterString += std::string(",camera-type=") + mArguments.get("camera-type").c_str();
    parameterString += std::string(",csi-port=") + mArguments.get("csi-port").c_str();
    parameterString += std::string(",slave=") + mArguments.get("slave").c_str();

    if (mArguments.get("custom-board").compare("1") == 0)
    {
        // it's a custom board, use the board specific extra configurations
        parameterString             += ",custom-board=1";

        mArguments.addOption("custom-config");
        mArguments.set("custom-config",
                            "board=E2379a-c01,"
                            "moduleName=ref_max9286_96705_ar0231rccbsf3324ae,"
                            "resolution=1920x1208,"
                            "inputFormat=raw12,"
                            "sensorNum=1,"
                            "interface=csi-ab,"
                            "i2cDevice=7,"
                            "desAddr=0x48,"
                            "brdcstSerAddr=0x40,"
                            "brdcstSensorAddr=0x10");

        sensorParams.auxiliarydata  = mArguments.get("custom-config").c_str();
    }


    sensorParams.parameters = parameterString.c_str();
    sensorParams.protocol = "camera.gmsl";

    status = dwSAL_createSensor(&mCamera, sensorParams, mSAL);

    if(status == DW_SUCCESS)
    {
        cout << "[DW_INIT_STEP_4] Camera init success" << endl;
    }
    else
    {
        cout << "[DW_INIT_STEP_4] Camera init fail : " << dwGetStatusName(status) <<endl;
        return false;
    }

    return true;
}

bool px2Cam::InitPipeline()
{
    dwStatus status;

    status = dwSensor_start(mCamera);

    dwCameraFrameHandle_t frame;
    status = DW_NOT_READY;
    do {
        status = dwSensorCamera_readFrame(&frame, 0, 66000, mCamera);
    } while (status == DW_NOT_READY);

    // something wrong happened, aborting
    if (status != DW_SUCCESS) {
        throw std::runtime_error("Cameras did not start correctly");
    }

    status = dwSensorCamera_returnFrame(&frame);

    status = dwSensorCamera_getSensorProperties(&mCamProp, mCamera);
    printf("Successfully initialized camera with resolution of %dx%d at framerate of %f FPS\n"
           ,mCamProp.resolution.x
           ,mCamProp.resolution.y
           ,mCamProp.framerate);

    // Initialize streamer
    dwImageProperties glImgProps{};
    glImgProps.width = mCamProp.resolution.x;
    glImgProps.height = mCamProp.resolution.y;
    glImgProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    glImgProps.type = DW_IMAGE_CUDA;

    status = dwImageStreamer_initialize(&mStreamerCUDA2GL, &glImgProps, DW_IMAGE_GL, mContext);

    if(status == DW_SUCCESS)
    {
        cout << "[DW_INIT_STEP_5] Pipleline init success" << endl;
    }
    else
    {
        cout << "[DW_INIT_STEP_5] Pipleline init fail : " << dwGetStatusName(status) <<endl;
        return false;
    }

    // Init Serializer
    if (mRecordCamera)
    {
        dwSerializerParams seriParams;
        seriParams.parameters = "";
        std::string seriParamsStr = "";
        seriParamsStr += std::string("format=") + std::string(mArguments.get("serializer-type"));
        seriParamsStr += std::string(",bitrate=") + std::string(mArguments.get("serializer-bitrate"));
        seriParamsStr += std::string(",framerate=") + std::string(mArguments.get("serializer-framerate"));
        seriParamsStr += std::string(",type=disk,file=") + std::string(mArguments.get("write-file"));
        seriParamsStr += std::string(",slave=") + std::string(mArguments.get("slave"));

        seriParams.parameters = seriParamsStr.c_str();
        seriParams.onData = nullptr;

        status = dwSensorSerializer_initialize(&mSerializer, &seriParams, mCamera);
        status = dwSensorSerializer_start(mSerializer);

        if(status == DW_SUCCESS)
        {
            cout << "[DW_INIT_STEP_6] Serializer init success" << endl;
        }
        else
        {
            cout << "[DW_INIT_STEP_6] Serializer init fail : " << dwGetStatusName(status) <<endl;
            return false;
        }
    }

    // Allocation Img Data memory
    cudaMalloc(&mPitchedImgCudaRGBA, CUDA_PITCH*CAM_IMG_HEIGHT*sizeof(uint8_t));

    cudaMalloc(&mGpuMat_data, CAM_IMG_WIDTH*CAM_IMG_HEIGHT*3*sizeof(uint8_t));

    mGpuMat = cv::cuda::GpuMat(CAM_IMG_HEIGHT, CAM_IMG_WIDTH, CV_8UC3, (uint8_t*) mGpuMat_data);

    if(mResizeEnable)
    {
        cudaMalloc(&mGpuMatResized_data, mResizeWidth*mResizeHeight*3*sizeof(uint8_t));

        mGpuMatResized = cv::cuda::GpuMat(mResizeHeight, mResizeWidth, CV_8UC3, (uint8_t*)mGpuMatResized_data);
    }

    cudaMalloc(&mTrtImg, mROIw*mROIh*3*sizeof(float));

    cudaMalloc(&mGpuMatResizedAndCropped_data, mROIw*mROIh*3*sizeof(uint8_t));

    mGpuMatResizedAndCropped = cv::cuda::GpuMat(mROIh, mROIw, CV_8UC3, (uint8_t*)mGpuMatResizedAndCropped_data);

    mMatResizedAndCropped = cv::Mat(mROIh, mROIw, CV_8UC3);

    mMatOri = cv::Mat(CAM_IMG_HEIGHT, CAM_IMG_WIDTH, CV_8UC3);

    return true;
}

bool px2Cam::UpdateCamImg()
{
    dwStatus status;

    status = dwSensorCamera_readFrame(&mFrameHandle, sibling, timeout_us, mCamera);

    if (status == DW_END_OF_STREAM)
    {
        cout << "Camera reached end of stream." << endl;
        return false;
    }
    else if((status == DW_NOT_READY) || (status == DW_TIME_OUT)){
        while((status == DW_NOT_READY) || (status == DW_TIME_OUT))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = dwSensorCamera_readFrame(&mFrameHandle, sibling, timeout_us, mCamera);
            printf("."); fflush(stdout);
        }
    }
    else if(status == DW_SUCCESS)
    {
//        cout << "[DW_PROC_STEP_1] Read frame success" << endl;
    }
    else
    {
        cout << "[DW_PROC_STEP_1] Read frame fail : " <<  dwGetStatusName(status) << endl;
    }

//    auto begin = std::chrono::high_resolution_clock::now();

    status = dwSensorCamera_getImage(&mFrameCUDAHandle, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, mFrameHandle);

    if(status == DW_SUCCESS)
    {
//        cout << "[DW_PROC_STEP_2] Get CUDA frame handle success" << endl;
    }
    else
    {
        cout << "[DW_PROC_STEP_2] Get CUDA frame handle fail : " <<  dwGetStatusName(status) << endl;
    }

    if(mRecordCamera)
    {
        status = dwSensorSerializer_serializeCameraFrameAsync(mFrameHandle, mSerializer);

        if(status == DW_SUCCESS)
        {
//            cout << "[DW_PROC_STEP_2.5] Serializing success" << endl;
        }
        else
        {
            cout << "[DW_PROC_STEP_2.5] Serializing fail : " <<  dwGetStatusName(status) << endl;
        }

    }

    status = dwImage_getCUDA(&mCamImgCuda, mFrameCUDAHandle);

    if(status == DW_SUCCESS)
    {
//        cout << "[DW_PROC_STEP_3] Get CUDA frame success" << endl;
    }
    else
    {
        cout << "[DW_PROC_STEP_3] Get CUDA frame fail : " <<  dwGetStatusName(status) << endl;
    }

    // Get Camera image capture time
    mCamTimestamp = mCamImgCuda->timestamp_us;

    // Copy dwImageCUDA to Pitched pointer
    cudaMemcpy(mPitchedImgCudaRGBA, mCamImgCuda->dptr[0], (CUDA_PITCH*CAM_IMG_HEIGHT), cudaMemcpyDeviceToDevice);

    const dim3 block(16,16);
    const dim3 grid((CAM_IMG_WIDTH*3 + block.x - 1)/block.x, (CAM_IMG_HEIGHT + block.y -1)/block.y);

    PitchedRGBA2GpuMat <<< grid, block >>> (mPitchedImgCudaRGBA, mGpuMat_data, CAM_IMG_WIDTH, CAM_IMG_HEIGHT, CUDA_PITCH);

    const dim3 gridROI((mROIw*3 + block.x - 1)/block.x, (mROIh + block.y - 1)/block.y);

    if(mResizeEnable)
    {
        cv::cuda::resize(mGpuMat, mGpuMatResized, cv::Size(mResizeWidth, mResizeHeight));

        GpuMat2Img <<< gridROI, block >>> (mGpuMatResized_data, mTrtImg, mGpuMatResizedAndCropped_data,
                                           mResizeWidth, mResizeHeight,
                                           mROIx, mROIy, mROIw, mROIh);
    }
    else
    {
        GpuMat2Img <<< gridROI, block >>> (mGpuMat_data, mTrtImg, mGpuMatResizedAndCropped_data,
                                           CAM_IMG_WIDTH, CAM_IMG_HEIGHT,
                                           mROIx, mROIy, mROIw, mROIh);
    }

    dwSensorCamera_returnFrame(&mFrameHandle);

//    auto end = std::chrono::high_resolution_clock::now();

//    cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << endl;

    return true;
}

void px2Cam::RenderCamImg()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    dwTime_t timeout = 132000;

    // stream that image to the GL domain
    CHECK_DW_ERROR(dwImageStreamer_producerSend(mFrameCUDAHandle, mStreamerCUDA2GL));

    CHECK_DW_ERROR(dwImageStreamer_consumerReceive(&mFrameGLHandle, timeout, mStreamerCUDA2GL));

    CHECK_DW_ERROR(dwImage_getGL(&mImgGl, mFrameGLHandle));

    // render received texture
    dwVector2f range{};
    range.x = mImgGl->prop.width;
    range.y = mImgGl->prop.height;
    CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, mRenderEngine));
    CHECK_DW_ERROR(dwRenderEngine_renderImage2D(mImgGl, {0.0f, 0.0f, range.x, range.y}, mRenderEngine));

    // returned the consumed image
    CHECK_DW_ERROR(dwImageStreamer_consumerReturn(&mFrameGLHandle, mStreamerCUDA2GL));

    // notify the producer that the work is done
    CHECK_DW_ERROR(dwImageStreamer_producerReturn(nullptr, timeout, mStreamerCUDA2GL));
}

void px2Cam::DrawBoundingBoxes(vector<cv::Rect>  bbRectList, vector<float32_t*> bbColorList, float32_t lineWidth)
{
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(lineWidth, mRenderEngine));
    for(uint bbInd = 0; bbInd < bbRectList.size(); bbInd++)
    {
        float32_t* bBoxColor = bbColorList[bbInd];
        dwRenderEngineColorRGBA bBoxColorDw;
        bBoxColorDw.x = bBoxColor[0];
        bBoxColorDw.y = bBoxColor[1];
        bBoxColorDw.z = bBoxColor[2];
        bBoxColorDw.w = bBoxColor[3];
        CHECK_DW_ERROR(dwRenderEngine_setColor(bBoxColorDw, mRenderEngine));

        cv::Rect bBoxRect = bbRectList[bbInd];
        dwRectf bBoxRectDw;
        bBoxRectDw.x = bBoxRect.x;
        bBoxRectDw.y = bBoxRect.y;
        bBoxRectDw.width = bBoxRect.width;
        bBoxRectDw.height = bBoxRect.height;

        dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &bBoxRectDw, sizeof(dwRectf), 0, 1, mRenderEngine);
    }
}

void px2Cam::DrawBoundingBoxesWithLabels(vector<cv::Rect>  bbRectList, vector<float32_t*> bbColorList, vector<const char*> bbLabelList, float32_t lineWidth)
{
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(lineWidth, mRenderEngine));
    for(uint bbInd = 0; bbInd < bbRectList.size(); bbInd++)
    {
        float32_t* bBoxColor = bbColorList[bbInd];
        dwRenderEngineColorRGBA bBoxColorDw;
        bBoxColorDw.x = bBoxColor[0];
        bBoxColorDw.y = bBoxColor[1];
        bBoxColorDw.z = bBoxColor[2];
        bBoxColorDw.w = bBoxColor[3];
        CHECK_DW_ERROR(dwRenderEngine_setColor(bBoxColorDw, mRenderEngine));

        cv::Rect bBoxRect = bbRectList[bbInd];
        dwRectf bBoxRectDw;
        bBoxRectDw.x = bBoxRect.x;
        bBoxRectDw.y = bBoxRect.y;
        bBoxRectDw.width = bBoxRect.width;
        bBoxRectDw.height = bBoxRect.height;

        const char* bbLabel = bbLabelList[bbInd];

        dwRenderEngine_renderWithLabel(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &bBoxRectDw, sizeof(dwRectf), 0, bbLabel, 1, mRenderEngine);
    }
}

void px2Cam::DrawBoundingBoxesWithLabelsPerClass(vector<vector<dwRectf> >  bbRectList, vector<const float32_t*> bbColorList, vector<vector<const char*> > bbLabelList, float32_t lineWidth)
{
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(lineWidth, mRenderEngine));
    for(uint classIdx = 0; classIdx < bbRectList.size(); classIdx++)
    {
        const float32_t* bBoxColor = bbColorList[classIdx];
        dwRenderEngineColorRGBA bBoxColorDw;
        bBoxColorDw.x = bBoxColor[0];
        bBoxColorDw.y = bBoxColor[1];
        bBoxColorDw.z = bBoxColor[2];
        bBoxColorDw.w = bBoxColor[3];
        CHECK_DW_ERROR(dwRenderEngine_setColor(bBoxColorDw, mRenderEngine));

        if (bbRectList[classIdx].size() == 0)
            continue;

        CHECK_DW_ERROR(dwRenderEngine_renderWithLabels(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &bbRectList[classIdx][0], sizeof(dwRectf), 0, &bbLabelList[classIdx][0], bbRectList[classIdx].size(), mRenderEngine));
    }
}

void px2Cam::DrawPoints(vector<cv::Point> ptList, float32_t ptSize, float32_t* ptColor)
{
    CHECK_DW_ERROR(dwRenderEngine_setPointSize(ptSize, mRenderEngine));
    vector<dwVector2f> ptDwList;
    for(uint ptInd = 0; ptInd < ptList.size(); ptInd++)
    {
        cv::Point pt = ptList[ptInd];
        dwVector2f ptDw;
        ptDw.x = pt.x;
        ptDw.y = pt.y;
        ptDwList.push_back(ptDw);
    }

    dwRenderEngineColorRGBA ptColorDw;
    ptColorDw.x = ptColor[0];
    ptColorDw.y = ptColor[1];
    ptColorDw.z = ptColor[2];
    ptColorDw.w = ptColor[3];

    CHECK_DW_ERROR(dwRenderEngine_setColor(ptColorDw, mRenderEngine));

    dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D, &ptDwList[0], sizeof(dwVector2f), 0, ptDwList.size(), mRenderEngine);
}

void px2Cam::DrawPolyLine(vector<cv::Point> ptList, float32_t lineWidth, float32_t* lineColor)
{
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(lineWidth, mRenderEngine));
    vector<dwVector2f> ptDwList;
    for(uint ptInd = 0; ptInd < ptList.size(); ptInd++)
    {
        cv::Point pt = ptList[ptInd];
        dwVector2f ptDw;
        ptDw.x = pt.x;
        ptDw.y = pt.y;
        ptDwList.push_back(ptDw);
    }

    dwRenderEngineColorRGBA lineColorDw;
    lineColorDw.x = lineColor[0];
    lineColorDw.y = lineColor[1];
    lineColorDw.z = lineColor[2];
    lineColorDw.w = lineColor[3];

    CHECK_DW_ERROR(dwRenderEngine_setColor(lineColorDw, mRenderEngine));

    dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_2D, &ptDwList[0], sizeof(dwVector2f), 0, ptDwList.size(), mRenderEngine);
}

void px2Cam::DrawPolyLineDw(vector<dwVector2f> ptList, float32_t lineWidth, dwVector4f lineColor)
{
    CHECK_DW_ERROR(dwRenderEngine_setLineWidth(lineWidth, mRenderEngine));

    CHECK_DW_ERROR(dwRenderEngine_setColor(lineColor, mRenderEngine));

    dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_2D, &ptList[0], sizeof(dwVector2f), 0, ptList.size(), mRenderEngine);
}

void px2Cam::DrawText(const char* text, cv::Point textPos, float32_t* textColor)
{
    dwVector2f textPosDw;
    textPosDw.x = textPos.x;
    textPosDw.y = textPos.y;


    dwRenderEngineColorRGBA textColorDw;
    textColorDw.x = textColor[0];
    textColorDw.y = textColor[1];
    textColorDw.z = textColor[2];
    textColorDw.w = textColor[3];

    CHECK_DW_ERROR(dwRenderEngine_setColor(textColorDw, mRenderEngine));

    dwRenderEngine_renderText2D(text, textPosDw, mRenderEngine);
}

void px2Cam::UpdateRendering()
{
    mWindow->swapBuffers();
}

dwContextHandle_t px2Cam::GetDwContext()
{
    return mContext;
}

trtImgData px2Cam::GetTrtImgData()
{
    mCurTrtImgData.timestamp_us = mCamTimestamp;
    mCurTrtImgData.trtImg = mTrtImg;
    return mCurTrtImgData;
}

matImgData px2Cam::GetCroppedMatImgData()
{
    mGpuMatResizedAndCropped.download(mMatResizedAndCropped);
    mCurCroppedMatImgData.timestamp_us = mCamTimestamp;
    mCurCroppedMatImgData.matImg = mMatResizedAndCropped;
    return mCurCroppedMatImgData;
}

matImgData px2Cam::GetOriMatImgData()
{
    mGpuMat.download(mMatOri);
    mCurOriMatImgData.timestamp_us = mCamTimestamp;
    mCurOriMatImgData.matImg = mMatOri;
    return mCurOriMatImgData;
}

dwImageCUDA* px2Cam::GetDwImageCuda()
{
    return mCamImgCuda;
}
