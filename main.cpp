#include <iostream>

using namespace std;

#include "px2camlib.h"
#include "px2od.h"
#include "px2ld.h"

int main()
{
    px2Cam px2CamObj;

    /**********************************************/
    /**
     * Not For NVIDIA Driveworks DNN. (Driveworks DNN uses all area of the camera image)
     * It is for custom tensorRT network input image, or other purpose..
     */
    imgCropParameters imgCropParams;
    imgCropParams.resizeRatio = 0.54;
    imgCropParams.roiX = 6; // Top left x of R.O.I.
    imgCropParams.roiY = 140; // Top left y of R.O.I.
    imgCropParams.roiW = 1024; // Width of R.O.I.
    imgCropParams.roiH = 512; // Height of R.O.I.

    /**********************************************/


    displayParameters dispParams;
    dispParams.onDisplay = true;
    dispParams.windowTitle = "Camera Viewer";
    dispParams.windowWidth = CAM_IMG_WIDTH/2;
    dispParams.windowHeight = CAM_IMG_HEIGHT/2;

    // Camera library initialize
    if(!px2CamObj.Init(imgCropParams,
                       dispParams,
                       MASTER_TEGRA))
        return -1;

    // Object Detector(DriveNet) initialize
    px2OD px2ODObj(&px2CamObj);
    px2ODObj.Init();

    // Lane Detector initialize
    px2LD px2LDObj(&px2CamObj);
//    px2LDObj.Init(0.3f);
    px2LDObj.Init(0.3f,
                  "/home/nvidia/swjung/git/px2Recog/data/invRectMap.xml",
                  "/home/nvidia/swjung/git/px2Recog/data/ipmMat.xml");

    while(1)
    {
        auto begin = std::chrono::high_resolution_clock::now();

        px2CamObj.UpdateCamImg();

        // Get dwImageCUDA (all area of the original camera image)
        dwImageCUDA* dnnInputImg = px2CamObj.GetDwImageCuda();

        /****************************************************
         * Object Detector
         */

        vector<vector<dwRectf> > outputODRectPerClass;
        vector<const float32_t*> outputODRectColorPerClass;
        vector<vector<const char*> > outputODLabelPerClass;
        vector<vector<float32_t> > outputODConfidencePerClass;
        vector<vector<int> > outputODIDPerClass;

        px2ODObj.DetectObjects(dnnInputImg,
                               outputODRectPerClass,
                               outputODRectColorPerClass,
                               outputODLabelPerClass,
                               outputODConfidencePerClass,
                               outputODIDPerClass);

        /****************************************************/


        /****************************************************
         * Lane Detector
         */

        vector<vector<dwVector2f> > outputLDPtsPerLane;
        vector<dwVector4f> outputLDColorPerLane;
        vector<string> outputLDPositionNamePerLane;
        vector<string> outputLDTypeNamePerLane;

        px2LDObj.DetectLanesByDW(dnnInputImg,
                                 outputLDPtsPerLane,
                                 outputLDColorPerLane,
                                 outputLDPositionNamePerLane,
                                 outputLDTypeNamePerLane);

        vector< vector<float> > outputLDEqsPerLane;

        cv::Mat topViewImg = cv::Mat::zeros(500,500, CV_8UC3);

        for(uint32_t laneIdx = 0U;  laneIdx < outputLDPtsPerLane.size(); laneIdx++)
        {
            vector<dwVector2f> outputLDPtsRectified = px2LDObj.DistortList2RectifiedList(outputLDPtsPerLane[laneIdx]);
            vector<dwVector2f> outputLDPtsTopview = px2LDObj.RectifiedList2TopviewList(outputLDPtsRectified);

            if (outputLDPtsTopview.size() < 4)
                continue;

            vector<float> outputLDEqs = px2LDObj.TopviewList2Eq(outputLDPtsTopview);
            outputLDEqsPerLane.push_back(outputLDEqs);


            for(uint y = 0; y < 500 ; y++)
            {
                float x = outputLDEqs[0] + (float)y*outputLDEqs[1] + (float)y*y*outputLDEqs[2] + (float)y*y*y*outputLDEqs[3];
                int xInt = (int) x;

                if((xInt >= 0) && (xInt < topViewImg.cols))
                {
                    topViewImg.at<cv::Vec3b>(y, xInt)[0] = cv::saturate_cast<uchar>(outputLDColorPerLane[laneIdx].z*255);
                    topViewImg.at<cv::Vec3b>(y, xInt)[1] = cv::saturate_cast<uchar>(outputLDColorPerLane[laneIdx].y*255);
                    topViewImg.at<cv::Vec3b>(y, xInt)[2] = cv::saturate_cast<uchar>(outputLDColorPerLane[laneIdx].x*255);
                }
            }
        }

        cv::imshow("topView", topViewImg);
        cv::waitKey(1);

        /****************************************************/


        px2CamObj.RenderCamImg();

        // Draw Object Detection Results
        px2CamObj.DrawBoundingBoxesWithLabelsPerClass(outputODRectPerClass, outputODRectColorPerClass, outputODLabelPerClass, 1.0f);

        // Draw Lane Detection Results
        for(uint32_t laneIdx = 0U; laneIdx < outputLDPtsPerLane.size(); ++laneIdx)
        {
            px2CamObj.DrawPolyLineDw(outputLDPtsPerLane[laneIdx], 6.0f, outputLDColorPerLane[laneIdx]);
        }

        px2CamObj.UpdateRendering();

        auto end = std::chrono::high_resolution_clock::now();

        cout << "Camera & Recognition Total process time :  "
             << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "ms" << endl;
    }
}
