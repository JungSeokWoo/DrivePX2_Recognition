/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */


/**
 * \file
 * \brief <b> NVIDIA Media VPI Interface </b>
 *
 * @b Description: This file contains the NvMedia VPI API.
 */

#ifndef _NvMediaVPI_H
#define _NvMediaVPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_common.h"
#include "nvmedia_image.h"
#include "nvmedia_array.h"
#include "nvmedia_image_pyramid.h"

/**
 * \defgroup NvMediaVPI Vision Programming Interface
 *
 * The NvMedia Vision Programming Interface (VPI) API contains NvMedia
 * functions for accessing the Computer Vision (CV) hardware accelerated
 * algorithms.
 *
 * @ingroup nvmedia_top
 * @{
 */

/** \brief Major Version number */
#define NVMEDIA_VPI_VERSION_MAJOR   1
/** \brief Minor Version number */
#define NVMEDIA_VPI_VERSION_MINOR   0

/** \brief Maximum tasks that can be queued - used by \ref NvMediaVPICreate. */
#define NVMEDIA_VPI_MAX_QUEUED_TASKS 64

/** \brief Number of independent HW engines present. */
#define NUM_VPI_ENGINES (2)

/**
 * \brief Holds an opaque object representing NvMediaVPI.
 */
typedef struct NvMediaVPI NvMediaVPI;

/**
 * \brief Returns the version information for the NvMediaVPI library.
 * \param[in] version A pointer to a \ref NvMediaVersion structure
 *                    filled by the NvMediaVPI library.
 * \retval NvMediaStatus The status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaVPIGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Creates an NvMediaVPI object.
 * \param[in] Id Engine ID (< \ref NUM_VPI_ENGINES)
 * \param[in] maxQueuedTasks
 *      This determines how many simultaneous tasks can be submitted to an
 *      instance at a time, before \ref NvMediaVPIFlush must be called.
 *      maxQueuedTasks will be clamped between 1 and
 *      \ref NVMEDIA_VPI_MAX_QUEUED_TASKS.
 * \return A pointer to \ref NvMediaVPI object if successful or NULL if unsuccessful.
 */
NvMediaVPI *
NvMediaVPICreate(
    const uint32_t Id,
    const uint32_t maxQueuedTasks
);

/**
 * \brief Destroys an NvMediaVPI object created by \ref NvMediaVPICreate.
 * \param[in] vpi Pointer to the object to be destroyed.
 * \return \ref NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIDestroy(
     NvMediaVPI *vpi
);

/**
 * \brief Flushes all queued operations.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \return \ref NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIFlush(
     NvMediaVPI *vpi
);

/**
 * \brief Registers an image for use with an NvMediaVPI function.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] image Pointer to image to be registered.
 * \return \ref NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIImageRegister(
    NvMediaVPI *vpi,
    NvMediaImage *image
);

/**
 * \brief Unregisters an image after use.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] image Pointer to the image to be unregistered.
 * \return \ref NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIImageUnregister(
    NvMediaVPI *vpi,
    NvMediaImage *image
);

/**
 * \brief Registers an array for use with an NvMediaVPI function.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] array Pointer to array to be registered.
 * \retval    NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIArrayRegister(
    NvMediaVPI *vpi,
    NvMediaArray *array
);

/**
 * \brief Unregisters an array after use.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] array Pointer to array to be unregistered.
 * \retval    NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIArrayUnregister(
    NvMediaVPI *vpi,
    NvMediaArray *array
);

/**
 * \brief Registers a pyramid for use with an NvMediaVPI function.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] pyramid Pointer to the pyramid to be registered.
 * \retval    NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIPyramidRegister(
    NvMediaVPI *vpi,
    NvMediaImagePyramid *pyramid
);

/**
 * \brief Unregisters a pyramid after use.
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] pyramid Pointer to the pyramid to be unregistered.
 * \retval    NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if input argument is invalid.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIPyramidUnregister(
    NvMediaVPI *vpi,
    NvMediaImagePyramid *pyramid
);

/**
 * \brief NvMediaVPIProcessStereo descriptor.
 */
typedef struct NvMediaVPIProcessStereoDescriptor NvMediaVPIProcessStereoDescriptor;

/**
 * \brief Enumerates the types of output for OF/ST processing.
 */
typedef enum {
    /*! Disparity relative to reference */
    NVMEDIA_VPI_STEREO_DISPARITY = 0,
    /*! Absolute Motion Vector output type. */
    NVMEDIA_VPI_MV,
    /*! Motion Vector hints for NVENC. */
    NVMEDIA_VPI_MV_HINTS,
} NvMediaVPIOFSTOutputType;

/**
 * \brief Creates \ref NvMediaVPIProcessStereoDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  width Width of the input images.
 * \param[in]  height Height of the input images.
 * \param[in]  outputType \ref NvMediaVPIOFSTOutputType for type of output.
 * \return     A pointer to the \ref NvMediaVPIProcessStereoDescriptor.
 */
NvMediaVPIProcessStereoDescriptor *
NvMediaVPICreateProcessStereoPairDescriptor(
    NvMediaVPI *vpi,
    const uint32_t width,
    const uint32_t height,
    const NvMediaVPIOFSTOutputType outputType
);

/**
 * \brief Queues an operation to process a stereo image pair.
 * \note \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in,out] descriptor Pointer returned by \ref NvMediaVPICreateProcessStereoPairDescriptor.
 * \param[in]     left Pointer to the left \ref NvMediaImage image.
 * \param[in]     right Pointer to the right \ref NvMediaImage image.
 * \param[out]    output Pointer to the output \ref NvMediaImage image.
 * \retval        NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIProcessStereoPairDesc(
    NvMediaVPI *vpi,
    NvMediaVPIProcessStereoDescriptor *descriptor,
    NvMediaImage *left,
    NvMediaImage *right,
    NvMediaImage *output
);

/**
 * \brief NvMediaVPIConvertMV descriptor.
 */
typedef struct NvMediaVPIConvertMVDescriptor NvMediaVPIConvertMVDescriptor;

/**
 * \brief Creates an \ref NvMediaVPIConvertMVDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object that NvMediaVPICreate() returns.
 * \param[in]  width Width of the input depth image.
 * \param[in]  height Height of the input depth image.
 * \param[in]  type \ref NvMediaSurfaceType of the input depth image.
 * \param[in]  strength Smoothing strength: (0 - 1).
 * \param[in]  scale Scale Factor.
 * \return     A pointer to the \ref NvMediaVPIConvertMVDescriptor.
 */
NvMediaVPIConvertMVDescriptor *
NvMediaVPICreateConvertMVDescriptor(
    NvMediaVPI *vpi,
    const uint32_t width,
    const uint32_t height,
    const NvMediaSurfaceType type,
    const float strength,
    const float scale
);

/**
 * \brief Converts motion vector image to either stereo disparity or refines it.
 * \note \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in,out] descriptor Pointer returned by \ref NvMediaVPICreateConvertMVDescriptor.
 * \param[in]     inputMVImage input Motion Vectors image which must be converted/refined.
 * \param[in]     inputColor input color \ref NvMediaImage image.
 * \param[out]    output Output \ref NvMediaImage image which refined.
 * \param[in]     outputType \ref NvMediaVPIOFSTOutputType.
 * \retval        NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIConvertMVDesc(
    NvMediaVPI *vpi,
    NvMediaVPIConvertMVDescriptor *descriptor,
    NvMediaImage *inputMVImage,
    NvMediaImage *inputColor,
    NvMediaImage *output,
    const NvMediaVPIOFSTOutputType outputType
);

/*
 * \brief NvMediaVPIGetKeyPointsHarrisDescriptor.
 */
typedef struct NvMediaVPIGetKeyPointsHarrisDescriptor NvMediaVPIGetKeyPointsHarrisDescriptor;

/**
 * \brief Creates a pointer to the \ref NvMediaVPIGetKeyPointsHarrisDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  type \ref NvMediaSurfaceType of the input image.
 * \return     A pointer to the \ref NvMediaVPIGetKeyPointsHarrisDescriptor.
 */
NvMediaVPIGetKeyPointsHarrisDescriptor *
NvMediaVPICreateGetKeyPointsHarrisDescriptor(
    NvMediaVPI *vpi,
    const NvMediaSurfaceType type
);

/**
 * \brief Enumerates the types of post-process non-maximum suppression.
 */
typedef enum {
    /** Specifies to use a 2D grid of 8x8 cells with no minimal distance. */
    NVMEDIA_VPI_2D_GRID_8x8_CELL_NO_MIN_DISTANCE = 0
} NvMediaVPINonMaxSuppressionType;

/**
 * \brief Holds the Harris key-point parameters.
 */
typedef struct {
    /*! Specifies the minimum threshold with which to eliminate Harris Corner scores (computed using
     * the normalized Sobel kernel). */
    const float_t strengthThresh;
    /*! Specifies the post-process non-maximum suppression type. */
    const NvMediaVPINonMaxSuppressionType nonMaxSuppressionType;
    /*! Specifies the radial Euclidean distance for non-maximum suppression */
    const float_t minDistance;
    /*! Specifies sensitivity threshold from the Harris-Stephens equation. */
    const float_t sensitivity;
    /*! Gradient window size. Must be 3, 5 or 7. */
    const uint32_t gradientSize;
    /*! Block window size used to compute the Harris Corner score. Must be 3, 5 or 7. */
    const uint32_t blockSize;
} NvMediaVPIGetKeyPointsHarrisParams;

/**
 * \brief Queues an operation to get Harris key points throughout an image.
 *        If the key-points array is not empty, then it tracks the key points as well.
 * \note
 *       \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in,out] descriptor Pointer returned by \ref NvMediaVPICreateGetKeyPointsHarrisDescriptor
 * \param[in]     input \ref NvMediaImage image.
 * \param[in]     params Pointer to \ref NvMediaVPIGetKeyPointsHarrisParams.
 * \param[in,out] keypoints \ref NvMediaArray containing key points.
 * \param[out]    scores \ref NvMediaArray containing scores.
 * \retval        NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIGetKeyPointsHarrisDesc(
    NvMediaVPI *vpi,
    NvMediaVPIGetKeyPointsHarrisDescriptor *descriptor,
    NvMediaImage *input,
    const NvMediaVPIGetKeyPointsHarrisParams *params,
    NvMediaArray *keypoints,
    NvMediaArray *scores
);

/*
 * \brief NvMediaVPIGetKeyPointsFastDescriptor.
 */
typedef struct NvMediaVPIGetKeyPointsFastDescriptor NvMediaVPIGetKeyPointsFastDescriptor;

/**
 * \brief Creates a pointer to the \ref NvMediaVPIGetKeyPointsFastDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  width Width of the input image.
 * \param[in]  height Height of the input image.
 * \param[in]  type \ref NvMediaSurfaceType of the input image.
 * \return     A pointer to the \ref NvMediaVPIGetKeyPointsFastDescriptor
 */
NvMediaVPIGetKeyPointsFastDescriptor *
NvMediaVPICreateGetKeyPointsFastDescriptor(
    NvMediaVPI *vpi,
    const uint32_t width,
    const uint32_t height,
    const NvMediaSurfaceType type
);

/**
 * \brief Queues an operation to get FAST key-points throughout an image.
 *        If the key points array is not empty, then it tracks the key points as well.
 * \note \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in,out] descriptor Pointer returned by \ref NvMediaVPICreateGetKeyPointsFastDescriptor.
 * \param[in]     input \ref NvMediaImage image.
 * \param[in]     strengthThresh Threshold on difference between intensity of the central pixel
 *                and pixels on Bresenham's circle of radius 3.
 * \param[in]     nonmaxSupression Boolean flag that indicates whether to apply
 *                non-maximum suppression.
 * \param[in,out] keypoints Pointer to an \ref NvMediaArray containing key-points.
 * \param[out]    scores Pointer to an \ref NvMediaArray containing scores.
 * \retval        NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIGetKeyPointsFastDesc(
    NvMediaVPI *vpi,
    NvMediaVPIGetKeyPointsFastDescriptor *descriptor,
    NvMediaImage *input,
    const uint32_t strengthThresh,
    const NvMediaBool nonmaxSupression,
    NvMediaArray *keypoints,
    NvMediaArray *scores
);

/** Defines the conditions that cause the Lucas-Kanade Feature Tracker
 *  to be terminated.
 */
typedef enum {
    /** Specifies termination by the number of iterations. */
    NVMEDIA_VPI_TERMINATION_CRITERION_ITERATIONS = 0,
    /** Specifies termination by matching the value against epsilon. */
    NVMEDIA_VPI_TERMINATION_CRITERION_EPSILON,
    /** Specifies termination by iterations or epsilon, depending
      * on which happens first. */
    NVMEDIA_VPI_TERMINATION_CRITERION_BOTH
} NvMediaVPITerminationCriterion;

/**
 * \brief NvMediaVPIGetSparseFlowPyrLKParams.
 */
typedef struct {
    /*! Specifies the termination criteria. */
    const NvMediaVPITerminationCriterion termination;
    /*! Specifies the error for terminating the algorithm. */
    float epsilon;
    /*! Number of Points. */
    uint32_t numPoints;
    /*! Specifies the number of iterations. */
    uint32_t numIterations;
    /*! Specifies the size of the window on which to perform the algorithm. It must be greater than
     *  or equal to 3 and less than or equal to 32. */
    uint32_t windowDimension;
    /*! Specifies the boolean flag that indicates whether to calculate tracking error. */
    int32_t calcError;
    /*! Number of levels in the pyramid. */
    uint32_t numLevels;
} NvMediaVPIGetSparseFlowPyrLKParams;

/**
 * \brief NvMediaVPISparseFlowPyr descriptor.
 */
typedef struct NvMediaVPIGetSparseFlowPyrLKDescriptor NvMediaVPIGetSparseFlowPyrLKDescriptor;

/**
 * \brief Creates a pointer to the \ref NvMediaVPIGetSparseFlowPyrLKDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  params Pointer to \ref NvMediaVPIGetSparseFlowPyrLKParams.
 * \return     A pointer to the \ref NvMediaVPIGetSparseFlowPyrLKDescriptor.
 */
NvMediaVPIGetSparseFlowPyrLKDescriptor *
NvMediaVPICreateGetSparseFlowPyrLKDescriptor(
    NvMediaVPI *vpi,
    const NvMediaVPIGetSparseFlowPyrLKParams *params
);

/**
 * \brief Queues an operation to track features in pyramid.
 * \note \ref NvMediaVPIFlush() must be called to actually flush the queue to  the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in,out] descriptor Pointer returned by \ref NvMediaVPICreateGetSparseFlowPyrLKDescriptor.
 * \param[in]     params Pointer to \ref NvMediaVPIGetSparseFlowPyrLKParams.
 * \param[in]     oldImages Input \ref NvMediaImagePyramid of first (old) images.
 * \param[in]     newImages Input \ref NvMediaImagePyramid of destination (new) images.
 * \param[in]     oldPoints Pointer to an array of key-points in the oldImages
 *                high resolution pyramid.
 * \param[in,out] newPoints Pointer to an array of key-points in the newImages
 *                high resolution pyramid. If populated initially, it is used for starting
 *                search in newImages.
 * \param[out]    newStatus pointer to array for tracking status for each input oldPoints.
 * \retval NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus NvMediaVPIGetSparseFlowPyrLKDesc(
    NvMediaVPI *vpi,
    NvMediaVPIGetSparseFlowPyrLKDescriptor *descriptor,
    const NvMediaVPIGetSparseFlowPyrLKParams *params,
    NvMediaImagePyramid *oldImages,
    NvMediaImagePyramid *newImages,
    NvMediaArray *oldPoints,
    NvMediaArray *newPoints,
    NvMediaArray *newStatus
);

/**
 * \brief Queues an operation to generate a gaussian pyramid from an input image.
 * \note
 *       \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  input \ref NvMediaImage image.
 * \param[out] output \ref NvMediaImagePyramid.
 * \retval     NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIGetImagePyramid(
    NvMediaVPI *vpi,
    NvMediaImage *input,
    NvMediaImagePyramid *output
);

/**
 * \brief Queues an operation to convolve an input image with client supplied convolution matrix.
 * \note
 *       \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  input \ref NvMediaImage image.
 * \param[in]  kernelData Pointer to the convolution kernel coefficients.
 * \param[in]  kernelWidth Convolution kernel width.
 * \param[in]  kernelHeight Convolution kernel height.
 * \param[out] output \ref NvMediaImage image.
 * \retval     NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIConvolveImage(
    NvMediaVPI *vpi,
    NvMediaImage *input,
    const float_t *kernelData,
    const uint32_t kernelWidth,
    const uint32_t kernelHeight,
    NvMediaImage *output
);

/**
 * \brief NvMediaVPIConvolveImageSeparableDescriptor.
 */
typedef struct NvMediaVPIConvolveImageSeparableDescriptor \
               NvMediaVPIConvolveImageSeparableDescriptor;

/**
 * \brief Creates a pointer to the \ref NvMediaVPIConvolveImageSeparableDescriptor.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  type \ref NvMediaSurfaceType of the input image.
 * \param[in]  kernelX Pointer to the x convolution kernel coefficients.
 * \param[in]  kernelXSize Size of the x convolution kernel.
 * \param[in]  kernelY Pointer to the y convolution kernel coefficients.
 * \param[in]  kernelYSize Size of the y convolution kernel.
 * \return     A pointer to the \ref NvMediaVPIConvolveImageSeparableDescriptor.
 */
NvMediaVPIConvolveImageSeparableDescriptor *
NvMediaVPICreateConvolveImageSeparableDescriptor(
    NvMediaVPI *vpi,
    NvMediaSurfaceType type,
    const float_t *kernelX,
    const uint32_t kernelXSize,
    const float_t *kernelY,
    const uint32_t kernelYSize
);

/**
 * \brief Queues an operation to convolve an image with a 2D kernel, using separable
 * convolution.
 * \note \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]  vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]  input \ref NvMediaImage image.
 * \param[in]  kernelX Pointer to the x convolution kernel coefficients.
 * \param[in]  kernelXSize Size of the x convolution kernel.
 * \param[in]  kernelY Pointer to the y convolution kernel coefficients.
 * \param[in]  kernelYSize Size of the y convolution kernel.
 * \param[in]  output Pointer to an \ref NvMediaImage image.
 * \retval     NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIConvolveImageSeparable(
    NvMediaVPI *vpi,
    NvMediaImage *input,
    const float_t *kernelX,
    const uint32_t kernelXSize,
    const float_t *kernelY,
    const uint32_t kernelYSize,
    NvMediaImage *output
);

/**
 * \brief Queues an operation to compute a generic Box filter window of the input image.
 * \note
 *       \ref NvMediaVPIFlush() must be called to actually flush the queue to the engine.
 *
 * \param[in]     vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in]     input \ref NvMediaImage image.
 * \param[in]     windowWidth Window width
 * \param[in]     windowHeight Window height
 * \param[in]     output \ref NvMediaImage image.
 * \retval        NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_NOT_SUPPORTED for unsupported operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING for full queue. Call \ref NvMediaVPIFlush .
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIFilterImageBox(
    NvMediaVPI *vpi,
    NvMediaImage *input,
    const uint32_t windowWidth,
    const uint32_t windowHeight,
    NvMediaImage *output
);

/**
 * Helper API to convert an array of keypoints to an array of 2D float32 coordinates.
 *
 * \param[in]     array Pointer to input \ref NvMediaArray of NVMEDIA_ARRAY_TYPE_KEYPOINT
 * \param[out]    pointer Pointer to location in memory to write 2D 32 bit float array
 * \param[in]     numPoints Number of points to convert.
 * \retval NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK for successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER for incorrect arguments.
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIKeypointArray2Point2Df(
    NvMediaArray *array,
    void *pointer,
    uint32_t numPoints
);

/**
 * \brief Destroy NvMediaVPI function descriptor.
 *
 * \param[in] vpi Pointer to the \ref NvMediaVPI object returned by \ref NvMediaVPICreate.
 * \param[in] descriptor Pointer to NvMediaVPI function descriptor.
 * \retval NvMediaStatus The status of the API.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK - successful queuing of operation.
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER - incorrect arguments.
 * \n \ref NVMEDIA_STATUS_NOT_INITIALIZED - uninitialized
 * \n \ref NVMEDIA_STATUS_ERROR for any other error.
 */
NvMediaStatus
NvMediaVPIDestroyDescriptor(
    NvMediaVPI *vpi,
    void *descriptor
);

/*
 * \defgroup history_NvMediaVPI History
 * Provides change history for the NVIDIA VPI API.
 *
 * \section history_NvMediaVPI Version History
 *
 * <b> Version 1.0 </b> March 2, 2018
 * - Initial API definitions.
 *
 */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_VPI_H */
