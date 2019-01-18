/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*
 * @file
 * @brief <b>NVIDIA CCP Interface: Camera Control protocol Support</b>
 * @b Description: This file declares an interface for CCP support.
 */

#ifndef CCP_H
#define CCP_H
#include <stdint.h>


/*
 * These are returned to the user library
 * on a ccp function call depicting the status
*/
typedef enum
{
    CCP_STATUS_OK = 0,
    CCP_STATUS_DENIED,
    CCP_STATUS_NOT_REGISTERED,
    CCP_STATUS_ALREADY_ON,
    CCP_STATUS_ALREADY_OFF,
    CCP_STATUS_SLAVE_RUNNING,
    CCP_REQ_FILE_OP_FAIL = 1000,
    CCP_REQ_TIMEOUT,
    CCP_REQ_INVALID,
    CCP_REQ_DENIED,
    CCP_REQ_FAILED,
    CCP_REQ_CONNECT_ERR,
} ccp_return_t;


/*
 * These represent the camera group id
 */
typedef enum {
    CCP_GROUP_A = 0x00,
    CCP_GROUP_B = 0x01,
    CCP_GROUP_C = 0x02,
    CCP_GROUP_D = 0x03
} ccp_cam_group_id;

/*
 * These represent the camera id
 */
typedef enum{
    /* Camera group A */
    CCP_CAM_A0 = 0x01,
    CCP_CAM_A1 = 0x02,
    CCP_CAM_A2 = 0x04,
    CCP_CAM_A3 = 0x08,
    /* Camera group B */
    CCP_CAM_B0 = 0x10,
    CCP_CAM_B1 = 0x20,
    CCP_CAM_B2 = 0x40,
    CCP_CAM_B3 = 0x80,
    /* Camera group C */
    CCP_CAM_C0 = 0x0100,
    CCP_CAM_C1 = 0x0200,
    CCP_CAM_C2 = 0x0400,
    CCP_CAM_C3 = 0x0800,
    /* Camera group D */
    CCP_CAM_D0 = 0x1000,
    CCP_CAM_D1 = 0x2000,
    CCP_CAM_D2 = 0x4000,
    CCP_CAM_D3 = 0x8000
} ccp_cam_id;

/*
 * These represent the tegra id
 */
typedef enum {
    CCP_TEGRA_A = 0x41,
    CCP_TEGRA_B = 0x42
} ccp_tegra_id;

/*
 * These represent the mode in which
 * request or release ownership is made
 */
typedef enum {
    CCP_CAM_MASTER = 0x01,
    CCP_CAM_SLAVE
} ccp_cam_mode;

/**
 * Request ownership for selected camera group.
 * @param cam_group Aggregator id (not a bitmask)
 * @param cam_master requested mode of operation from ccp_cam_mode
 * @retval ccp_return_t
 */
ccp_return_t ccp_request_ownership(ccp_cam_group_id cam_group,ccp_cam_mode cam_master);

/**
 * Release ownership for selected camera group.
 * @param cam_group Aggregator id (not a bitmask)
 * @param cam_master requested mode of operation from ccp_cam_mode
 * @retval ccp_return_t
 * */
ccp_return_t ccp_release_ownership(ccp_cam_group_id cam_group,ccp_cam_mode cam_master);

/**
 * Turn on power for all cameras in the aggregator.
 * @param cam_group_id Aggregator id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_cam_pwr_on(ccp_cam_group_id cam_group);

/**
 * Turn off power for all cameras in the aggregator.
 * @param cam_group_id Aggregator id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_cam_pwr_off(ccp_cam_group_id cam_group);

/*
 * Turn on power for camera unit.
 * @param cam_id camera id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_cam_unit_pwr_on(ccp_cam_id cam_id);

/**
 * Turn off power for camera unit.
 * @param cam_id camera id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_cam_unit_pwr_off(ccp_cam_id cam_id);

/**
 * Turn on power for camera aggregator.
 * @param cam_group Aggregator id (Not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_aggreg_pwr_on(ccp_cam_group_id cam_group);

/**
 * Turn off power for camera aggregator.
 * @param cam_group Aggregator id (Not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_aggreg_pwr_off(ccp_cam_group_id cam_group);

/**
 * Set frsync owner
 * @param tegra_id Tegra ID
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_frsync_owner(ccp_tegra_id tegra_id);

/**
 * Set frsync enable.
 * @param cam_group Aggregator id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_frsync_enable(ccp_cam_group_id cam_group);

/**
 * Set frsync disable.
 * @param cam_group Aggregator id (not a bitmask)
 * @retval ccp_return_t
 */
ccp_return_t ccp_set_frsync_disable(ccp_cam_group_id cam_group);

/**
 * Get camera unit power status.
 * @param cam_id This will be filled with the camera power status
 * @retval ccp_return_t
 */
ccp_return_t ccp_get_cam_unit_pwr_status(uint16_t *cam_id);

/**
 * Get aggregator power status.
 * @param cam_group This will be filled with the  aggregator power status
 * @retval ccp_return_t
 */
ccp_return_t ccp_get_aggreg_pwr_status(uint8_t *cam_group);

/**
 * Get frsync enable status.
 * @param cam_group This will be filled with frsync enable status
 * @retval ccp_return_t
 */
ccp_return_t ccp_get_frsync_enable_status(uint8_t *cam_group);
#endif
