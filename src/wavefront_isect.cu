/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optix_device.h>
#include <vector_types.h>
#include <stdint.h>

#include "optixdata.h"

extern "C"
{
    __constant__ Params params;
}


//------------------------------------------------------------------------------
//
// Hit program copies hit attribute into hit PRD 
//
//------------------------------------------------------------------------------

extern "C" __global__
void __closesthit__ch()
{
    const float2 uv = optixGetTriangleBarycentrics();
    const float t = optixGetRayTmax();
    const unsigned int id = optixGetPrimitiveIndex();
    optixSetPayload_0(id);
    optixSetPayload_1(float_as_int(t));
    optixSetPayload_2(float_as_int(uv.x));
    optixSetPayload_3(float_as_int(uv.y));
}


//------------------------------------------------------------------------------
//
// Miss program
//
//------------------------------------------------------------------------------

extern "C" __global__
void __miss__ms()
{
    optixSetPayload_0(unsigned int(-1));
}


//------------------------------------------------------------------------------
//
// OptixRay generation
//
//------------------------------------------------------------------------------

extern "C" __global__
void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint32_t linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;
    bool ray_mask = (params.m_ray_mask_size == 1) ? params.m_ray_mask[0] : params.m_ray_mask[linear_idx];
    if (ray_mask)
    {
		float3 origin;
		origin.x = params.m_ray_origin_x[linear_idx];
		origin.y = params.m_ray_origin_y[linear_idx];
		origin.z = params.m_ray_origin_z[linear_idx];

		float3 direction;
		direction.x = params.m_ray_dir_x[linear_idx];
		direction.y = params.m_ray_dir_y[linear_idx];
		direction.z = params.m_ray_dir_z[linear_idx];

		float tmin = params.m_ray_tmin[linear_idx];
		float tmax = params.m_ray_tmax[linear_idx];

        uint32_t id, t, u, v;
        optixTrace(params.m_optix_handle, origin, direction, tmin, tmax, 0.0f, OptixVisibilityMask(1),
                   params.m_do_closest ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 0, 0, id, t, u, v);

        params.m_result_t[linear_idx] = int_as_float(t);
        params.m_result_tri_id[linear_idx] = id;
        params.m_result_barycentric_u[linear_idx] = int_as_float(u);
        params.m_result_barycentric_v[linear_idx] = int_as_float(v);
    }
}