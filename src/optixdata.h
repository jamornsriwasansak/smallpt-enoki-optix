#include <optix.h>

struct OptixRay {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

struct OptixRayResult {
    int id;
    float t;
    float u;
    float v;
};

struct Params
{
    const float * m_ray_origin_x;
    const float * m_ray_origin_y;
    const float * m_ray_origin_z;
    const float * m_ray_dir_x;
    const float * m_ray_dir_y;
    const float * m_ray_dir_z;
    const float * m_ray_tmin;
    const float * m_ray_tmax;
    int * m_result_tri_id;
    float * m_result_barycentric_u;
    float * m_result_barycentric_v;
    float * m_result_t;
    OptixTraversableHandle m_optix_handle;
    bool m_do_closest;
};

struct RayGenData {};
struct HitGroupData {};
struct MissData {};
