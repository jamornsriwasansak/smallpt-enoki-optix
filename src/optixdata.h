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
    bool closest;
    OptixRay * rays;
    OptixRayResult * results;
    OptixTraversableHandle handle;
    int * m_tri_id;
    float * m_ray_origin_x;
    float * m_ray_origin_y;
    float * m_ray_origin_z;
    float * m_ray_dir_x;
    float * m_ray_dir_y;
    float * m_ray_dir_z;
    float * m_ray_tmin;
    float * m_ray_tmax;
};

struct RayGenData {};
struct HitGroupData {};
struct MissData {};
