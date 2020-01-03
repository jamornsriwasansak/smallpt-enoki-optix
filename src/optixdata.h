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
};

struct RayGenData {};
struct HitGroupData {};
struct MissData {};
