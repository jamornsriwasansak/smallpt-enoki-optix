#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>
#include <curand_kernel.h>

// triangle intersection
rtBuffer<optix::float3> vertexBuffer;
rtBuffer<optix::int3> indexBuffer;
rtBuffer<optix::float2> texCoordBuffer;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void meshIntersect(const int primIndex)
{
	const optix::int3 vertexIndex = indexBuffer[primIndex];
	const optix::float3 p0 = vertexBuffer[vertexIndex.x];
	const optix::float3 p1 = vertexBuffer[vertexIndex.y];
	const optix::float3 p2 = vertexBuffer[vertexIndex.z];

	optix::float3 n;
	float t, beta, gamma;
	if (optix::intersect_triangle_branchless(ray, p0, p1, p2, n, t, beta, gamma))
	{
		// note: ray tmin and tmax are already editted inside "intersect_triangle_*"
		if (rtPotentialIntersection(t))
		{
			rtReportIntersection(0);
		}
	}
}

// main program
struct RayHitQuery
{
	optix::float3 mOrigin;
	int mGeomId;
	optix::float3 mDir;
	int mPrimId;
	float tMin;
	float tMax;
	float time;
};

struct PerRayData_HitQuery
{
	float tMin;
	float tMax;
	int mGeomId;
	int mPrimId;
	float time;
};

struct PerRayData_VisQuery
{
	bool isHit;
};

rtBuffer<RayHitQuery, 1> rayOriginBuffer;
rtDeclareVariable(rtObject, topObject, , );
rtDeclareVariable(optix::uint, launchIndex, rtLaunchIndex, );

// closest Hit Program
RT_PROGRAM void rtClosestHit()
{
}

// visibility Test Program
RT_PROGRAM void rtAnyHit()
{
	
}

// entry
RT_PROGRAM void intersect()
{
	RayHitQuery & rayHitQuery = rayOriginBuffer[launchIndex];
	optix::Ray ray(rayHitQuery.mOrigin, rayHitQuery.mDir, 0, rayHitQuery.tMin, rayHitQuery.tMax);
	PerRayData_HitQuery prd;
	rtTrace(topObject, ray, prd);
	rayHitQuery.tMin = prd.tMin;
	rayHitQuery.tMax = prd.tMax;
	rayHitQuery.time = prd.time;
	rayHitQuery.mGeomId = prd.mGeomId;
	rayHitQuery.mPrimId = prd.mPrimId;
}
