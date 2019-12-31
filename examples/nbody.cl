#define SOFT 1e-9f

__kernel void bodyForces(const int nBody, const float dt,
            __global float * px, __global float * py, __global float * pz,
            __global float * vx, __global float * vy, __global float * vz){

  int i = get_global_id(0);
  if (i < nBody) {

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j=0; j<nBody; j++){

      float dx = px[j] - px[i];
      float dy = py[j] - py[i];
      float dz = pz[j] - pz[i];

      float distSqr = dx*dx + dy*dy + dz*dz + SOFT;
      float invDist = rsqrt(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;

    }

    vx[i] += dt*Fx;
    vy[i] += dt*Fy;
    vz[i] += dt*Fz;

  }

}

__kernel void bodyForces2(const int nBody, const float dt,
            __global float * px, __global float * py, __global float * pz,
            __local float * pxl, __local float * pyl, __local float * pzl,
            __global float * vx, __global float * vy, __global float * vz){

  int i = get_global_id(0);
  int ii = get_local_id(0);
  int s = get_local_size(0);

  if (i >= nBody) {
    return;
  }

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int jj=0; jj<nBody; jj=jj+s){

    pxl[ii] = px[jj+ii];
    pyl[ii] = py[jj+ii];
    pzl[ii] = pz[jj+ii];

barrier(CLK_LOCAL_MEM_FENCE);

    for (int j=0; j<s; j++){

      float dx = pxl[j] - px[i];
      float dy = pyl[j] - py[i];
      float dz = pzl[j] - pz[i];

      float distSqr = dx*dx + dy*dy + dz*dz + SOFT;
      float invDist = rsqrt(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;

    }
    }

    vx[i] += dt*Fx;
    vy[i] += dt*Fy;
    vz[i] += dt*Fz;

}

__kernel void integrateBodies(const int nBody, const float dt,
            __global float * px, __global float * py, __global float * pz,
            __global float * vx, __global float * vy, __global float * vz){

  int i = get_global_id(0);
  if (i < nBody) {

    px[i] += dt*vx[i];
    py[i] += dt*vy[i];
    pz[i] += dt*vz[i];

  }

}
