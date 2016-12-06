//VisibilityChecker_CUDA.cu
VisibilityChecker_CUDA::VisibilityChecker_CUDA(Vec3 observationPoint, std::vector<Vec3>& checkPoints,
	std::vector<Vec3>& points, std::vector<Face>& faces, std::vector<bool>& result) 
{
    int faceSize = faces.size();
    int checkPointSize = checkPoints.size();
    Face* h_faces = &faces[0];
    Vec3* h_points = &points[0], h_checkPoints = &checkPoints[0];
    Face* d_faces;
    Vec3* d_points, d_checkPoints;

    cudaMalloc((void**)&d_points, sizeof(Vec3)*points.size());
    cudaMalloc((void**)&d_checkPoints, sizeof(Vec3)*checkPointSize);
    cudaMalloc((void**)&d_faces, sizeof(Face)*faceSize);
    cudaCheckErrors("cudaMalloc fail");
    cudaMemcpy(d_points, h_points, sizeof(Vec3)*points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_checkPoints, h_checkPoints, sizeof(Vec3)*checkPointSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, h_faces, sizeof(Face)*faceSize, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy fail");    
    result.resize(checkPointSize);

    int reductionSize = ceil(faceSize*1.f/1024);
    bool* h_result_raw = (bool*) malloc(sizeof(bool)*reductionSize*checkPointSize);
    bool* d_result_raw;
    cudaMalloc((void**)&d_result_raw, sizeof(bool)*(reductionSize+1)*checkPointSize);
    cudaCheckErrors("cudaMemalloc result fail");
    int blockWidth = ceil(sqrt(checkPointSize));
    dim3 blockSize(reductionSize, blockWidth, blockWidth);
    dim3 threadSize(1024, 1, 1);
    checkVisibility<<<blockSize, threadSize, 1024*sizeof(bool)>>>(observationPoint, reductionSize, 
		d_faces, faceSize, d_points, d_checkPoints, checkPointSize, d_result_raw);
    cudaCheckErrors("kernel fail");
    cudaDeviceSynchronize();
    cudaCheckErrors("sync fail");

    cudaMemcpy(h_result_raw, d_result_raw, 
		sizeof(bool)*reductionSize*checkPointSize, cudaMemcpyDeviceToHost);
    cudaFree(d_result_raw);
    cudaFree(d_points);
    cudaFree(d_checkPoints);
    cudaFree(d_faces);
    
    for (int i = 0; i < result.size(); i++) 
        result[i] = true;
    for (int i = 0; i < result.size(); i++) 
        for (int j = 0; j < reductionSize; j++) 
            result[i] = result[i] && h_result_raw[i*reductionSize+j];            
}

__global__ void checkVisibility(Vec3 observationPoint, int reductionSize, Face* faces, int faceSize, 
	Vec3* points, Vec3* checkPoints, int checkPointSize, bool* result) 
{
    extern __shared__ bool visibility_local[];

    int thread_idx = threadIdx.x;
    int face_idx = blockIdx.x*blockDim.x+thread_idx;
    int check_idx = blockIdx.y*gridDim.y+blockIdx.z;
    bool visible = false; 
    if (face_idx < faceSize && check_idx < checkPointSize) {
        Vec3 toPoint = checkPoints[check_idx];
        Vec3 V0 = points[faces[face_idx].x], V1 = points[faces[face_idx].y], V2 = points[faces[face_idx].z];
        Vec3 d0 = V0-toPoint, d1 = V1-toPoint, d2 = V2-toPoint;
        Vec3 dir = (toPoint-observationPoint);
        dir.normalize();
        if (d0.norm() < 0.1 || d1.norm() < 0.1 || d2.norm() < 0.1)
            visible = true;
        
        Vec3 edge1 = V1 - V0, edge2 = V2 - V0;
        Vec3 pvec = dir.cross(edge2);
        float det = edge1.dot(pvec);
        if (fabs(det) < EPSILON)    visible = true;
        float inv_det = 1.0/det;
        Vec3 tvec = observationPoint - V0;
        float u = tvec.dot(pvec)*inv_det;
        if (u < 0.0 || u > 1.0)
            visible = true;
        Vec3 qvec = tvec.cross(edge1);
        float v = dir.dot(qvec)*inv_det;
        if (v < 0.0 || u + v > 1.0)
            visible = true;
        float t = edge2.dot(qvec)*inv_det;
        if (t > (toPoint -observationPoint).norm() || t < 0.0f)
            visible = true;
    }
    if (face_idx >= faceSize || check_idx >= checkPointSize)
        visible = true;
    visibility_local[thread_idx] = visible;
    __syncthreads();

    for (int i = blockDim.x/2; i > 0; i>>=1) {
        if (thread_idx < i) 
            visibility_local[thread_idx] = visibility_local[thread_idx] && visibility_local[thread_idx+i];     
        __syncthreads();
    }
    __syncthreads();
    if (check_idx < checkPointSize && face_idx < faceSize) {
        if (thread_idx == 0) 
            result[check_idx*gridDim.x+blockIdx.x] = visibility_local[0];        
    }
}
