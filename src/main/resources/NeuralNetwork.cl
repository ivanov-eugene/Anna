#define REAL double

/*
 * Multiply two matrices: matrixC = matrixA * matrixB
 */
__kernel void matrixMultiplication(
    __global REAL *matrixA,
    __global REAL *matrixB,
    __global REAL *matrixC,
    int widthA, int widthB) 
{
  int i = get_global_id(0); 
  int j = get_global_id(1);
  REAL sum = 0.0f;
  for (int k = 0; k < widthA; k++) {
      sum += matrixA[j * widthA + k] * matrixB[i + k * widthB];
  }
  matrixC[j * widthB + i] = sum;
}

/*
 * Calculate the sum of all vector elements.
 */
__kernel void vectorTotalByReduce(
    __global REAL* input,
    __global REAL* output, 
    __local REAL* sdata, 
    int firstLocalZeroIndex)
{
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int bnum = get_num_groups(0);

    unsigned int globalInputIndex = gid * 2;
    if (bid + 1 == bnum) { // if very last workgroup
      unsigned int localInputIndex = tid * 2;
      if (localInputIndex >= firstLocalZeroIndex)
        sdata[tid] = 0;
      else if (localInputIndex + 1 == firstLocalZeroIndex)
        sdata[tid] = input[globalInputIndex];
      else
        sdata[tid] = input[globalInputIndex] + input[globalInputIndex + 1];
    }
    else {
      sdata[tid] = input[globalInputIndex] + input[globalInputIndex + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    unsigned int localSize = get_local_size(0);
    for (unsigned int s = localSize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global memory
    if (tid == 0) output[bid] = sdata[0];
}

/*
 * Calculate the sum of all vector elements' squares.
 */
__kernel void vectorSquaredTotalByReduce(
    __global REAL* input,
    __global REAL* output, 
    __local REAL* sdata, 
    int firstLocalZeroIndex)
{
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int bnum = get_num_groups(0);
    
    unsigned int globalInputIndex = gid * 2;
    if (bid + 1 == bnum) { // if very last workgroup
      unsigned int localInputIndex = tid * 2;
      if (localInputIndex >= firstLocalZeroIndex)
        sdata[tid] = 0;
      else if (localInputIndex + 1 == firstLocalZeroIndex)
        sdata[tid] = pown(input[globalInputIndex], 2);
      else
        sdata[tid] = pown(input[globalInputIndex], 2) + pown(input[globalInputIndex + 1], 2);
    }
    else {
      sdata[tid] = pown(input[globalInputIndex], 2) + pown(input[globalInputIndex + 1], 2);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    unsigned int localSize = get_local_size(0);
    for (unsigned int s = localSize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global memory
    if (tid == 0) output[bid] = sdata[0];
}
 
/*
 * Calculate one step of the forward propagation algorithm.
 */
__kernel void forwardPropagation(
    __global REAL *inMatrix,
    __global REAL *thetaMatrix,
    __global REAL *outMatrix,
    int firstThetaBiasIndex, int inWidth) 
{
  int i = get_global_id(0); 
  int j = get_global_id(1);
  int thetaWidth = get_global_size(0);
  REAL sum = 0.0f;
  for (int k = 0; k < inWidth; k++) {
      sum += inMatrix[j * inWidth + k] * thetaMatrix[k * thetaWidth + i];
  }
  sum += thetaMatrix[firstThetaBiasIndex + i]; // add bias
  outMatrix[j * thetaWidth + i] = 1.0f / (1.0f + exp(-sum)); // activation via sigmoid function
}

/*
 * Calculate gradient's Theta as part of the back propagation algorithm.
 */
__kernel void calculateGradientTheta(
    __global REAL *matrixA,
    __global REAL *matrixPriorA,
    __global int *vectorY, // null except very last A
    __global REAL *matrixTheta,
    __global REAL *matrixResult,
    int numOfSamples, REAL lambda, int useVectorY) 
{
  int i = get_global_id(0); 
  int j = get_global_id(1);
  int widthPriorA = get_global_size(0) - 1; // priorA's width is equal to Theta's height minus 1
  int widthA = get_global_size(1);

  REAL sum = 0.0f;
  for (int k = 0; k < numOfSamples; k++) { 
      REAL d = matrixA[k * widthA + j]; // transposed form
      if (useVectorY != 0 && j == vectorY[k])
        d -= 1.0f;
      if (i == widthPriorA) 
        sum += d; // * 1
      else 
        sum += d * matrixPriorA[k * widthPriorA + i];
  }
  sum = sum / numOfSamples;
  if (i != widthPriorA) 
    sum += matrixTheta[i * widthA + j] * (lambda / numOfSamples);
  matrixResult[i * widthA + j] = sum;
}

/*
 * Calculate gradient's Delta as part of the back propagation algorithm.
 */
__kernel void calculateGradientDelta(
    __global REAL *matrixPriorDelta, // or last A if vectorY is not null
    __global int *vectorY, // null except very last A
    __global REAL *matrixTheta,
    __global REAL *matrixA,
    __global REAL *matrixResultDelta,
    int priorDeltaWidth, int useVectorY) 
{
  int i = get_global_id(0); 
  int j = get_global_id(1);
  int resultDeltaWidth = get_global_size(1);

  REAL sum = 0.0f;
  for (int k = 0; k < priorDeltaWidth; k++) { 
      REAL d = matrixPriorDelta[i * priorDeltaWidth + k];
      if (useVectorY != 0 && k == vectorY[i])
        d -= 1.0f;
      sum += d * matrixTheta[j * priorDeltaWidth + k]; // transposed form
  } 
  int index = i * resultDeltaWidth + j;
  matrixResultDelta[index] = sum * matrixA[index] * (1 - matrixA[index]);
}

/*
 * Calculate Log part of the cost function.
 */
__kernel void costFunctionPartLog(
    __global int *vectorY,
    __global REAL *matrixA,
    __global REAL *matrixLogA) 
{
  int i = get_global_id(0); // column index
  int j = get_global_id(1); // row index
  int width = get_global_size(0);

  REAL f = matrixA[j * width + i];
  if (i != vectorY[j]) 
    f = 1.0f - f;
  matrixLogA[j * width + i] = -log(f);
}
