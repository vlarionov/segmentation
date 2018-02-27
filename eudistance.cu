///////////////////////////////////////////////////////////////////////////////////////////////
//===========================================================================================//
//    Code to calculate 2D segmentation of input image using CUDA                            //
//                                                                                           //
//    Author - Vladislav Larionov (Feb 2018)                                                 //
//                                                                                           //
//    uses chan-vese segmentation as presented in                                            //
//              http://www.math.ucla.edu/~lvese/PAPERS/IEEEIP2001.pdf                        //
//                                                                                           //
//    adapted from python version found at                                                   //
//              https://github.com/kevin-keraudren/chanvese                                  //
//                                                                                           //
//===========================================================================================//
///////////////////////////////////////////////////////////////////////////////////////////////

// include libraries
#include <iostream>
#include <math.h>
#include <fstream>
#include <cmath>
#include <float.h>

using namespace std; 

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
// GPU kernel function that calculates the exact euclidean distance
// from current pixel to the closest background pixel (value = 0)
//====================================================================
// inputs
//====================================================================
// N           - total number of input pixels
// image       - pointer to the start of the array containing image data
// distanceMap - used to store the output. input is overwritten on exit
//====================================================================
// changed variables
//====================================================================
// distanceMap - contains the smallest euclidean distance from the pixel
//                 with value 0. Pixels of value 0 are considered background pixels
//                 and this mapping gives the closest distance to a background
//                 pixel, where the adjecent pixel distance is equal to 1
//---------------------------------------------------------------------------------------------

__global__
void euclideanMap(long int N, float *image, float *distanceMap)
{

  // blockIdx.x contains the index of the current thread block
  // gridDim.x contains the number of blocks in the grid

  // threadIdx.x contains the index of the current thread within its block
  // blockDim.x contains the number of threads in the block


  long int indexX = blockDim.x*blockIdx.x + threadIdx.x;

  long int indexY = blockDim.x*gridDim.x*threadIdx.y + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;

  long int index = indexX + indexY;

  long int imgindexX;

  long int imgindexY;

  long int distance;

  long int compare;

  // check index to make sure it is in the image 
  if(index < N)
  {

    // set distance to maximum possible value
    distance = 1080*1080 + 1115*1115;   // 2D hardcoded case
    //distance = 1080*1080 + 1115*1115 + 1735*1735;    // 3D hardcoded case

    if( image[index] == 0.0 )
    {
      distanceMap[index] = 0.0;
    }

    else
    {
      indexX = index % 1080;
      indexY = index / 1080;

      for(long int i = 0; i<N; i++)
      {
        if(image[i] == 0.0)
        {
          imgindexX = i % 1080;
          imgindexY = i / 1080;

          compare = (imgindexX-indexX)*(imgindexX-indexX) + (imgindexY-indexY)*(imgindexY-indexY) ;
          if(compare < distance)
          {
            distance = compare;
          }

        }

      }

      distanceMap[index] = sqrtf( (float) distance );

    }

  }

}

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//function to negate a binary image
//    ** input image is reduced to binary; 0 ->0 , else->1 **
//    function calculates the logical opposite of each array element
//      - 0.0 gets mapped to 1.0
//      - all other entries get mapped to 0.0
//====================================================================
// inputs
//====================================================================
// N           - total number of input pixels
// image       - pointer to the start of the array containing image data
//====================================================================
// changed variables
//====================================================================
// image       - if input is == 0.0 output is 1.0
//                  else            output is 0.0
//---------------------------------------------------------------------------------------------

__global__
void negateImage(long int N, float *image)
{
  long int indexX = blockDim.x*blockIdx.x + threadIdx.x;

  long int indexY = blockDim.x*gridDim.x*threadIdx.y + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;

  long int index = indexX + indexY;

  // check index to make sure it is in the image 
  if(index < N)
  {

    if(image[index] == 0.0)
    {
      image[index] = 1.0;
    }

    else
    {
      image[index] = 0.0;
    }

  }

}

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//routine to create a Signed Distance Function from the boundary of image
// the result is stored in distance
// 
//    this routine calculates the signed distance function of an input image
//      according to the formula 
//          EuDist = EuDist( Neg(IMG) ) - EuDist(IMG) + IMG - 0.5
//            where IMG is a boolean representation of the input image IMG
//            and EuDist is the exact euclidean mapping of image IMG
//            and Neg maps 0.0 -> 1.0 and all other arguments -> 0.0
//====================================================================
// inputs
//====================================================================
// N           - total number of input pixels
// image       - pointer to the start of the array containing image data
// distance    - pointer to start of array containing euclidean distance mapping of image
// distanceNeg - pointer to start of array containing eulcidean distance mapping of negated image
//====================================================================
// changed variables
//====================================================================
// distance    - contains the signed distance function (SDF) from the mask boundary
//---------------------------------------------------------------------------------------------


__global__
void SDF(long int N, float *image, float *distance, float *distanceNeg)
{
  long int indexX = blockDim.x*blockIdx.x + threadIdx.x;

  long int indexY = blockDim.x*gridDim.x*threadIdx.y + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;

  long int index = indexX + indexY;

  // check index to make sure index is in the image 
  if(index < N)
  {
    distance[index] = distanceNeg[index] -distance[index] +image[index] - 0.5;
  }

}

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//routine to calculate curvature of active contour/surface for all points in the narrow band

__global__
void get_curvature(int imgDimX, int imgDimY, long int maxIndex, long int *narrowBandIdx , float *distance, float *curvature)
{
  long int indexX = blockDim.x*blockIdx.x + threadIdx.x;

  long int indexY = blockDim.x*gridDim.x*threadIdx.y + blockDim.x*gridDim.x*blockDim.y*blockIdx.y;

  long int index = indexX + indexY;

  float eps = FLT_EPSILON;

  // check index to make sure index is in the narrow band range 
  if(index < maxIndex)
  {

    long int bandIndexX = narrowBandIdx[index] % imgDimX;

    long int bandIndexY = narrowBandIdx[index] / imgDimX;


    //set neighbor indeces and check to make sure they are in bounds

    long int bandIndexXP = bandIndexX + 1;
    long int bandIndexXM = bandIndexX - 1;
    long int bandIndexYP = bandIndexY + 1;
    long int bandIndexYM = bandIndexY - 1;

    if(bandIndexXP >= imgDimX) { bandIndexXP = imgDimX -1 ;}
    if(bandIndexXM < 0 ) { bandIndexXM = 0 ;}

    if(bandIndexYP >= imgDimY) { bandIndexYP = imgDimY -1 ;}
    if(bandIndexYM < 0 ) { bandIndexYM = 0 ;}

    // get central derivatives of the SDF for curvature calculation
    //   first order derivatives
    float SDF_x = distance[bandIndexY*imgDimX + bandIndexXP] - distance[bandIndexY*imgDimX + bandIndexXM];
    float SDF_y = distance[bandIndexYP*imgDimX + bandIndexX] - distance[bandIndexYM*imgDimX + bandIndexX];

    //   second order derivatives
    float SDF_xx = distance[bandIndexY*imgDimX + bandIndexXP] - 2.0*distance[bandIndexY*imgDimX + bandIndexX] \
                       + distance[bandIndexY*imgDimX + bandIndexXM] ;
    float SDF_yy = distance[bandIndexYP*imgDimX + bandIndexX] - 2.0*distance[bandIndexY*imgDimX + bandIndexX] \
                       + distance[bandIndexYM*imgDimX + bandIndexX] ;

    // this one seems very suspect... it should have the opposite sign...
    float SDF_xy = 0.25 * ( distance[bandIndexYM*imgDimX+bandIndexXP] \
                          + distance[bandIndexYP*imgDimX+bandIndexXM] \
                          - distance[bandIndexYM*imgDimX+bandIndexXM] \
                          - distance[bandIndexYP*imgDimX+bandIndexXP] ) ;

    float SDF_x2 = SDF_x*SDF_x;
    float SDF_y2 = SDF_y*SDF_y;

    // compute curvature
    //   CURRENTLY ONLY WORKS FOR SINGLE PRECSION FLOATS ===============================================
    //       make sure to change to pow for double precision
    curvature[index] = ( (SDF_x2 * SDF_yy + SDF_y2 * SDF_xx -2.0*SDF_x *SDF_y*SDF_xy) / \
                          ( powf( (SDF_x2 + SDF_y2 +eps) , 2.0 ) )  ) * (powf( (SDF_x2 + SDF_y2) , 0.5) );

  }

}




//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

// main program
//   sets up image parameters and launches kernels to calculate 2D segmentation 

int main(void)
{

  // for 3D calculation one will probably need the index to be a long int
  // since the range for regular int ends at 32,767

  int iterations;
  float eps = FLT_EPSILON;

  // input image parameters
  //   need to find a way to automate this ------
  long int N = 1204200;
  int imgDimX = 1080;
  int imgDimY = 1115;

  int narrowGridN; // used to determine kernel size for narrow band calculation
  int numBlocksBand;


// cuda thread parameters

  // define the number of threads per block
  //  followed by the number of blocks
  //     "some places" say to use multiples of 32 
  //      for the number of threads

  int numThreadsX = 32;
  int numThreadsY = 32;
  int numBlocksX = (imgDimX + numThreadsX -1) / numThreadsX;
  int numBlocksY = (imgDimY + numThreadsY -1) / numThreadsY;

  dim3 blocks(numBlocksX,numBlocksY);
  dim3 threads(numThreadsX,numThreadsY);

  // pointers to image and mask arrays

  float *distance;
  float *d_distance;
  float *distanceNeg;
  float *d_distanceNeg;

  float *image;
  float *d_image;

  float *mask;
  float *d_mask;
  float *maskNeg;
  float *d_maskNeg;


  float transfer; // used for conversion of input data from int to float


  long int *narrowBandIndx;
  long int *d_narrowBandIndx;

  long int maxIndex;
  long int numInteriorPx;

  float insideSum;
  float outsideSum;

  float Force[N];

  float *curvature;

  float *d_curvature;

  // allocate CPU side arrays
  distance = (float*)malloc(N*sizeof(float));
  distanceNeg = (float*)malloc(N*sizeof(float));
  image = (float*)malloc(N*sizeof(float));
  mask = (float*)malloc(N*sizeof(float));
  maskNeg = (float*)malloc(N*sizeof(float));

  narrowBandIndx = (long int*)malloc(N*sizeof(long int));

  curvature = (float*)malloc(N*sizeof(float));

  // allocate GPU memory
  cudaMalloc(&d_distance, N*sizeof(float));
  cudaMalloc(&d_distanceNeg, N*sizeof(float));
  cudaMalloc(&d_image, N*sizeof(float));
  cudaMalloc(&d_mask, N*sizeof(float));
  cudaMalloc(&d_maskNeg, N*sizeof(float));

  cudaMalloc(&d_narrowBandIndx, N*sizeof(long int));

  cudaMalloc(&d_curvature, N*sizeof(float));

// The following code is a quick way to prototype
//
//     efficient memory management is sacrificed for
//       ease of protyping. the UNIFIED ARRAY can be accessed
//       by the same name from both CPU and GPU, but is
//       sometimes innefficient if you want to transfer
//       data CPU <==> GPU and the number of elements is small
//-------------------------------------------------------------
//
//=============================================================
//-------------------------------------------------------------
  // Allocate Unified Memory – accessible from CPU or GPU
  //cudaMallocManaged(&d_distance, N*sizeof(float));
  //cudaMallocManaged(&d_distanceNeg, N*sizeof(float));
  //cudaMallocManaged(&image, N*sizeof(int));
  //cudaMallocManaged(&d_image, N*sizeof(float));
  //cudaMallocManaged(&imageNeg, N*sizeof(float));
  //cudaMallocManaged(&d_mask, N*sizeof(float));
  //cudaMallocManaged(&d_maskNeg, N*sizeof(float));
//--------------------------------------------------------------
//==============================================================

  //read in data from file

  // read in image and mask -- in the future mask will be created by program
  ifstream inputmask;
  inputmask.open("mask.dat");
  ifstream inputdata;
  inputdata.open("rawImage.dat");

  for(long int i = 0; i<N; i++)
  {
      inputmask >> transfer;
      mask[i] = (float) transfer;
      maskNeg[i] = (float) transfer;

      inputdata >> transfer;
      image[i] = (float) transfer;
      
  }
  inputmask.close();
  inputdata.close();


  // transfer data to GPU from CPU
  cudaMemcpy(d_mask, mask, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maskNeg, maskNeg, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_image, image, N*sizeof(float), cudaMemcpyHostToDevice);


  // execute kernel code on the GPU
  // first parameter determines the number of thread blocks
  // second parameter determines the number of threads in a thread block


// build a signed distance function from the mask
  euclideanMap<<<blocks, threads>>>(N, d_mask, d_distance);
  negateImage<<<blocks, threads>>>(N, d_maskNeg);
  euclideanMap<<<blocks, threads>>>(N, d_maskNeg, d_distanceNeg);
  // SDF is stored in the second argument
  SDF<<<blocks, threads>>>(N, d_mask,d_distance,d_distanceNeg); 


  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // begin main loop of algorithm here==========================

  iterations = 0;
  // while loop


    // transfer data to CPU from GPU
    cudaMemcpy(distance, d_distance, N*sizeof(float), cudaMemcpyDeviceToHost);


    // obtain the curve's narrow band
    //   this is probably for the contour integral along the boundary?

    maxIndex = 0;
    for(long int i=0; i<N; i++)
    {
      if(distance[i] <= 1.2 && distance[i] >= -1.2)
      {
        narrowBandIndx[maxIndex] = i;
        maxIndex++ ;
      }
    }

    if (maxIndex > 0)
    {
      // print number of iterations
      cout << "number of iterations is    " << iterations << endl;

      // find interior and exterior mean
      numInteriorPx = 0;
      insideSum = 0.0;
      outsideSum = 0.0;
      for(long int i = 0; i<N; i++)
      {
        if( distance[i] <= 0 )
        {
          insideSum += image[i];
          numInteriorPx ++ ;
        }
        else
        {
          outsideSum += image[i];
        }
      }
      insideSum /= ( ((float) numInteriorPx) + eps  );
      outsideSum /= ( ((float) (N-numInteriorPx) ) + eps);

      cout << numInteriorPx << endl;

      // force from image information - this is an array operation
      //     put it on the GPU
      for(long int i =0; i<maxIndex;i++)
      {
        Force[i] =  ( image[narrowBandIndx[i]] - insideSum )*( image[narrowBandIndx[i]] - insideSum ) \
                     - ( image[narrowBandIndx[i]] - outsideSum )*( image[narrowBandIndx[i]] - outsideSum );
      }


      // transfer data to GPU from CPU
      cudaMemcpy(d_narrowBandIndx, narrowBandIndx, maxIndex*sizeof(long int), cudaMemcpyHostToDevice);

      // declare size of grid
      narrowGridN = (int) sqrt(maxIndex);
      narrowGridN++ ;

      // based on number of threads per block being 32
      numBlocksBand = (narrowGridN + 32 -1) / 32;

      dim3 blocksBand(numBlocksBand,numBlocksBand);
      dim3 threadsBand(numThreadsX,numThreadsX);


      // force from curvature penalty
      // def get_curvature
      get_curvature<<<blocksBand, threadsBand>>>(imgDimX, imgDimY, maxIndex, d_narrowBandIndx , d_distance, d_curvature);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      // transfer data to CPU from GPU
      cudaMemcpy(curvature, d_curvature, maxIndex*sizeof(float), cudaMemcpyDeviceToHost);

    }



  // end main loop of algorithm here============================

  // print to output file
  ofstream myfile;
  myfile.open("curvature.dat");

  //for(long int i = 0; i<N; i++)
  for(long int i = 0; i<maxIndex; i++)
  {
    myfile << curvature[i] << endl;
  }


  myfile.close();



  // Free memory-----------------------------------------------------

  // free GPU device memory
  cudaFree(d_distance);
  cudaFree(d_distanceNeg);
  cudaFree(d_image);
  cudaFree(d_mask);
  cudaFree(d_maskNeg);
  cudaFree(d_curvature);


  // free CPU side allocated arrays----------------------------------
  free(distance);
  free(distanceNeg);
  free(image);
  free(mask);
  free(maskNeg);
  free(narrowBandIndx);
  //free(Force);
  
  return 0;
}
# segmentation
