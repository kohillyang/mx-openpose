/*
 * Adapted from caffe2: github.com/caffe2/caffe2
 */
#include "bilinear.h"
#include "mobula_op.h"
#include <memory>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)
template<typename Dtype>
void putGaussianMaps(Dtype* entry, float src_x, float src_y, int stride, int grid_x, int grid_y, float sigma){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-src_x)*(x-src_x) + (y-src_y)*(y-src_y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1)
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template <typename T>
// bboxes, keypoints, h, w, nperson, nparts, out_temp
MOBULA_KERNEL heat_gen_kernel(const T* keypoints, const int h, const int w, const int stride,
                              const int nperson, const int nparts, const float sigma, T* output) {
    for(int i = 0; i < nperson; i++){
        for(int j=0; j < nparts; j++){
            float x = keypoints[i * nparts * 3 + j * 3 + 0];
            float y = keypoints[i * nparts * 3 + j * 3 + 0];
            int available = keypoints[i * nparts * 3 + j * 3 + 0];
            if(available){
                putGaussianMaps(output, x, y, stride, w / stride, h / stride, sigma);
            }
        }
    }
}


}  // namespace mobula
