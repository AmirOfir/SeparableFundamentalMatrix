#ifndef _OPENCV_FM_FINDER_H_
#define _OPENCV_FM_FINDER_H_
#include "precomp.hpp"


namespace cv {
namespace separableFundamentalMatrix {

Mat findFundamentalMatFullRansac(InputArray pts1, InputArray pts2, float inlier_ratio = 0.4);

}
}
#endif