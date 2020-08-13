#include "np_cv_imp.hpp"

using namespace cv;
using namespace std;
namespace cv {
    namespace separableFundamentalMatrix {

        void WhereCondition(const InputArray pts, vector<int> &firstDimension,
            vector<int> &secondDimension, bool(*condition)(float val))
        {
            Mat points = pts.getMat();

            // First dimension
            for (size_t i = 0; i < points.size().height; i++)
            {

            }

        }

    }
}