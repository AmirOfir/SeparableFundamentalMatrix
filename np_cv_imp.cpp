#include "np_cv_imp.hpp"



void WhereCondition(const InputArray pts, vector<int> &firstDimension, 
    vector<int> &secondDimension, bool(*condition)(float val))
{
    Mat points = pts.getMat();

    // First dimension
    for (size_t i = 0; i < points.size().height; i++)
    {

    }

}
