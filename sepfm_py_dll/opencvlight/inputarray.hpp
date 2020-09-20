#ifndef OPL_INPUTARRAY
#define OPL_INPUTARRAY

#include "mat.hpp"

namespace cv
{
class InputArray
{
    void *obj;
    int flags;
    int type;
    int _dims;
    void Init(int flags, const void *obj)
    {
        flags = flags, obj = obj;
    }
        
public:
    InputArray(const Mat& m) 
    { 
        Init(1, &m); 
        _dims = m.dims;
    }
    template<typename _Tp> InputArray(const std::vector<Point_<_Tp>>& vec) 
    { 
        Init(2, &vec); 
        type = traits::Type<_Tp>::value;
    }
    template<typename _Tp> InputArray(const std::vector<Point3_<_Tp>>& vec) 
    { 
        Init(3, &vec); 
        type = traits::Type<_Tp>::value;
    }
    Mat getMat()
    {
        if (flags == 1) return *(const Mat*)obj;
        if (flags == 2 && type == CV_32F) return Mat(*(const vector<Point2f>*)obj);
    }
    int dims()
    {
        return _dims;
    }
    bool isMat()
    {
        return flags == 1;
    }
};

class OutputArray
{
private:
    Mat *mat;
public:
    OutputArray(Mat mat) : mat(&mat) {}
    void create(int _rows, int _cols, int mtype)
    {
        mat->reset(_rows, _cols, mtype);
    }
    void create(int dims, const int* size, int type)
    {
        mat->reset(size[0], size[1], type);
    }
    Mat getMat()
    {
        return (*mat);
    }
};

}
#endif // !OPL_INPUTARRAY