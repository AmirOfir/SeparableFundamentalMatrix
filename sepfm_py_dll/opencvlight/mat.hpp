
#ifndef OPL_MAT
#define OPL_MAT

#include <vector>
#include "cvdef.h"
#include "types.hpp"
#include "traits.h"
using namespace std;
namespace cv
{
class Mat
{
private:
    uchar *data;

    template <typename _Tp> void setAllValues(_Tp value)
    {
        _Tp *f = (_Tp*)data;
        for (int i = 0; i < rows*cols; ++i)
            f[i] = value;
    }
    template <typename _Tp> void copyDataFrom(_Tp *data)
    {
        if (data == NULL) return;
        _Tp *target = (_Tp*)(this->data);
        int size = rows * cols;
        for (int i = 0; i < size; ++i)
            target[i] = data[i];
    }
    void createData(int size, int type)
    {
        switch (type)
        {
        case CV_32F:
            data = (uchar *)(new float[size]);
            break;
        case CV_64F:
            data = (uchar *)(new double[size]);
            break;
        default:
            throw "Unknown type";
        }
    }
    
    int _type;
    bool delOnExit = true;
public:
    int rows;
    int cols;
    int dims;
    int steps[2];
    Mat(int rows=0,int cols=0, int type=0, uchar *data=NULL): dims(2), rows(rows), cols(cols), _type(type)
    {
        if (rows && cols && type)
        {
            createData(rows*cols, type);
            switch (_type)
            {
            case CV_32F:
                copyDataFrom((float*)data);
                steps[0] = cols * sizeof(float);
                steps[1] = sizeof(float);
                break;
            case CV_64F:
                copyDataFrom((double*)data);
                steps[0] = cols * sizeof(double);
                steps[1] = sizeof(double);
                break;
            default:
                throw "Unknown type";
            }
        }
            
    }
    template<typename _Tp> Mat(const vector<Point_<_Tp>> &vec)
    {
        rows = vec.size();
        cols = 2;
        type = traits::Type<_Tp>::value;
        dims = 2;
        createData(rows*cols, type);
        _Tp *d = this->data;
        for (auto p : vec)
        {
            *d = p.x;
            ++d;
            *d = p.y;
            ++d;
        }
    }
    ~Mat()
    {
        if (delOnExit && data) delete[] data;
    }

    static Mat zeros(int rows, int cols, int type)
    {
        Mat mat(rows,cols,type);
        switch (type)
        {
        case CV_32F:
            mat.setAllValues<float>(0);
            break;
        case CV_64F:
            mat.setAllValues<double>(0);
            break;
        default:
            throw "Not implemented";
            break;
        }
        return mat;
    }

    template<typename _Tp> _Tp& at(int row, int col)
    {
        _Tp *d = (_Tp*)data;
        return (d + (cols * row))[col];
    }
    template<typename _Tp> _Tp& at(int row, int col) const
    {
        const _Tp *d = (_Tp*)data;
        return (d + (cols * row))[col];
    }
    int type() { return _type; }
    void reset(int rows, int cols, int type)
    {
        if (data)
            delete[] data;
        this->rows = rows;
        this->cols = cols;
        this->_type = type;
        createData(rows*cols, type);
    }
    Mat row(int row)
    {
        Mat m(1, cols, _type);
        
    }
    template <typename _Tp> _Tp* ptr()
    {
        return (_Tp*)data;
    }
    int elemSize() const { return steps[1]; }
    
    void transpose(Mat &dst)
    {
        if (dst.rows != cols || dst.cols != rows|| dst._type != _type)
            dst.reset(cols, rows, _type);
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                if (_type == CV_32F)
                    dst.at<float>(col, row) = this->at<float>(row, col);
                else
                    dst.at<double>(col, row) = this->at<double>(row, col);
            }
        }
    }
    template <typename _Tp> void copyTo(Mat &dst)
    {
        if (dst.rows != rows || dst.cols != cols|| dst._type != _type)
            dst.reset(rows, cols, _type);
        _Tp *psrc = ptr<_Tp>();
        _Tp *pdst = dst.ptr<_Tp>();
        for (int i = 0)
        
    }
    void copyTo(Mat &dst)
    {
        if (dst.rows != rows || dst.cols != cols|| dst._type != _type)
            dst.reset(rows, cols, _type);
        if (_type == CV_32F)
            dst.copyDataFrom<float>((float*)data);
        else if (_type == CV_64F)
            dst.copyDataFrom<double>((double*)data);
    }
    void operator=(double v)
    {
        setAllValues(v);
    }
};

}
#endif
