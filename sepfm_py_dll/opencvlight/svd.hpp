#ifndef OPL_SVD
#define OPL_SVD

#include "cvdef.h"
#include "cv_cpu_dispatch.h"
#include "inputarray.hpp"

namespace cv
{
/** @brief Singular Value Decomposition

Class for computing Singular Value Decomposition of a floating-point
matrix. The Singular Value Decomposition is used to solve least-square
problems, under-determined linear systems, invert matrices, compute
condition numbers, and so on.

If you want to compute a condition number of a matrix or an absolute value of
its determinant, you do not need `u` and `vt`. You can pass
flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that full-size u
and vt must be computed, which is not necessary most of the time.

@sa invert, solve, eigen, determinant
*/
class SVD
{
private:
    static inline size_t alignSize(size_t sz, int n)
    {
        return (sz + n-1) & -n;
    }
public:
    enum Flags {
        /** allow the algorithm to modify the decomposed matrix; it can save space and speed up
            processing. currently ignored. */
        MODIFY_A = 1,
        /** indicates that only a vector of singular values `w` is to be processed, while u and vt
            will be set to empty matrices */
        NO_UV    = 2,
        /** when the matrix is not square, by default the algorithm produces u and vt matrices of
            sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
            specified, u and vt will be full-size square orthogonal matrices.*/
        FULL_UV  = 4
    };

    /** @overload
    initializes an empty SVD structure and then calls SVD::operator()
    @param src decomposed matrix. The depth has to be CV_32F or CV_64F.
    @param flags operation flags (SVD::Flags)
      */
    SVD(Mat src, int flags = 0)
    {
        CV_Assert(flags == SVD::FULL_UV);

        int type = src.type();
        int m = src.rows, n = src.cols;
        bool full_uv = (flags & SVD::FULL_UV) != 0;
        CV_Assert( type == CV_32F || type == CV_64F );

        bool at = false;
        if( m < n )
        {
            std::swap(m, n);
            at = true;
        }

        int urows = full_uv ? m : n;

        //size_t esz = src.elemSize(), astep = alignSize(m*esz, 16), vstep = alignSize(n*esz, 16);
        //AutoBuffer<uchar> _buf(urows*astep + n*vstep + n*esz + 32);
        //uchar* buf = alignPtr(_buf.data(), 16);
        Mat temp_a(n, m, type);
        Mat temp_w(n, 1, type);
        Mat temp_u(urows, m, type);
        Mat temp_v(n, n, type);

        temp_u = 0;
        if( !at )
            transpose(src, temp_a);
        else
            src.copyTo(temp_a);
    }

    Mat u, w, vt;
};


///////////////////////////////////////////// SVD /////////////////////////////////////////////



}
#endif // !OPL_SVD