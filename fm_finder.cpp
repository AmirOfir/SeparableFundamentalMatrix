#include "precomp.hpp"
#include "fm_finder.hpp"
#include <iostream>
#include "sfm_ransac.hpp"

using namespace cv;

template <typename _Tp>
bool haveCollinearPoints( const Mat& m, int count )
{
    int j, k, i = count-1;
    const Point_<_Tp>* ptr = m.ptr<Point_<_Tp>>();

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points
    // also checks that points are not too close to each other
    for( j = 0; j < i; j++ )
    {
        _Tp dx1 = ptr[j].x - ptr[i].x;
        _Tp dy1 = ptr[j].y - ptr[i].y;
        for( k = 0; k < j; k++ )
        {
            _Tp dx2 = ptr[k].x - ptr[i].x;
            _Tp dy2 = ptr[k].y - ptr[i].y;
            if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                return true;
        }
    }
    return false;
}

// ------------------------------------------------------
//                   OPENCV findFundamentalMat
// Imported from fundam.cpp
/*static int run7Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
{
    double a[7*9], w[7], u[9*9], v[9*9], c[4], r[3] = {0};
    double* f1, *f2;
    double t0, t1, t2;
    Mat A( 7, 9, CV_64F, a );
    Mat U( 7, 9, CV_64F, u );
    Mat Vt( 9, 9, CV_64F, v );
    Mat W( 7, 1, CV_64F, w );
    Mat coeffs( 1, 4, CV_64F, c );
    Mat roots( 1, 3, CV_64F, r );
    const Point2f* m1 = _m1.ptr<Point2f>();
    const Point2f* m2 = _m2.ptr<Point2f>();
    double* fmatrix = _fmatrix.ptr<double>();
    int i, k, n;

    Point2d m1c(0, 0), m2c(0, 0);
    double t, scale1 = 0, scale2 = 0;
    const int count = 7;

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        m1c += Point2d(m1[i]);
        m2c += Point2d(m2[i]);
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./count;
    m1c *= t;
    m2c *= t;

    for( i = 0; i < count; i++ )
    {
        scale1 += norm(Point2d(m1[i].x - m1c.x, m1[i].y - m1c.y));
        scale2 += norm(Point2d(m2[i].x - m2c.x, m2[i].y - m2c.y));
    }

    scale1 *= t;
    scale2 *= t;

    if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )
        return 0;

    scale1 = std::sqrt(2.)/scale1;
    scale2 = std::sqrt(2.)/scale2;

    // form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    for( i = 0; i < 7; i++ )
    {
        double x0 = (m1[i].x - m1c.x)*scale1;
        double y0 = (m1[i].y - m1c.y)*scale1;
        double x1 = (m2[i].x - m2c.x)*scale2;
        double y1 = (m2[i].y - m2c.y)*scale2;

        a[i*9+0] = x1*x0;
        a[i*9+1] = x1*y0;
        a[i*9+2] = x1;
        a[i*9+3] = y1*x0;
        a[i*9+4] = y1*y0;
        a[i*9+5] = y1;
        a[i*9+6] = x0;
        a[i*9+7] = y0;
        a[i*9+8] = 1;
    }

    // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
    // the solution is linear subspace of dimensionality 2.
    // => use the last two singular vectors as a basis of the space
    // (according to SVD properties)
    SVDecomp( A, W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV );
    f1 = v + 7*9;
    f2 = v + 8*9;

    // f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary fundamental matrix,
    // as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
    // so f ~ lambda*f1 + (1 - lambda)*f2.
    // use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
    // it will be a cubic equation.
    // find c - polynomial coefficients.
    for( i = 0; i < 9; i++ )
        f1[i] -= f2[i];

    t0 = f2[4]*f2[8] - f2[5]*f2[7];
    t1 = f2[3]*f2[8] - f2[5]*f2[6];
    t2 = f2[3]*f2[7] - f2[4]*f2[6];

    c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

    c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
    f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
    f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
    f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
    f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
    f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
    f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

    t0 = f1[4]*f1[8] - f1[5]*f1[7];
    t1 = f1[3]*f1[8] - f1[5]*f1[6];
    t2 = f1[3]*f1[7] - f1[4]*f1[6];

    c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
    f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
    f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
    f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
    f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
    f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
    f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

    c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

    // solve the cubic equation; there can be 1 to 3 roots ...
    n = solveCubic( coeffs, roots );

    if( n < 1 || n > 3 )
        return n;

    // transformation matrices
    Matx33d T1( scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 );
    Matx33d T2( scale2, 0, -scale2*m2c.x, 0, scale2, -scale2*m2c.y, 0, 0, 1 );

    for( k = 0; k < n; k++, fmatrix += 9 )
    {
        // for each root form the fundamental matrix
        double lambda = r[k], mu = 1.;
        double s = f1[8]*r[k] + f2[8];

        // normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
        if( fabs(s) > DBL_EPSILON )
        {
            mu = 1./s;
            lambda *= mu;
            fmatrix[8] = 1.;
        }
        else
            fmatrix[8] = 0.;

        for( i = 0; i < 8; i++ )
            fmatrix[i] = f1[i]*lambda + f2[i]*mu;

        // de-normalize
        Mat F(3, 3, CV_64F, fmatrix);
        F = T2.t() * F * T1;

        // make F(3,3) = 1
        if(fabs(F.at<double>(8)) > FLT_EPSILON )
            F *= 1. / F.at<double>(8);
    }

    return n;
}*/

static int run8Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
{
    Point2d m1c(0,0), m2c(0,0);
    double t, scale1 = 0, scale2 = 0;

    const Point2f* m1 = _m1.ptr<Point2f>();
    const Point2f* m2 = _m2.ptr<Point2f>();
    CV_Assert( (_m1.cols == 1 || _m1.rows == 1) && _m1.size() == _m2.size());
    int i, count = _m1.checkVector(2);

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        m1c += Point2d(m1[i]);
        m2c += Point2d(m2[i]);
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./count;
    m1c *= t;
    m2c *= t;

    for( i = 0; i < count; i++ )
    {
        scale1 += norm(Point2d(m1[i].x - m1c.x, m1[i].y - m1c.y));
        scale2 += norm(Point2d(m2[i].x - m2c.x, m2[i].y - m2c.y));
    }

    scale1 *= t;
    scale2 *= t;

    if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )
        return 0;

    scale1 = std::sqrt(2.)/scale1;
    scale2 = std::sqrt(2.)/scale2;

    Matx<double, 9, 9> A;

    // form a linear system Ax=0: for each selected pair of points m1 & m2,
    // the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
    // to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
    for( i = 0; i < count; i++ )
    {
        double x1 = (m1[i].x - m1c.x)*scale1;
        double y1 = (m1[i].y - m1c.y)*scale1;
        double x2 = (m2[i].x - m2c.x)*scale2;
        double y2 = (m2[i].y - m2c.y)*scale2;
        Vec<double, 9> r( x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1 );
        A += r*r.t();
    }

    Vec<double, 9> W;
    Matx<double, 9, 9> V;

    eigen(A, W, V);

    for( i = 0; i < 9; i++ )
    {
        if( fabs(W[i]) < DBL_EPSILON )
            break;
    }

    if( i < 8 )
        return 0;

    Matx33d F0( V.val + 9*8 ); // take the last column of v as a solution of Af = 0

    // make F0 singular (of rank 2) by decomposing it with SVD,
    // zeroing the last diagonal element of W and then composing the matrices back.

    Vec3d w;
    Matx33d U;
    Matx33d Vt;

    SVD::compute( F0, w, U, Vt);
    w[2] = 0.;

    F0 = U*Matx33d::diag(w)*Vt;

    // apply the transformation that is inverse
    // to what we used to normalize the point coordinates
    Matx33d T1( scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 );
    Matx33d T2( scale2, 0, -scale2*m2c.x, 0, scale2, -scale2*m2c.y, 0, 0, 1 );

    F0 = T2.t()*F0*T1;

    // make F(3,3) = 1
    if( fabs(F0(2,2)) > FLT_EPSILON )
        F0 *= 1./F0(2,2);

    Mat(F0).copyTo(_fmatrix);

    return 1;
}

class FMEstimatorCallback CV_FINAL : public PointSetRegistrator::Callback
{
public:
    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const CV_OVERRIDE
    {
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        return !haveCollinearPoints<float>(ms1, count) && !haveCollinearPoints<float>(ms2, count);
    }

    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const CV_OVERRIDE
    {
        double f[9*3];
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int count = m1.checkVector(2);
        Mat F(count == 7 ? 9 : 3, 3, CV_64F, f);
        int n = count == 7 ? 0 /*run7Point(m1, m2, F)*/ : run8Point(m1, m2, F);

        if( n == 0 )
            _model.release();
        else
            F.rowRange(0, n*3).copyTo(_model);

        return n;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const CV_OVERRIDE
    {
        Mat __m1 = _m1.getMat(), __m2 = _m2.getMat(), __model = _model.getMat();
        int i, count = __m1.checkVector(2);
        const Point2f* m1 = __m1.ptr<Point2f>();
        const Point2f* m2 = __m2.ptr<Point2f>();
        const double* F = __model.ptr<double>();
        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();

        for( i = 0; i < count; i++ )
        {
            double a, b, c, d1, d2, s1, s2;

            a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
            b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
            c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

            s2 = 1./(a*a + b*b);
            d2 = m2[i].x*a + m2[i].y*b + c;

            a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
            b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
            c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

            s1 = 1./(a*a + b*b);
            d1 = m1[i].x*a + m1[i].y*b + c;

            err[i] = (float)std::max(d1*d1*s1, d2*d2*s2);
        }
    }
};


cv::Mat cv::separableFundamentalMatrix::findFundamentalMatFullRansac(InputArray _points1, InputArray _points2, float inlierRatio)
{
    int maxIterations = int((log(0.01) / log(1 - pow(inlierRatio, 5)))) + 1;
    int method = FM_RANSAC;
    auto _mask = noArray();
    double ransacReprojThreshold = 3;
    double confidence = 0.99;

    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat m1, m2, F;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? m1 : m2;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert( m1.checkVector(2) == m2.checkVector(2) );

    if( npoints < 7 )
        return Mat();

    Ptr<PointSetRegistrator::Callback> cb = makePtr<FMEstimatorCallback>();
    int result;

    if( npoints == 7 || method == FM_8POINT )
    {
        result = cb->runKernel(m1, m2, F);
        if( _mask.needed() )
        {
            _mask.create(npoints, 1, CV_8U, -1, true);
            Mat mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
            mask.setTo(Scalar::all(1));
        }
    }
    else
    {
        if( ransacReprojThreshold <= 0 )
            ransacReprojThreshold = 3;
        if( confidence < DBL_EPSILON || confidence > 1 - DBL_EPSILON )
            confidence = 0.99;

        bool status;
        if ((method & ~3) == FM_RANSAC && npoints >= 15)
            status = createRANSACPointSetRegistrator(cb, 7, ransacReprojThreshold, confidence, maxIterations)->run(m1, m2, F, _mask, result);
        else
            //result = createLMeDSPointSetRegistrator(cb, 7, confidence)->run(m1, m2, F, _mask);
            NULL;
    }

    if( result <= 0 )
        return Mat();

    return F;
}

