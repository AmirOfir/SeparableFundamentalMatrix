#include "SFM_finder.hpp"
#include "matching_lines.hpp"
#include "sfm_ransac.hpp"

using namespace cv::separableFundamentalMatrix;

bool haveCollinearPoints( const Mat& m, int count )
{
    int j, k, i = count-1;
    const Point2f* ptr = m.ptr<Point2f>();

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points
    // also checks that points are not too close to each other
    for( j = 0; j < i; j++ )
    {
        double dx1 = ptr[j].x - ptr[i].x;
        double dy1 = ptr[j].y - ptr[i].y;
        for( k = 0; k < j; k++ )
        {
            double dx2 = ptr[k].x - ptr[i].x;
            double dy2 = ptr[k].y - ptr[i].y;
            if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                return true;
        }
    }
    return false;
}

/**
    * Compute the fundamental matrix using the 7-point algorithm.
    *
    * \f[
    *  (\mathrm{m2}_i,1)^T \mathrm{fmatrix} (\mathrm{m1}_i,1) = 0
    * \f]
    *
    * @param _m1 Contain points in the reference view. Depth CV_32F with 2-channel
    *            1 column or 1-channel 2 columns. It has 7 rows.
    * @param _m2 Contain points in the other view. Depth CV_32F with 2-channel
    *            1 column or 1-channel 2 columns. It has 7 rows.
    * @param _fmatrix Output fundamental matrix (or matrices) of type CV_64FC1.
    *                 The user is responsible for allocating the memory before calling
    *                 this function.
    * @return Number of fundamental matrices. Valid values are 1, 2 or 3.
    *  - 1, row 0 to row 2 in _fmatrix is a valid fundamental matrix
    *  - 2, row 3 to row 5 in _fmatrix is a valid fundamental matrix
    *  - 3, row 6 to row 8 in _fmatrix is a valid fundamental matrix
    *
    * Note that the computed fundamental matrix is normalized, i.e.,
    * the last element \f$F_{33}\f$ is 1.
    */
int run7Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
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
}

/**
    * Compute the fundamental matrix using the 8-point algorithm.
    *
    * \f[
    *  (\mathrm{m2}_i,1)^T \mathrm{fmatrix} (\mathrm{m1}_i,1) = 0
    * \f]
    *
    * @param _m1 Contain points in the reference view. Depth CV_32F with 2-channel
    *            1 column or 1-channel 2 columns. It has 8 rows.
    * @param _m2 Contain points in the other view. Depth CV_32F with 2-channel
    *            1 column or 1-channel 2 columns. It has 8 rows.
    * @param _fmatrix Output fundamental matrix (or matrices) of type CV_64FC1.
    *                 The user is responsible for allocating the memory before calling
    *                 this function.
    * @return 1 on success, 0 on failure.
    *
    * Note that the computed fundamental matrix is normalized, i.e.,
    * the last element \f$F_{33}\f$ is 1.
    */
template <typename _TSrc>
int run8Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
{
    #define PointT Point_<_TSrc>
    Point2d m1c(0,0), m2c(0,0);
    double t, scale1 = 0, scale2 = 0;

    const PointT* m1 = _m1.ptr<PointT>();
    const PointT* m2 = _m2.ptr<PointT>();
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

void SFMEstimatorCallback::setFixedMatrices(InputArray _m1, InputArray _m2)
{
    _m1.getMat().convertTo(fixed1, CV_64F);
    _m2.getMat().convertTo(fixed2, CV_64F);
}

bool SFMEstimatorCallback::checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
{
    Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
    return !haveCollinearPoints(ms1, count) && !haveCollinearPoints(ms2, count);
}

int SFMEstimatorCallback::runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
{
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    cv::vconcat(m1, fixed1, m1);
    cv::vconcat(m2, fixed2, m2);

    Mat F = cv::findFundamentalMat(m1, m2, FM_8POINT);
    
    F.copyTo(_model);

    return F.empty() ? 0 : 1;
    /*int count = m1.checkVector(2);
    CV_Assert(count == 8);

    Mat F(3, 3, CV_64F);
    int n;
    if (m1.type() == traits::Type<float>::value)
        n = run8Point<float>(m1, m2, F);
    else
        n = run8Point<double>(m1, m2, F);

    if( n == 0 )
        _model.release();
    else
        F.rowRange(0, n*3).copyTo(_model);
        */
    //return n;
}

void SFMEstimatorCallback::computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
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
  

SeparableFundamentalMatFindCommand::SeparableFundamentalMatFindCommand(InputArray _points1, InputArray _points2, int _imSizeHOrg, 
    int _imSizeWOrg, float _inlierRatio, int _inlierThreashold, float _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
    int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints)
    :imSizeHOrg(_imSizeHOrg), imSizeWOrg(_imSizeWOrg), inlierRatio(_inlierRatio), inlierThreashold(_inlierThreashold),
    houghRescale(_houghRescale),  numMatchingPtsToUse(_numMatchingPtsToUse), pixelRes(_pixelRes),
    minHoughPints(_minHoughPints), thetaRes(_thetaRes),  maxDistancePtsLine(_maxDistancePtsLine), topLineRetries(_topLineRetries), 
    minSharedPoints(_minSharedPoints)
{
    _points1.getMat().convertTo(points1, CV_32F);
    _points2.getMat().convertTo(points2, CV_32F);
    
    nPoints = points1.rows;
    if (nPoints == 2 && points1.cols > 2)
    {
        points1 = points1.reshape(2, points1.cols);
        nPoints = points1.rows;
    }
    if (points2.rows == 2 && points2.cols == nPoints)
        points2 = points2.reshape(2, nPoints);
        
    int pts1Count = _points1.isVector() ? _points1.getMat().size().width : _points1.getMat().size().height;
    if (DEFAULT_HOUGH_RESCALE == houghRescale)
        houghRescale = float(2 * pts1Count) / imSizeHOrg;
    else if (houghRescale > 1) // Only subsample
        houghRescale = 1;
    isExecuting = false;
}

Mat SeparableFundamentalMatFindCommand::Execute()
{
    isExecuting = true;

    Mat f;
    
    auto topMatchingLines = FindMatchingLines(imSizeHOrg, imSizeWOrg, points1, points2, topLineRetries, houghRescale, maxDistancePtsLine,
        minHoughPints, pixelRes, thetaRes, numMatchingPtsToUse, minSharedPoints, inlierRatio);

    // We don't have at least one line
    if (!topMatchingLines.size()) return f;
    
    Mat mask;
    f = Mat(3, 3, CV_64F);

    vector<tuple<Mat,Mat>> lines = SFMRansac::prepareLinesForRansac(topMatchingLines);

    Ptr<SFMEstimatorCallback> cb = makePtr<SFMEstimatorCallback>();
    int result;

    for (auto &line : lines)
    {
        cb->setFixedMatrices(get<0>(line), get<1>(line));
        result = createRANSACPointSetRegistrator(cb, 5, 3.)->run(points1, points2, f, mask);

        if (result > 0)
            break;
    }

    if( result <= 0 )
        return Mat();
    return f;
}

/*
cv::Mat findFundamentalMat( InputArray _points1, InputArray _points2, vector<Mat> matchingLines,
                                int method, double ransacReprojThreshold, double confidence,
                                int maxIters, OutputArray _mask )
{
    CV_INSTRUMENT_REGION();

    if (method >= 32 && method <= 38)
        return usac::findFundamentalMat(_points1, _points2, method,
            ransacReprojThreshold, confidence, maxIters, _mask);

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

        if( (method & ~3) == FM_RANSAC && npoints >= 15 )
            result = createRANSACPointSetRegistrator(cb, 7, ransacReprojThreshold, confidence, maxIters)->run(m1, m2, F, _mask);
        else
            result = createLMeDSPointSetRegistrator(cb, 7, confidence)->run(m1, m2, F, _mask);
    }

    if( result <= 0 )
        return Mat();

    return F;
}*/

/*

*/
// pts1 is Mat of shape(X,2)
// pts2 is Mat of shape(X,2)
Mat cv::separableFundamentalMatrix::findSeparableFundamentalMat(InputArray _points1, InputArray _points2, int _imSizeHOrg, int _imSizeWOrg,
        float _inlierRatio, int _inlierThreashold, float _houghRescale, int _numMatchingPtsToUse, int _pixelRes,
        int _minHoughPints, int _thetaRes, float _maxDistancePtsLine, int _topLineRetries, int _minSharedPoints)
{
    
    SeparableFundamentalMatFindCommand command(_points1, _points2, _imSizeHOrg, _imSizeWOrg, _inlierRatio, _inlierThreashold, _houghRescale,
        _numMatchingPtsToUse, _pixelRes, _minHoughPints, _thetaRes, _maxDistancePtsLine, _topLineRetries, _minSharedPoints);
    return command.Execute();


        /*
    int pts1Count = pts1.isVector() ? pts1.getMat().size().width : pts1.getMat().size().height;
    if (hough_rescale == DEFAULT_HOUGH_RESCALE)
        hough_rescale = float(2 * pts1Count) / im_size_h_org;
    else if (hough_rescale > 1) // Only subsample
        hough_rescale = 1;

    auto topMatchingLines = FindMatchingLines(im_size_h_org, im_size_w_org, pts1, pts2, top_line_retries, hough_rescale, 
        max_distance_pts_line, min_hough_points, pixel_res, theta_res, num_matching_pts_to_use, min_shared_points, inlier_ratio);

    // We have at least one line
    if (topMatchingLines.size())
    {
        Mat data = prepareDataForRansac(pts1, pts2);
        vector<Mat> lines = prepareLinesForRansac(topMatchingLines);
        uint maxNumIterations = (uint)std::floor(1 + std::log(0.01) / std::log(1 - pow(inlier_ratio, 5)));
    }
    */
    return Mat();
}