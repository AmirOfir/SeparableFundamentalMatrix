#include "SFM_finder.hpp"
#include "matching_lines.hpp"
#include "sfm_ransac.hpp"

using namespace cv::separableFundamentalMatrix;

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

void SFMEstimatorCallback::setFixedMatrices(InputArray _m1, InputArray _m2)
{
    _m1.getMat().convertTo(fixed1, CV_64F);
    _m2.getMat().convertTo(fixed2, CV_64F);
}

bool SFMEstimatorCallback::checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
{
    Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
    return !haveCollinearPoints<double>(ms1, count) && !haveCollinearPoints<double>(ms2, count);
}

int SFMEstimatorCallback::runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
{
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    cv::vconcat(m1, fixed1, m1);
    cv::vconcat(m2, fixed2, m2);

    Mat F = cv::findFundamentalMat(m1, m2, FM_8POINT);
    
    if (!F.empty() && F.data[0] != NULL)
    {
        if( _model.empty() )
            _model.create(3, 3, CV_64F);
        
        F.convertTo(_model, F.type());
        return true;
    }

    return false;
}

void SFMEstimatorCallback::computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
{
    Mat __m1 = _m1.getMat(), __m2 = _m2.getMat(), __model = _model.getMat();
    int i, count = __m1.checkVector(2);
    _err.create(count, 1, CV_32F);
    float* err = _err.getMat().ptr<float>();
    
    const Point2d* m1 = __m1.ptr<Point2d>();
    const Point2d* m2 = __m2.ptr<Point2d>();
    const double* F = __model.ptr<double>();
    
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

        err[i] =(float) 0.5 * (sqrt(d1*d1*s1) + sqrt(d2*d2*s2));
        //err[i] = (double)std::max(d1*d1*s1, d2*d2*s2);
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
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    
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