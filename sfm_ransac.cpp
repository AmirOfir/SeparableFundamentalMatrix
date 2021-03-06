// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "pointset_registrator.hpp"

using namespace cv;
using namespace std;
using namespace cv::separableFundamentalMatrix;

// Code from Calib3d which is sadly private
namespace cv
{

int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
{
    if( modelPoints <= 0 )
        CV_Error( Error::StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = std::log(num);
    denom = std::log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
}

class RANSACPointSetRegistrator : public PointSetRegistrator
{
public:
    RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
      : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters) {}

    int findInliers( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh ) const
    {
        cb->computeError( m1, m2, model, err );
        mask.create(err.size(), CV_8U);

        CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
        const float* errptr = err.ptr<float>();
        uchar* maskptr = mask.ptr<uchar>();
        float t = (float)(thresh/**thresh*/);
        int i, n = (int)err.total(), nz = 0;
        for( i = 0; i < n; i++ )
        {
            int f = errptr[i] <= t;
            maskptr[i] = (uchar)f;
            nz += f;
        }
        return nz;
    }

    bool getSubset( const Mat& m1, const Mat& m2, Mat& ms1, Mat& ms2, RNG& rng, int maxAttempts=1000 ) const
    {
        vector<int> idx(modelPoints, 0);

        const int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        const int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;

        int esz1 = (int)m1.elemSize1() * d1;
        int esz2 = (int)m2.elemSize1() * d2;
        CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        const int count = m1.checkVector(d1);
        const int count2 = m2.checkVector(d2);
        CV_Assert(count >= modelPoints && count == count2);

        const int *m1ptr = m1.ptr<int>();
        const int *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>();
        int *ms2ptr = ms2.ptr<int>();

        for( int iters = 0; iters < maxAttempts; ++iters )
        {
            int i;

            for( i = 0; i < modelPoints; ++i )
            {
                int idx_i;

                for ( idx_i = rng.uniform(0, count);
                    std::find(idx.begin(), idx.end(), idx_i) != idx.end();
                    idx_i = rng.uniform(0, count) )
                {}

                idx[i] = idx_i;

                for( int k = 0; k < esz1; ++k )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];

                for( int k = 0; k < esz2; ++k )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
            }

            if( cb->checkSubset(ms1, ms2, i) )
                return true;
        }

        return false;
    }

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask, int &inliers) const CV_OVERRIDE
    {
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat err, mask, model, bestModel, ms1, ms2;

        int iter, niters = MAX(maxIters, 1);
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        Mat bestMask0, bestMask;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            bestMask0 = bestMask = _mask.getMat();
            CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
        }
        else
        {
            bestMask.create(count, 1, CV_8U);
            bestMask0 = bestMask;
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            bestMask.setTo(Scalar::all(1));
            return true;
        }

        for( iter = 0; iter < niters; iter++ )
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                int goodCount = findInliers( m1, m2, model_i, err, mask, threshold );

                if( goodCount > MAX(maxGoodCount, modelPoints-1) )
                {
                    std::swap(mask, bestMask);
                    model_i.copyTo(bestModel);
                    maxGoodCount = goodCount;
                    //niters = RANSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters );
                }
            }
        }

        if( maxGoodCount > 0 )
        {
            if( bestMask.data != bestMask0.data )
            {
                if( bestMask.size() == bestMask0.size() )
                    bestMask.copyTo(bestMask0);
                else
                    transpose(bestMask, bestMask0);
            }
            bestModel.copyTo(_model);
            result = true;
            inliers = maxGoodCount;
        }
        else
            _model.release();

        return result;
    }

    void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

    Ptr<PointSetRegistrator::Callback> cb;
    int modelPoints;
    double threshold;
    double confidence;
    int maxIters;
};

Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
    int _modelPoints, double _threshold, double _confidence, int _maxIters)
{
    return Ptr<PointSetRegistrator>(
        new RANSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters));
}

}
