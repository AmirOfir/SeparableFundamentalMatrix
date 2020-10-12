#include <Python.h>
#include <ndarrayobject.h>

#include "precomp.hpp"
#include "SFM_finder.hpp"
#include "fm_finder.hpp"
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::separableFundamentalMatrix;

static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}



static PyObject* failmsgp(const char *fmt, ...)
{
char str[1000];

va_list ap;
va_start(ap, fmt);
vsnprintf(str, sizeof(str), fmt, ap);
va_end(ap);

PyErr_SetString(PyExc_TypeError, str);
return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}


class NumpyAllocator:
		public MatAllocator {
public:
	NumpyAllocator() {
		stdAllocator = Mat::getStdAllocator();
	}
	~NumpyAllocator() {
	}

	UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
			size_t* step) const {
		UMatData* u = new UMatData(this);
		u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
		npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
		for (int i = 0; i < dims - 1; i++)
			step[i] = (size_t) _strides[i];
		step[dims - 1] = CV_ELEM_SIZE(type);
		u->size = sizes[0] * step[0];
		u->userdata = o;
		return u;
	}

	UMatData* allocate(int dims0, const int* sizes, int type, void* data,
			size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const {
		if (data != 0) {
			CV_Error(Error::StsAssert, "The data should normally be NULL!");
			// probably this is safe to do in such extreme case
			return stdAllocator->allocate(dims0, sizes, type, data, step, (AccessFlag) flags,
					usageFlags);
		}
		PyEnsureGIL gil;

		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int) (sizeof(size_t) / 8);
		int typenum =
				depth == CV_8U ? NPY_UBYTE :
				depth == CV_8S ? NPY_BYTE :
				depth == CV_16U ? NPY_USHORT :
				depth == CV_16S ? NPY_SHORT :
				depth == CV_32S ? NPY_INT :
				depth == CV_32F ? NPY_FLOAT :
				depth == CV_64F ?
									NPY_DOUBLE :
									f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
		int i, dims = dims0;
		cv::AutoBuffer<npy_intp> _sizes(dims + 1);
		for (i = 0; i < dims; i++)
			_sizes[i] = sizes[i];
		if (cn > 1)
			_sizes[dims++] = cn;
		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
		if (!o)
			CV_Error_(Error::StsError,
					("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		return allocate(o, dims0, sizes, type, step);
	}

    bool allocate(cv::UMatData *u, AccessFlag accessFlags,cv::UMatUsageFlags usageFlags) const
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

	void deallocate(UMatData* u) const {
		if (u) {
			PyEnsureGIL gil;
			PyObject* o = (PyObject*) u->userdata;
			Py_XDECREF(o);
			delete u;
		}
	}

	const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

int init_numpy() {
    import_array();
    return 0;
}

const static int numpy_initialized = init_numpy();

// Credits: https://github.com/Algomorph/pyboostcvconverter/blob/master/src/pyboost_cv3_converter.cpp

Mat fromNDArrayToMat(PyObject* o) {
    cv::Mat m;
	bool allowND = true;
	if (!PyArray_Check(o)) {
		failmsg("argument is not a numpy array");
		if (!m.data)
			m.allocator = &g_numpyAllocator;
	} else {
		PyArrayObject* oarr = (PyArrayObject*) o;

		bool needcopy = false, needcast = false;
		int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
		int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
					typenum == NPY_USHORT ? CV_16U :
					typenum == NPY_SHORT ? CV_16S :
					typenum == NPY_INT ? CV_32S :
					typenum == NPY_INT32 ? CV_32S :
					typenum == NPY_FLOAT ? CV_32F :
					typenum == NPY_DOUBLE ? CV_64F : -1;

		if (type < 0) {
			if (typenum == NPY_INT64 || typenum == NPY_UINT64
					|| type == NPY_LONG) {
				needcopy = needcast = true;
				new_typenum = NPY_INT;
				type = CV_32S;
			} else {
				failmsg("Argument data type is not supported");
				m.allocator = &g_numpyAllocator;
				return m;
			}
		}

#ifndef CV_MAX_DIM
		const int CV_MAX_DIM = 32;
#endif

		int ndims = PyArray_NDIM(oarr);
		if (ndims >= CV_MAX_DIM) {
			failmsg("Dimensionality of argument is too high");
			if (!m.data)
				m.allocator = &g_numpyAllocator;
			return m;
		}

		int size[CV_MAX_DIM + 1];
		size_t step[CV_MAX_DIM + 1];
		size_t elemsize = CV_ELEM_SIZE1(type);
		const npy_intp* _sizes = PyArray_DIMS(oarr);
		const npy_intp* _strides = PyArray_STRIDES(oarr);
		bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

		for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
			// these checks handle cases of
			//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
			//  b) transposed arrays, where _strides[] elements go in non-descending order
			//  c) flipped arrays, where some of _strides[] elements are negative
			if ((i == ndims - 1 && (size_t) _strides[i] != elemsize)
					|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
				needcopy = true;
		}

		if (ismultichannel && _strides[1] != (npy_intp) elemsize * _sizes[2])
			needcopy = true;

		if (needcopy) {

			if (needcast) {
				o = PyArray_Cast(oarr, new_typenum);
				oarr = (PyArrayObject*) o;
			} else {
				oarr = PyArray_GETCONTIGUOUS(oarr);
				o = (PyObject*) oarr;
			}

			_strides = PyArray_STRIDES(oarr);
		}

		for (int i = 0; i < ndims; i++) {
			size[i] = (int) _sizes[i];
			step[i] = (size_t) _strides[i];
		}

		// handle degenerate case
		if (ndims == 0) {
			size[ndims] = 1;
			step[ndims] = elemsize;
			ndims++;
		}

		if (ismultichannel) {
			ndims--;
			type |= CV_MAKETYPE(0, size[2]);
		}

		if (ndims > 2 && !allowND) {
			failmsg("%s has more than 2 dimensions");
		} else {

			m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
			m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
			m.addref();

			if (!needcopy) {
				Py_INCREF(o);
			}
		}
		m.allocator = &g_numpyAllocator;
	}
	return m;
}

PyObject* fromMatToNDArray(const Mat& m) {
	if (!m.data)
		Py_RETURN_NONE;
		Mat temp,
	*p = (Mat*) &m;
	if (!p->u || p->allocator != &g_numpyAllocator) {
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	PyObject* o = (PyObject*) p->u->userdata;
	Py_INCREF(o);
	return o;
}

int pyopencv_to(const PyObject* o, Mat& m, const char* name, bool allowND)
{
    
    if( !PyArray_Check(o) )    // this line makes error without message
    {
        failmsg("%s is not a numpy array", name);
        return false;
    }

    // NPY_LONG (64 bit) is converted to CV_32S (32 bit)
    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
        typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
        typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
        typenum == NPY_FLOAT ? CV_32F :
        typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("%s data type = %d is not supported", name, typenum);
        return false;
    }

    int ndims = PyArray_NDIM(o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", name);
        return false;
    }

    m = cv::Mat(ndims, size, type, PyArray_DATA(o), step);

    if( m.data )
    {
        m.u->refcount = *refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
        // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        cv::Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return true;
}

PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!(p->u->refcount) || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(&(p->u->refcount));
}

static PyObject * findSeparableMat(PyObject *self, PyObject *args) {
    PyObject *pyPts1, *pyPts2;
    int im_size_h_org, im_size_w_org;

    if (!PyArg_ParseTuple(args, "OOii", &pyPts1, &pyPts2, &im_size_h_org, &im_size_w_org))
        return NULL;

    Mat pts1 = fromNDArrayToMat(pyPts1);
    Mat pts2 = fromNDArrayToMat(pyPts2);
    
    // Mat findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
    //     float inlier_ratio = 0.4, int inlier_threshold = 3,
    //     double hough_rescale = DEFAULT_HOUGH_RESCALE, int num_matching_pts_to_use = 150, int pixel_res = 4, int min_hough_points = 4,
    //     int theta_res = 180, float max_distance_pts_line = 3, int top_line_retries = 2, int min_shared_points = 4);
    Mat res = findSeparableFundamentalMat(pts1, pts2, im_size_h_org, im_size_w_org);

    return fromMatToNDArray(res);
}
static PyObject * findFundamentalMatRegular(PyObject *self, PyObject *args) {
    PyObject *pyPts1, *pyPts2;

    if (!PyArg_ParseTuple(args, "OO", &pyPts1, &pyPts2))
        return NULL;

    Mat pts1 = fromNDArrayToMat(pyPts1);
    Mat pts2 = fromNDArrayToMat(pyPts2);
    
    // Mat findSeparableFundamentalMat(InputArray pts1, InputArray pts2, int im_size_h_org, int im_size_w_org,
    //     float inlier_ratio = 0.4, int inlier_threshold = 3,
    //     double hough_rescale = DEFAULT_HOUGH_RESCALE, int num_matching_pts_to_use = 150, int pixel_res = 4, int min_hough_points = 4,
    //     int theta_res = 180, float max_distance_pts_line = 3, int top_line_retries = 2, int min_shared_points = 4);
    Mat res = findFundamentalMatFullRansac(pts1, pts2);

    return fromMatToNDArray(res);
}

static PyMethodDef SepFMMethods[] = {
    {"findSeparableMat", findSeparableMat, METH_VARARGS, "Finds a Separable Fundamental matrix."},
	{"findFundamentalMatRegular", findFundamentalMatRegular, METH_VARARGS, "Finds a fundamental matrix, complete ransac."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sepFMModule = {
    PyModuleDef_HEAD_INIT,
    "sepfm",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    SepFMMethods
};

PyMODINIT_FUNC PyInit_sepfm(void) {
    return PyModule_Create(&sepFMModule);
}
