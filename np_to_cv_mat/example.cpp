#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


// -------------
// pure C++ code
// -------------

std::vector<double> length(const std::vector<double>& pos)
{
  size_t N = pos.size() / 2;

  std::vector<double> output(N*3);

  for ( size_t i = 0 ; i < N ; ++i ) {
    output[i*3+0] = pos[i*2+0];
    output[i*3+1] = pos[i*2+1];
    output[i*3+2] = std::pow(pos[i*2+0]*pos[i*2+1],.5);
  }

  return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array cv_mat_example(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> array)
{
  // Get buffer info
  py::buffer_info info = array.request();
  
  // Shape properties
  int nrows, ncols, indim, nchan=0;
  nrows = array.shape()[0];
  ncols = array.shape()[1];
  indim = info.ndim;
    
  // Convert array to CV::Mat and do opencv operations
  cv:: Mat m;
  if(indim == 3){
    nchan = 3;
    m = cv::Mat(nrows, ncols, CV_8UC3);
  } else if (indim == 2) {
    nchan = 1;
    m = cv::Mat(nrows, ncols, CV_8UC1);
  }
  
  memcpy(m.data, array.data(), nrows*ncols*nchan*sizeof(uint8_t));

  // std::vector<double> result = length(pos);
  ssize_t              ndim    = 3;
  std::vector<ssize_t> shape   = { nrows , ncols, nchan};
  std::vector<ssize_t> strides = { int(ncols*nchan*sizeof(uint8_t)), int(nchan*sizeof(uint8_t)), int(sizeof(uint8_t))};

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    m.data,                                 /* data as contiguous array  */
    sizeof(uint8_t),                          /* size of one scalar        */
    py::format_descriptor<uint8_t>::format(), /* data type                 */
    ndim,                                     /* number of dimensions      */
    shape,                                    /* shape of the matrix       */
    strides                                   /* strides for each axis     */
  ));
}

// wrap as Python module
PYBIND11_MODULE(example,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("cv_mat_example", &cv_mat_example, "Calculate the length of an array of vectors");
}
