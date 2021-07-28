cc = "g++"

# Opencv
cv_idir = `pkg-config --cflags opencv4`
cv_ldir = -L/usr/lib
cv_libs = `pkg-config --libs opencv4`

# Pybind11
py11_idir = -I/home/vj/.local/lib/python3.9/site-packages/pybind11/include
$(python3 -m pybind11 --includes)

# Python headers
py_idir = -I/usr/include/python3.9

# All
idirs = $(cv_idir) $(py11_idir) $(py_idir)
ldirs = $(cv_ldir)
libs  = $(cv_libs)

make:
	gcc -O3 -shared -std=gnu++11  $(idirs) $(ldirs) $(libs) example.cpp -o example.so -fPIC

clean:
	rm *.so
