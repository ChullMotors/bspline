# bspline
Differentiable tensor-product bspline interpolation in python

Inspired by [1]


The implementation, unlike other python spline libraries out there [2-4], is both PyTorch autograd-native and provides support for nonuniform grids in 1D, 2D, and 3D (currently implemented for 3D). It could be useful to anyone working with differentiable physics, ML practitioners who need a spline inside PyTorch, or engineers who like bspline-fortran but are trying to avoid Fortran. 

[1] bspline-fortran module. https://github.com/jacobwilliams/bspline-fortran 

[2] scipy.interpolate. https://docs.scipy.org/doc/scipy/reference/interpolate.htm l 

[3] torch.nn.functional.interpolate https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html 

[4] torch cubic spline. https://github.com/patrick-kidger/torchcubicspline 