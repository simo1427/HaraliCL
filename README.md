# HaraliCL
A program that computes GLCM sliding window images using an accelerator that supports OpenCL.

The tool is split in two parts - host-side and device-side.

The device-side code is written in OpenCL which gives one the freedom of deivce for execution - a CPU, GPU, FPGA, etc.

The host-side code is as of now implemented in Python and uses [`pyopencl`](https://github.com/inducer/pyopencl)
The current demo provided here also requires [`scikit-image`](https://github.com/scikit-image/scikit-image)
