# HaraliCL
*Work in progress.*

A program that computes GLCM sliding window images using an accelerator that supports OpenCL. 

The tool is split in two parts - host-side and device-side.

The device-side code is written in OpenCL which gives one the freedom of deivce for execution - a CPU, GPU, FPGA, etc.

The host-side code is as of now implemented in Python and uses [`pyopencl`](https://github.com/inducer/pyopencl)
The current demo provided here also requires [`scikit-image`](https://github.com/scikit-image/scikit-image)

The program applies concepts from [HaraliCU](https://github.com/andrea-tango/HaraliCU) <sup><a href="#1">[1]</a></sup>.

## References
<span id="1">[1]</span> Rundo, L., Tangherloni, A., Galimberti, S., Cazzaniga, P., Woitek, R., Sala, E., Nobile, M. S., & Mauri, G. (2019). HaraliCU: GPU-Powered Haralick Feature Extraction on Medical Images Exploiting the Full Dynamics of Gray-Scale Levels. Lecture Notes in Computer Science, 304–318. https://doi.org/10.1007/978-3-030-25636-4_24
