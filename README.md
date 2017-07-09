AdaTM
------

AdaTM is Adaptive Tensor Memoization algorithm for CP decomposition and its matricized tensor times Khatri-Rao product (MTTKRP) operation. AdaTM is to speedup a higher-order sparse MTTKRP sequence by removing redundant computations within it, and includes a model-driven framework to predict the optimal performance.


## Supported operations:

* Sparse matricized tensor times Khatri-Rao product (MTTKRP)
* Sparse CANDECOMP/PARAFAC decomposition

## Build requirements:

- [CMake](https://cmake.org) (>v2.6.0)

- SPLATT v1.1.1: [website](http://shaden.io/splatt.html) or [direct downloadable file](http://shaden.io/releases/splatt/splatt-1.1.1.tgz)


## Build:

1. Download [SPLATT](http://shaden.io/splatt.html) library.

2. Copy all contents in AdaTM to SPLATT source code using `cp * [SPLATT DIR]'. Then the usage is the same with SPLATT.

3. Type `./configure --adatm`

4. `make'; `make install'.

5. Run MTTKRP: `splatt bench [TENSOR] -a adatm'

6. Run CPD: `splatt cpd [TENSOR]'


## Limitation

AdaTM is only implemented for CSF format, not including COO format yet.
The code is closely built upon SPLATT library by using its CSF format and single MTTKRP implementation. It is released as a patch of SPLATT. We're working together to get more SPLATT functions exposed and will release more improved version in the future.


<br/>The algorithms and details are described in the following publications.
## Publication
* **Model-Driven Sparse CP Decomposition for Higher-Order Tensors**. Jiajia Li, Jee Choi, Ioakeim Perros, Jimeng Sun, Richard Vuduc. 31st IEEE International Parallel & Distributed Processing Symposium (IPDPS). 2017. [[pdf]](http://fruitfly1026.github.io/static/files/ipdps17-jli.pdf) [[slides]](http://fruitfly1026.github.io/static/files/ipdps17-jli-slides.pdf)


## Contributiors

* Jiajia Li (Contact: jiajiali@gatech.edu)

## License
AdaTM is released under the MIT License, you're free to redistribute it and/or modify it. Please see the 'LICENSE' file for details.
