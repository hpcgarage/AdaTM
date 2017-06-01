AdaTM
------

AdaTM is Adaptive Tensor Memoization algorithm for CP decomposition and its matricized tensor times Khatri-Rao product (MTTKRP) operation. AdaTM is to speedup a higher-order sparse MTTKRP sequence by removing redundant computations within it.
The code is built upon SPLATT library, as a patch for it.


## Supported operations:

* Sparse matricized tensor times Khatri-Rao product (MTTKRP)
* Sparse CANDECOMP/PARAFAC decomposition

## Build requirements:

- [CMake](https://cmake.org) (>v2.6.0)

- [SPLATT](http://shaden.io/splatt.html)


## Build:

1. Download [SPLATT](http://shaden.io/splatt.html) library.

2. Copy all contents in AdaTM to SPLATT source code using `cp * [SPLATT DIR]'. Then the usage is the same with SPLATT.

3. Type `./configure --adatm`

4. `make'; `make install'.

5. Run MTTKRP: `splatt bench [TENSOR] -a adatm'

6. Run CPD: `splatt cpd [TENSOR]'


<br/>The algorithms and details are described in the following publications.
## Publication
* **Model-Driven Sparse CP Decomposition for Higher-Order Tensors**. Jiajia Li, Jee Choi, Ioakeim Perros, Jimeng Sun, Richard Vuduc. 31st IEEE International Parallel & Distributed Processing Symposium (IPDPS). 2017. [[pdf]](http://fruitfly1026.github.io/static/files/ipdps17-jli.pdf)


## Contributiors

* Jiajia Li (Contact: jiajiali@gatech.edu)