# MeshFlow-Online-Video-Stabilization
Implementation of MeshFlow: Minimum latency online video stabilization based on 2016 paper written by Liu et al.

There is still much to do, but the basis of algorithm is working.

Problems that came up during development:
1) The current algorithm is having a big troubles with handling spatial high-frequency jitter composed with rolling shutter sensor camera distortions. (e.g. videos with camera fixed to the vehicle front mirror) We've tried a lot of different configurations of Optical Flow, corner extraction an MeshFlow (e.g. motion propagation radius, etc), but didn't meet satisfactory results. So you feel free to issue this problems :)
2) Well, the quality of C++ code and the design here is somewhat disasterous:) We will try to improve it with time, but if you suddenly feel the need to try this code, then feel free to do what you want

Based on the [Python](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization) implementation of MeshFlow algorithm. Thanks to @sudheerachary. 
