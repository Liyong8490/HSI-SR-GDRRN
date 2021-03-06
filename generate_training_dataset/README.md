# HSI-SR-GDRRN: Training Dataset Generation
Single Hyperspectral Image Super-Resolution with Grouped Deep Recursive Residual Network

Some functions are cited from [1] http://see.xidian.edu.cn/faculty/wsdong/

# Dependencies

50 hyperspectral images of Harvard dataset[2] from http://vision.seas.harvard.edu/hyperspec/explore.html

# Generating dataset

- To generate training dataset: run 'generate_fusiondata.m' using matlab.

- To generate testing dataset: run 'generate_test_data.m' using matlab. You must manually remove images using to generate training dataset.

# references
[1] Weisheng Dong, Fazuo Fu, Guangming Shi, and Xun Cao, Jinjian Wu, Guangyu Li, and Xin Li, “Hyperspectral Image Super-Resolution via Non-Negative Structured Sparse Representation”, IEEE Trans. On Image Processing, vol. 25, no. 5, pp. 2337-2352, May 2016. 

[2] Ayan Chakrabarti and Todd Zickler, "Statistics of Real-World Hyperspectral Images," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.