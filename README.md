# amp_cs
Approximate message passing (AMP) for compressed sensing (CS) signal recovery.
* CS model: y = A * x + noise
* Goal: recover signal x given y and A

## state_evolution.ipynb

Compares the empirical average mean-squared error (MSE) of AMP decoding over many random decoding instances with its prediction from state evolution (SE).

We see that SE gives an accurate prediction of the per-iteration MSE of AMP decoders.

## AMP_mmse_denoiser.ipynb

Compares the mean squared error (MSE) performance of AMP decoders with soft-thresholding denoisers and minimum MSE (MMSE) denoisers, assuming the signal distribution is known to the decoder.

We see that 
1. For each AMP decoding instance, the MMSE denoiser achieves a lower final MSE compared to the soft-thresholding denoiser.
2. The MMSE denoiser achieves a larger "successful recovery" region in the undersampling ratio vs. sparsity ratio MSE phase diagram compared to the soft-thresholding denoiser.

Note: the MMSE denoiser is the conditional expectation operator.

## AMP_vs_IST.ipynb

Compares the distributions of the effective observation term s^t = x^t + A^T * z^t of the AMP algorithm and the iterative soft-thresholding (IST) algorithm, where t is the iteration index.

We see that for the AMP, the distribution is close to Gaussian, with mean close to the correct value (the signal value x), whereas for the IST, the distribution does not have the correct mean and is not Gaussian.

Note: the only difference between the AMP algorithm and the IST algorithm is that the AMP has additional "Onsager" correction term in the calculation of the residual vector at each iteration.

## AMP_cs_image_fast_transform.ipynb

Uses the AMP decoder for signal recovery in compressed sensing (CS) for images. Due to the high-dimensionality of images, we use a subsampled discrete cosine transform (DCT) sensing matrix and the DCT fast transform to reduce computational complexity.