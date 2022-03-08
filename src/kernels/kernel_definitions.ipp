/*
 * kernel_definitions.ipp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

using avocado::backend::avStatus_t;
using avocado::backend::avDataType_t;
using avocado::backend::av_int64;
using avocado::backend::avActivationType_t;
using avocado::backend::avBinaryOp_t;
using avocado::backend::avUnaryOp_t;
using avocado::backend::avReduceOp_t;
using avocado::backend::avSoftmaxMode_t;
using avocado::backend::avMetricType_t;
using avocado::backend::avLossType_t;
using avocado::backend::avGemmOperation_t;
using avocado::backend::cpu::ContextDescriptor;
using avocado::backend::cpu::TensorDescriptor;
using avocado::backend::cpu::MemoryDescriptor;
using avocado::backend::cpu::ConvolutionDescriptor;
using avocado::backend::cpu::PoolingDescriptor;
using avocado::backend::cpu::DropoutDescriptor;
using avocado::backend::cpu::OptimizerDescriptor;

/*
 * Tensor operations.
 */
avStatus_t cpu_changeTypeHost(const ContextDescriptor &context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
		av_int64 elements);
avStatus_t cpu_changeType(const ContextDescriptor &context, MemoryDescriptor &dst, avDataType_t dstType, const MemoryDescriptor &src,
		avDataType_t srcType, av_int64 elements);
avStatus_t cpu_concatTensors(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem,
		const std::vector<const TensorDescriptor*> &aDesc, const std::vector<const MemoryDescriptor*> &aMem);
avStatus_t cpu_splitTensors(const ContextDescriptor &context, const std::vector<const TensorDescriptor*> &cDesc, std::vector<MemoryDescriptor*> &cMem,
		const TensorDescriptor &aDesc, const MemoryDescriptor &aMem);
avStatus_t cpu_transpose(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem, const TensorDescriptor &aDesc,
		const MemoryDescriptor &aMem, const int newDimOrder[]);
avStatus_t cpu_scaleTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const void *alpha,
		const TensorDescriptor &cDesc, MemoryDescriptor &cMem);
avStatus_t cpu_addScalarToTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const void *scalar,
		const TensorDescriptor &cDesc, MemoryDescriptor &cMem);
avStatus_t cpu_addBias(const ContextDescriptor &context, const void *alpha1, const void *alpha2, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const TensorDescriptor &yDesc,
		MemoryDescriptor &yMem, const void *beta1, const void *beta2, const void *beta3, const MemoryDescriptor &zMem, avActivationType_t activation);
avStatus_t cpu_binaryOp(const ContextDescriptor &context, avBinaryOp_t operation, const void *alpha1, const TensorDescriptor &aDesc,
		const MemoryDescriptor &aMem, const void *alpha2, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *beta,
		const TensorDescriptor &cDesc, MemoryDescriptor &cMem);
avStatus_t cpu_unaryOp(const ContextDescriptor &context, avUnaryOp_t operation, const void *alpha, const TensorDescriptor &aDesc,
		const MemoryDescriptor &aMem, const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);
avStatus_t cpu_reduceTensor(const ContextDescriptor &context, avReduceOp_t operation, const void *alpha, const TensorDescriptor &aDesc,
		const MemoryDescriptor &aMem, const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

/*
 * Activation functions.
 */
avStatus_t cpu_activationForward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);
avStatus_t cpu_activationBackward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha, const TensorDescriptor &yDesc,
		const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const void *beta, const TensorDescriptor &dxDesc,
		MemoryDescriptor &dxMem);
avStatus_t cpu_softmaxForward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);
avStatus_t cpu_softmaxBackward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha, const TensorDescriptor &yDesc,
		const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const void *beta, const TensorDescriptor &dxDesc,
		MemoryDescriptor &dxMem);

/*
 * Batch normalization and affine transform.
 */
avStatus_t cpu_affineForward(const ContextDescriptor &context, avActivationType_t activation, const TensorDescriptor &wDesc,
		const MemoryDescriptor &wMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);
avStatus_t cpu_batchNormInference(const ContextDescriptor &context, avActivationType_t activation, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem,
		const TensorDescriptor &scaleBiasMeanVarDesc, const MemoryDescriptor &scaleMem, const MemoryDescriptor &biasMem,
		const MemoryDescriptor &meanMem, const MemoryDescriptor &varianceMem, double epsilon);
avStatus_t cpu_batchNormForward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem,
		const TensorDescriptor &scaleBiasMeanVarDesc, const MemoryDescriptor &scaleMem, const MemoryDescriptor &biasMem, MemoryDescriptor &meanMem,
		MemoryDescriptor &varianceMem, double epsilon);
avStatus_t cpu_batchNormBackward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const TensorDescriptor &yDesc, const MemoryDescriptor &yMem, const void *beta, const TensorDescriptor &dxDesc,
		MemoryDescriptor &dxMem, const TensorDescriptor &dyDesc, MemoryDescriptor &dyMem, const TensorDescriptor &scaleMeanVarDesc,
		const MemoryDescriptor &scaleMem, const MemoryDescriptor &meanMem, const MemoryDescriptor &varianceMem, const void *alpha2, const void *beta2,
		MemoryDescriptor &scaleUpdateMem, MemoryDescriptor &biasUpdateMem, double epsilon);

/*
 * Dropout/
 */
avStatus_t cpu_dropoutForward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &states);
avStatus_t cpu_dropoutBackward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &dyDesc,
		const MemoryDescriptor &dyMem, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const MemoryDescriptor &states);

/*
 * Pooling.
 */
avStatus_t cpu_poolingForward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);
avStatus_t cpu_poolingBackward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha, const TensorDescriptor &xDesc,
		const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const void *beta, const TensorDescriptor &dxDesc,
		MemoryDescriptor &dxMem);

/*
 * Convolutions.
 */
avStatus_t cpu_im2row(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &filterDesc,
		const TensorDescriptor &srcDesc, const MemoryDescriptor &srcMem, const TensorDescriptor &rowDesc, MemoryDescriptor &rowMem);
avStatus_t cpu_convolutionImplicitGemmForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
		const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
		const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem,
		const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation);
avStatus_t cpu_convolutionWinogradFusedForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
		const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
		const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem,
		const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation);
avStatus_t cpu_winogradWeightTransform(const ContextDescriptor &context, const ConvolutionDescriptor &config, int transformSize,
		const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem);
avStatus_t cpu_winogradInputTransform(const ContextDescriptor &context, const ConvolutionDescriptor &config, int transformSize,
		const TensorDescriptor &wDesc, const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &matricesDesc,
		MemoryDescriptor &matricesMem);
avStatus_t cpu_winogradOutputTransform(const ContextDescriptor &context, const ConvolutionDescriptor &config, int transformSize,
		const TensorDescriptor &wDesc, const void *alpha1, const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem,
		const TensorDescriptor &yDesc, MemoryDescriptor &yMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2,
		const TensorDescriptor &zDesc, const MemoryDescriptor &zMem, const void *beta, avActivationType_t activation);
avStatus_t cpu_winogradGradientTransform(const ContextDescriptor &context, const ConvolutionDescriptor &config, int transformSize,
		const TensorDescriptor &wDesc, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const TensorDescriptor &matricesDesc,
		MemoryDescriptor &matricesMem);
avStatus_t cpu_winogradUpdateTransform(const ContextDescriptor &context, const ConvolutionDescriptor &config, int transformSize, const void *alpha,
		const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem, const void *beta, const TensorDescriptor &dwDesc,
		MemoryDescriptor &dwMem);

//avStatus_t cpu_convolutionBiasActivationForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha1,
//		const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
//		const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem,
//		const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation, MemoryDescriptor &workspaceMem);
//
//avStatus_t cpu_convolutionForward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
//		const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
//		const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem);
//
//avStatus_t cpu_convolutionBackward(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
//		const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *beta,
//		const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem);
//
//avStatus_t cpu_convolutionUpdate(const ContextDescriptor &context, const ConvolutionDescriptor &config, const void *alpha,
//		const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const void *beta,
//		const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem);

/*
 * Training operations.
 */
avStatus_t cpu_metricFunction(const ContextDescriptor &context, avMetricType_t metricType, const TensorDescriptor &outputDesc,
		const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result);
avStatus_t cpu_lossFunction(const ContextDescriptor &context, avLossType_t lossType, const TensorDescriptor &outputDesc,
		const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result);
avStatus_t cpu_lossGradient(const ContextDescriptor &context, avLossType_t lossType, const void *alpha, const TensorDescriptor &outputDesc,
		const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, const void *beta,
		const TensorDescriptor &gradientDesc, MemoryDescriptor &gradientMem, bool isFused);
avStatus_t cpu_optimizerLearn(const ContextDescriptor &context, const OptimizerDescriptor &config, const void *alpha, const TensorDescriptor &dwDesc,
		const MemoryDescriptor &dwMem, const void *beta, const TensorDescriptor &wDesc, MemoryDescriptor &wMem, MemoryDescriptor &workspaceMem);
avStatus_t cpu_regularizerL2(const ContextDescriptor &context, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, const TensorDescriptor &wDesc,
		const MemoryDescriptor &wMem, const void *coefficient, const void *offset, void *loss);

/*
 * Additional kernels that are not exposed in the main API
 */

