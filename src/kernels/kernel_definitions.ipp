/*
 * kernel_definitions.ipp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

avocado::backend::avStatus_t cpu_changeType(avocado::backend::avContextDescriptor_t context, avocado::backend::avMemoryDescriptor_t dst,
		avocado::backend::avDataType_t dstType, const avocado::backend::avMemoryDescriptor_t src, avocado::backend::avDataType_t srcType,
		avocado::backend::avSize_t elements);

avocado::backend::avStatus_t cpu_changeTypeHost(avocado::backend::avContextDescriptor_t context, void *dst, avocado::backend::avDataType_t dstType,
		const void *src, avocado::backend::avDataType_t srcType, avocado::backend::avSize_t elements);

avocado::backend::avStatus_t cpu_concatTensors(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const avocado::backend::avTensorDescriptor_t aDesc[],
		const avocado::backend::avMemoryDescriptor_t aMem[], int nbTensors);

avocado::backend::avStatus_t cpu_splitTensors(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc[],
		avocado::backend::avMemoryDescriptor_t cMem[], const avocado::backend::avTensorDescriptor_t aDesc,
		const avocado::backend::avMemoryDescriptor_t aMem, int nbTensors);

avocado::backend::avStatus_t cpu_transpose(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const avocado::backend::avTensorDescriptor_t aDesc,
		const avocado::backend::avMemoryDescriptor_t aMem, const int newDimOrder[]);

avocado::backend::avStatus_t cpu_scaleTensor(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t aDesc,
		const avocado::backend::avMemoryDescriptor_t aMem, const void *alpha, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t cpu_addScalarToTensor(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *scalar,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t cpu_binaryOp(avocado::backend::avContextDescriptor_t context, avocado::backend::avBinaryOp_t operation,
		const void *alpha1, const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *alpha2,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t cpu_unaryOp(avocado::backend::avContextDescriptor_t context, avocado::backend::avUnaryOp_t operation, const void *alpha,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t cpu_reduceTensor(avocado::backend::avContextDescriptor_t context, avocado::backend::avReduceOp_t operation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t cpu_addBias(avocado::backend::avContextDescriptor_t context, const void *alpha3, const void *alpha1,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *alpha2,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem,
		avocado::backend::avActivationType_t activation);

avocado::backend::avStatus_t cpu_activationForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t cpu_activationBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t cpu_softmaxForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avSoftmaxMode_t mode,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t cpu_softmaxBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avSoftmaxMode_t mode,
		const void *alpha, const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t cpu_affineForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const avocado::backend::avTensorDescriptor_t wDesc, const avocado::backend::avMemoryDescriptor_t wMem,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha,
		const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t cpu_batchNormInference(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t scaleBiasMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t biasMem, const avocado::backend::avMemoryDescriptor_t meanMem,
		const avocado::backend::avMemoryDescriptor_t varianceMem, double epsilon);

avocado::backend::avStatus_t cpu_batchNormForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t scaleBiasMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t biasMem, avocado::backend::avMemoryDescriptor_t meanMem,
		avocado::backend::avMemoryDescriptor_t varianceMem, double epsilon);

avocado::backend::avStatus_t cpu_batchNormBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem,
		const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, avocado::backend::avMemoryDescriptor_t dyMem,
		const avocado::backend::avTensorDescriptor_t scaleMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t meanMem, const avocado::backend::avMemoryDescriptor_t varianceMem, const void *alpha2,
		const void *beta2, avocado::backend::avMemoryDescriptor_t scaleUpdateMem, avocado::backend::avMemoryDescriptor_t biasUpdateMem,
		double epsilon);

avocado::backend::avStatus_t cpu_dropoutForward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avDropoutDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		avocado::backend::avMemoryDescriptor_t states);

avocado::backend::avStatus_t cpu_dropoutBackward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avDropoutDescriptor_t config, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, const avocado::backend::avTensorDescriptor_t dxDesc,
		avocado::backend::avMemoryDescriptor_t dxMem, const avocado::backend::avTensorDescriptor_t states);

avocado::backend::avStatus_t cpu_poolingForward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avPoolingDescriptor_t config,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t cpu_poolingBackward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avPoolingDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta, const avocado::backend::avTensorDescriptor_t dxDesc,
		avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t cpu_im2row(avocado::backend::avContextDescriptor_t context, const avocado::backend::avConvolutionDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t filterDesc, const avocado::backend::avTensorDescriptor_t srcDesc,
		const avocado::backend::avMemoryDescriptor_t srcMem, const avocado::backend::avTensorDescriptor_t colDesc,
		avocado::backend::avMemoryDescriptor_t colMem);

avocado::backend::avStatus_t cpu_convolutionBiasActivationForward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha1, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t bDesc,
		const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha2, const avocado::backend::avTensorDescriptor_t zDesc,
		const avocado::backend::avMemoryDescriptor_t zMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem, const avocado::backend::avActivationType_t activation,
		avocado::backend::avMemoryDescriptor_t workspace);

avocado::backend::avStatus_t cpu_convolutionForward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t cpu_convolutionBackward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t dxDesc,
		avocado::backend::avMemoryDescriptor_t dxMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const void *beta, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, avocado::backend::avMemoryDescriptor_t workspaceMem);

avocado::backend::avStatus_t cpu_convolutionUpdate(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta, const avocado::backend::avTensorDescriptor_t dwDesc,
		avocado::backend::avMemoryDescriptor_t dwMem);

avocado::backend::avStatus_t cpu_metricFunction(avocado::backend::avContextDescriptor_t context, avocado::backend::avMetricType_t metricType,
		const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, void *result);

avocado::backend::avStatus_t cpu_lossFunction(avocado::backend::avContextDescriptor_t context, avocado::backend::avLossType_t lossType,
		const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, void *result);

avocado::backend::avStatus_t cpu_lossGradient(avocado::backend::avContextDescriptor_t context, avocado::backend::avLossType_t lossType,
		const void *alpha, const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t gradientDesc, avocado::backend::avMemoryDescriptor_t gradientMem, bool isFused);

avocado::backend::avStatus_t cpu_optimizerLearn(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avOptimizerDescriptor_t config, const avocado::backend::avTensorDescriptor_t wDesc,
		avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t dwDesc,
		const avocado::backend::avTensorDescriptor_t dwMem, avocado::backend::avMemoryDescriptor_t workspace);

avocado::backend::avStatus_t cpu_regularizerL2(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avTensorDescriptor_t gradientDesc, avocado::backend::avMemoryDescriptor_t gradientMem,
		const avocado::backend::avTensorDescriptor_t weightDesc, const avocado::backend::avMemoryDescriptor_t weightMem, const void *coefficient,
		const void *offset, void *loss);

/*
 * Additional kernels that are not exposed in the main API
 */

avocado::backend::avStatus_t cpu_convolution2dImplicitGemm(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha1, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t bDesc,
		const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha2, const avocado::backend::avTensorDescriptor_t zDesc,
		const avocado::backend::avMemoryDescriptor_t zMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem, const avocado::backend::avActivationType_t activation);

avocado::backend::avStatus_t cpu_winogradWeightTransform(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t matricesDesc,
		avocado::backend::avMemoryDescriptor_t matricesMem);

avocado::backend::avStatus_t cpu_winogradInputTransform(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t matricesDesc,
		avocado::backend::avMemoryDescriptor_t matricesMem, const avocado::backend::avTensorDescriptor_t wDesc);

avocado::backend::avStatus_t cpu_winogradOutputTransform(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha1, const avocado::backend::avTensorDescriptor_t matricesDesc,
		const avocado::backend::avMemoryDescriptor_t matricesMem, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem, const avocado::backend::avTensorDescriptor_t bDesc,
		const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha2, const avocado::backend::avTensorDescriptor_t zDesc,
		const avocado::backend::avMemoryDescriptor_t zMem, const void *beta, const avocado::backend::avActivationType_t activation,
		const avocado::backend::avTensorDescriptor_t wDesc);

avocado::backend::avStatus_t cpu_winogradGradientTransform(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, const avocado::backend::avTensorDescriptor_t matricesDesc,
		avocado::backend::avMemoryDescriptor_t matricesMem, const avocado::backend::avTensorDescriptor_t wDesc);

avocado::backend::avStatus_t cpu_winogradUpdateTransform(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t matricesDesc,
		const avocado::backend::avMemoryDescriptor_t matricesMem, const void *beta, const avocado::backend::avTensorDescriptor_t dwDesc,
		avocado::backend::avMemoryDescriptor_t dwMem);

avocado::backend::avStatus_t cpu_winogradFusedForward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha1, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t bDesc,
		const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha2, const avocado::backend::avTensorDescriptor_t zDesc,
		const avocado::backend::avMemoryDescriptor_t zMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem, const avocado::backend::avActivationType_t activation);
