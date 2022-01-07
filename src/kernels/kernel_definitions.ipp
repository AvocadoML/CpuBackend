/*
 * kernel_definitions.ipp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */

avocado::backend::avStatus_t changeType(avocado::backend::avContextDescriptor_t context, avocado::backend::avMemoryDescriptor_t dst,
		avocado::backend::avDataType_t dstType, const avocado::backend::avMemoryDescriptor_t src, avocado::backend::avDataType_t srcType,
		avocado::backend::avSize_t elements);

avocado::backend::avStatus_t concatTensors(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const avocado::backend::avTensorDescriptor_t aDesc[],
		const avocado::backend::avMemoryDescriptor_t aMem[], int nbTensors);

avocado::backend::avStatus_t splitTensors(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc[],
		avocado::backend::avMemoryDescriptor_t cMem[], const avocado::backend::avTensorDescriptor_t aDesc,
		const avocado::backend::avMemoryDescriptor_t aMem, int nbTensors);

avocado::backend::avStatus_t transpose(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const avocado::backend::avTensorDescriptor_t aDesc,
		const avocado::backend::avMemoryDescriptor_t aMem, const int newDimOrder[]);

avocado::backend::avStatus_t scaleTensor(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const void *alpha);

avocado::backend::avStatus_t addScalarToTensor(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t cDesc,
		avocado::backend::avMemoryDescriptor_t cMem, const void *scalar);

avocado::backend::avStatus_t binaryOp(avocado::backend::avContextDescriptor_t context, avocado::backend::avBinaryOp_t operation, const void *alpha1,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *alpha2,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t unaryOp(avocado::backend::avContextDescriptor_t context, avocado::backend::avUnaryOp_t operation, const void *alpha,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t reduceTensor(avocado::backend::avContextDescriptor_t context, avocado::backend::avReduceOp_t operation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem);

avocado::backend::avStatus_t addBias(avocado::backend::avContextDescriptor_t context, const void *alpha3, const void *alpha1,
		const avocado::backend::avTensorDescriptor_t aDesc, const avocado::backend::avMemoryDescriptor_t aMem, const void *alpha2,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t cDesc, avocado::backend::avMemoryDescriptor_t cMem,
		avocado::backend::avActivationType_t activation);

avocado::backend::avStatus_t activationForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t activationBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t softmaxForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avSoftmaxMode_t mode,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t softmaxBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avSoftmaxMode_t mode,
		const void *alpha, const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t affineForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const avocado::backend::avTensorDescriptor_t wDesc, const avocado::backend::avMemoryDescriptor_t wMem,
		const avocado::backend::avTensorDescriptor_t bDesc, const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha,
		const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t batchNormInference(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t scaleBiasMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t biasMem, const avocado::backend::avMemoryDescriptor_t meanMem,
		const avocado::backend::avMemoryDescriptor_t varianceMem, double epsilon);

avocado::backend::avStatus_t batchNormForward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		const avocado::backend::avTensorDescriptor_t scaleBiasMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t biasMem, avocado::backend::avMemoryDescriptor_t meanMem,
		avocado::backend::avMemoryDescriptor_t varianceMem, double epsilon);

avocado::backend::avStatus_t batchNormBackward(avocado::backend::avContextDescriptor_t context, avocado::backend::avActivationType_t activation,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem,
		const avocado::backend::avTensorDescriptor_t yDesc, const avocado::backend::avMemoryDescriptor_t yMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, avocado::backend::avMemoryDescriptor_t dyMem,
		const avocado::backend::avTensorDescriptor_t scaleMeanVarDesc, const avocado::backend::avMemoryDescriptor_t scaleMem,
		const avocado::backend::avMemoryDescriptor_t meanMem, const avocado::backend::avMemoryDescriptor_t varianceMem, const void *alpha2,
		const void *beta2, avocado::backend::avMemoryDescriptor_t scaleUpdateMem, avocado::backend::avMemoryDescriptor_t biasUpdateMem,
		double epsilon);

avocado::backend::avStatus_t dropoutForward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avDropoutDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem,
		avocado::backend::avMemoryDescriptor_t states);

avocado::backend::avStatus_t dropoutBackward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avDropoutDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem,
		const avocado::backend::avTensorDescriptor_t states);

avocado::backend::avStatus_t poolingForward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avPoolingDescriptor_t config,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t yDesc, avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t poolingBackward(avocado::backend::avContextDescriptor_t context, const avocado::backend::avPoolingDescriptor_t config,
		const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc, const avocado::backend::avMemoryDescriptor_t xMem,
		const avocado::backend::avTensorDescriptor_t dyDesc, const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t dxDesc, avocado::backend::avMemoryDescriptor_t dxMem);

avocado::backend::avStatus_t im2row(avocado::backend::avContextDescriptor_t context, const avocado::backend::avConvolutionDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t filterDesc, const avocado::backend::avTensorDescriptor_t srcDesc,
		const avocado::backend::avMemoryDescriptor_t srcMem, const avocado::backend::avTensorDescriptor_t colDesc,
		avocado::backend::avMemoryDescriptor_t colMem);

avocado::backend::avStatus_t convolutionBiasActivationForward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha1, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const avocado::backend::avTensorDescriptor_t bDesc,
		const avocado::backend::avMemoryDescriptor_t bMem, const void *alpha2, const avocado::backend::avTensorDescriptor_t zDesc,
		const avocado::backend::avMemoryDescriptor_t zMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem, const avocado::backend::avActivationType_t activation,
		avocado::backend::avMemoryDescriptor_t workspace);

avocado::backend::avStatus_t convolutionForward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const void *beta, const avocado::backend::avTensorDescriptor_t yDesc,
		avocado::backend::avMemoryDescriptor_t yMem);

avocado::backend::avStatus_t convolutionBackward(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t dxDesc,
		avocado::backend::avMemoryDescriptor_t dxMem, const avocado::backend::avTensorDescriptor_t wDesc,
		const avocado::backend::avMemoryDescriptor_t wMem, const void *beta, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, avocado::backend::avMemoryDescriptor_t workspaceMem);

avocado::backend::avStatus_t convolutionUpdate(avocado::backend::avContextDescriptor_t context,
		const avocado::backend::avConvolutionDescriptor_t config, const void *alpha, const avocado::backend::avTensorDescriptor_t xDesc,
		const avocado::backend::avMemoryDescriptor_t xMem, const avocado::backend::avTensorDescriptor_t dyDesc,
		const avocado::backend::avMemoryDescriptor_t dyMem, const void *beta, const avocado::backend::avTensorDescriptor_t dwDesc,
		avocado::backend::avMemoryDescriptor_t dwMem);

avocado::backend::avStatus_t metricFunction(avocado::backend::avContextDescriptor_t context, avocado::backend::avMetricType_t metricType,
		const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, void *result);

avocado::backend::avStatus_t lossFunction(avocado::backend::avContextDescriptor_t context, avocado::backend::avLossType_t lossType,
		const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, void *result);

avocado::backend::avStatus_t lossGradient(avocado::backend::avContextDescriptor_t context, avocado::backend::avLossType_t lossType, const void *alpha,
		const avocado::backend::avTensorDescriptor_t outputDesc, const avocado::backend::avMemoryDescriptor_t outputMem,
		const avocado::backend::avTensorDescriptor_t targetDesc, const avocado::backend::avMemoryDescriptor_t targetMem, const void *beta,
		const avocado::backend::avTensorDescriptor_t gradientDesc, avocado::backend::avMemoryDescriptor_t gradientMem, bool isFused);

avocado::backend::avStatus_t optimizerLearn(avocado::backend::avContextDescriptor_t context, const avocado::backend::avOptimizerDescriptor_t config,
		const avocado::backend::avTensorDescriptor_t wDesc, avocado::backend::avMemoryDescriptor_t wMem,
		const avocado::backend::avTensorDescriptor_t dwDesc, const avocado::backend::avTensorDescriptor_t dwMem,
		avocado::backend::avMemoryDescriptor_t workspace);

avocado::backend::avStatus_t regularizerL2(avocado::backend::avContextDescriptor_t context, const avocado::backend::avTensorDescriptor_t gradientDesc,
		avocado::backend::avMemoryDescriptor_t gradientMem, const avocado::backend::avTensorDescriptor_t weightDesc,
		const avocado::backend::avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss);

