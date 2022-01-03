/*
 * kernel_definitions.ipp
 *
 *  Created on: Nov 23, 2021
 *      Author: Maciej Kozarzewski
 */
#include <avocado/backend/backend_defs.h>
using namespace avocado::backend;

avStatus_t changeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
		avDataType_t srcType, avSize_t elements);

avStatus_t concatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
		const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors);

avStatus_t splitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
		const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors);

avStatus_t transpose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
		const avMemoryDescriptor_t aMem, const int newDimOrder[]);

avStatus_t scaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha);

avStatus_t addScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *scalar);

avStatus_t binaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
		const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
		const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

avStatus_t unaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
		const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

avStatus_t reduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
		const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

avStatus_t addBias(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
		const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
		const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation);

avStatus_t activationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

avStatus_t activationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t yDesc,
		const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
		const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

avStatus_t softmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

avStatus_t softmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
		const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
		const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

avStatus_t affineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
		const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
		const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
		avMemoryDescriptor_t yMem);

avStatus_t batchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
		const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
		const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

avStatus_t batchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
		const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
		avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon);

avStatus_t batchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const void *beta,
		const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc, avMemoryDescriptor_t dyMem,
		const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t meanMem,
		const avMemoryDescriptor_t varianceMem, double epsilon);

avStatus_t batchNormUpdate(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem,
		const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t scaleBiasDesc,
		avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, const avMemoryDescriptor_t meanMem,
		const avMemoryDescriptor_t varianceMem, double epsilon);

avStatus_t dropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states);

avStatus_t dropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
		const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states);

avStatus_t poolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

avStatus_t poolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
		const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
		const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

avStatus_t im2row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
		const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc, avMemoryDescriptor_t colMem);

avStatus_t getConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
		const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, avSize_t *result);

avStatus_t convolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
		const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
		const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
		const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
		const avActivationType_t activation, avMemoryDescriptor_t workspace);

avStatus_t convolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
		const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
		const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

avStatus_t convolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
		const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
		const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem);

avStatus_t metricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
		const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result);

avStatus_t lossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
		const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result);

avStatus_t lossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
		const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
		const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused);

avStatus_t getOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, avSize_t *result);

avStatus_t optimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const avTensorDescriptor_t wDesc,
		avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, avMemoryDescriptor_t workspace);

avStatus_t regularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
		const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss);

