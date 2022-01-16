/*
 * gemms.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CpuBackend/cpu_backend.h>
#include <backend_descriptors.hpp>

#if USE_BLIS
#  ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#    include "../../../extern/blis/blis.h"
#    pragma GCC diagnostic pop
#  else
#    include "../../../extern/blis/blis.h"
#  endif
#endif

#if USE_OPENBLAS
#    ifdef __linux__
#      include <openblas/include/cblas.h>
#    else
#      include <openblas/cblas.h>
#    endif
#endif

namespace avocado
{
	namespace backend
	{

		avStatus_t cpuGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if USE_BLIS
			trans_t op_A = is_transpose(aOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
			trans_t op_B = is_transpose(bOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
#endif
#if USE_OPENBLAS
			CBLAS_TRANSPOSE op_A = is_transpose(aOp) ? CblasTrans : CblasNoTrans;
			CBLAS_TRANSPOSE op_B = is_transpose(bOp) ? CblasTrans : CblasNoTrans;
#endif

			int M = is_transpose(aOp) ? getTensor(aDesc).dimension(1) : getTensor(aDesc).dimension(0);
			int N = is_transpose(bOp) ? getTensor(bDesc).dimension(0) : getTensor(bDesc).dimension(1);
			int K = is_transpose(aOp) ? getTensor(aDesc).dimension(0) : getTensor(aDesc).dimension(1);
			;

			int LDA = getTensor(aDesc).dimension(1);
			int LDB = getTensor(bDesc).dimension(1);
			int LDC = getTensor(cDesc).dimension(1);

			switch (getTensor(cDesc).dtype())
			{
//				case AVOCADO_DTYPE_BFLOAT16:
//				{
//					float c_alpha = getAlphaValue<float>(alpha);
//					float c_beta = getBetaValue<float>(beta);
//					const uint16_t *A_ptr = getPointer<uint16_t>(aMem);
//					const uint16_t *B_ptr = getPointer<uint16_t>(bMem);
//					float *C_ptr = getPointer<float>(cMem);
//#if USE_OPENBLAS
//					cblas_sbgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
//#endif
//					return AVOCADO_STATUS_SUCCESS;
//				}
				case AVOCADO_DTYPE_FLOAT32:
				{
					float c_alpha = getAlphaValue<float>(alpha);
					float c_beta = getBetaValue<float>(beta);
					const float *A_ptr = getPointer<float>(aMem);
					const float *B_ptr = getPointer<float>(bMem);
					float *C_ptr = getPointer<float>(cMem);
#if USE_BLIS
					bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<float*>(A_ptr), LDA, 1, const_cast<float*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
					cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
#endif
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double c_alpha = getAlphaValue<double>(alpha);
					double c_beta = getBetaValue<double>(beta);
					const double *A_ptr = getPointer<double>(aMem);
					const double *B_ptr = getPointer<double>(bMem);
					double *C_ptr = getPointer<double>(cMem);
#if USE_BLIS
					bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<double*>(A_ptr), LDA, 1, const_cast<double*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
					cblas_dgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
#endif
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_COMPLEX32:
				{
					std::complex<float> c_alpha = getAlphaValue<std::complex<float>>(alpha);
					std::complex<float> c_beta = getBetaValue<std::complex<float>>(beta);
					const std::complex<float> *A_ptr = getPointer<std::complex<float>>(aMem);
					const std::complex<float> *B_ptr = getPointer<std::complex<float>>(bMem);
					std::complex<float> *C_ptr = getPointer<std::complex<float>>(cMem);
#if USE_BLIS
					bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<float>*>(A_ptr), LDA, 1, const_cast<std::complex<float>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
					cblas_cgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, &c_alpha, A_ptr, LDA, B_ptr, LDB, &c_beta, C_ptr, LDC);
#endif
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_COMPLEX64:
				{
					std::complex<double> c_alpha = getAlphaValue<std::complex<double>>(alpha);
					std::complex<double> c_beta = getBetaValue<std::complex<double>>(beta);
					const std::complex<double> *A_ptr = getPointer<std::complex<double>>(aMem);
					const std::complex<double> *B_ptr = getPointer<std::complex<double>>(bMem);
					std::complex<double> *C_ptr = getPointer<std::complex<double>>(cMem);
#if USE_BLIS
					bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<double>*>(A_ptr), LDA, 1, const_cast<std::complex<double>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
					cblas_zgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, &c_alpha, A_ptr, LDA, B_ptr, LDB, &c_beta, C_ptr, LDC);
#endif
					return AVOCADO_STATUS_SUCCESS;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
		avStatus_t cpuGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
#if USE_BLIS
			trans_t op_A = is_transpose(aOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
			trans_t op_B = is_transpose(bOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
#endif
#if USE_OPENBLAS
			CBLAS_TRANSPOSE op_A = is_transpose(aOp) ? CblasTrans : CblasNoTrans;
			CBLAS_TRANSPOSE op_B = is_transpose(bOp) ? CblasTrans : CblasNoTrans;
#endif
			int M = is_transpose(aOp) ? getTensor(aDesc).dimension(2) : getTensor(aDesc).dimension(1);
			int N = is_transpose(bOp) ? getTensor(bDesc).dimension(1) : getTensor(bDesc).dimension(2);
			int K = is_transpose(aOp) ? getTensor(aDesc).dimension(1) : getTensor(aDesc).dimension(2);

			int LDA = getTensor(aDesc).dimension(2);
			int LDB = getTensor(bDesc).dimension(2);
			int LDC = getTensor(cDesc).dimension(2);

			switch (getTensor(cDesc).dtype())
			{
//				case AVOCADO_DTYPE_BFLOAT16:
//				{
//					float c_alpha = getAlphaValue<float>(alpha);
//					float c_beta = getBetaValue<float>(beta);
//					for (int i = 0; i < getTensor(aDesc).firstDim(); i++)
//					{
//						const uint16_t *A_ptr = getPointer<uint16_t>(aMem) + i * M * K;
//						const uint16_t *B_ptr = getPointer<uint16_t>(bMem) + i * N * K;
//						float *C_ptr = getPointer<float>(cMem) + i * M * N;
//#if USE_OPENBLAS
//						cblas_sbgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
//#endif
//					}
//					return AVOCADO_STATUS_SUCCESS;
//				}
				case AVOCADO_DTYPE_FLOAT32:
				{
					float c_alpha = getAlphaValue<float>(alpha);
					float c_beta = getBetaValue<float>(beta);
					for (int i = 0; i < getTensor(aDesc).firstDim(); i++)
					{
						const float *A_ptr = getPointer<float>(aMem) + i * M * K;
						const float *B_ptr = getPointer<float>(bMem) + i * N * K;
						float *C_ptr = getPointer<float>(cMem) + i * M * N;
#if USE_BLIS
						bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<float*>(A_ptr), LDA, 1, const_cast<float*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
						cblas_sgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
#endif
					}
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double c_alpha = getAlphaValue<double>(alpha);
					double c_beta = getBetaValue<double>(beta);
					for (int i = 0; i < getTensor(aDesc).firstDim(); i++)
					{
						const double *A_ptr = getPointer<double>(aMem) + i * M * K;
						const double *B_ptr = getPointer<double>(bMem) + i * N * K;
						double *C_ptr = getPointer<double>(cMem) + i * M * N;
#if USE_BLIS
						bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<double*>(A_ptr), LDA, 1, const_cast<double*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
						cblas_dgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, c_alpha, A_ptr, LDA, B_ptr, LDB, c_beta, C_ptr, LDC);
#endif
					}
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_COMPLEX32:
				{
					std::complex<float> c_alpha = getAlphaValue<std::complex<float>>(alpha);
					std::complex<float> c_beta = getBetaValue<std::complex<float>>(beta);
					for (int i = 0; i < getTensor(aDesc).firstDim(); i++)
					{
						const std::complex<float> *A_ptr = getPointer<std::complex<float>>(aMem) + i * M * K;
						const std::complex<float> *B_ptr = getPointer<std::complex<float>>(bMem) + i * N * K;
						std::complex<float> *C_ptr = getPointer<std::complex<float>>(cMem) + i * M * N;
#if USE_BLIS
						bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<float>*>(A_ptr), LDA, 1, const_cast<std::complex<float>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
						cblas_cgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, &c_alpha, A_ptr, LDA, B_ptr, LDB, &c_beta, C_ptr, LDC);
#endif
					}
					return AVOCADO_STATUS_SUCCESS;
				}
				case AVOCADO_DTYPE_COMPLEX64:
				{
					std::complex<double> c_alpha = getAlphaValue<std::complex<double>>(alpha);
					std::complex<double> c_beta = getBetaValue<std::complex<double>>(beta);
					for (int i = 0; i < getTensor(aDesc).firstDim(); i++)
					{
						const std::complex<double> *A_ptr = getPointer<std::complex<double>>(aMem) + i * M * K;
						const std::complex<double> *B_ptr = getPointer<std::complex<double>>(bMem) + i * N * K;
						std::complex<double> *C_ptr = getPointer<std::complex<double>>(cMem) + i * M * N;
#if USE_BLIS
						bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<double>*>(A_ptr), LDA, 1, const_cast<std::complex<double>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
#endif
#if USE_OPENBLAS
						cblas_zgemm(CBLAS_ORDER::CblasRowMajor, op_A, op_B, M, N, K, &c_alpha, A_ptr, LDA, B_ptr, LDB, &c_beta, C_ptr, LDC);
#endif
					}
					return AVOCADO_STATUS_SUCCESS;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

