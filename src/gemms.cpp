/*
 * gemms.cpp
 *
 *  Created on: Aug 8, 2021
 *      Author: Maciej Kozarzewski
 */

#include "kernel_definitions.hpp"
#include <Avocado/backend_descriptors.hpp>

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
#  include <cblas.h>
#endif

#include <complex>

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t cpu_gemm(const ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
		{
#if USE_BLIS
			trans_t op_A = is_transpose(aOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
			trans_t op_B = is_transpose(bOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
#endif
#if USE_OPENBLAS
			CBLAS_TRANSPOSE op_A = is_transpose(aOp) ? CblasTrans : CblasNoTrans;
			CBLAS_TRANSPOSE op_B = is_transpose(bOp) ? CblasTrans : CblasNoTrans;
#endif

			int M = is_transpose(aOp) ? aDesc.dimension(1) : aDesc.dimension(0);
			int N = is_transpose(bOp) ? bDesc.dimension(0) : bDesc.dimension(1);
			int K = is_transpose(aOp) ? aDesc.dimension(0) : aDesc.dimension(1);

			int LDA = aDesc.dimension(1);
			int LDB = bDesc.dimension(1);
			int LDC = cDesc.dimension(1);

			switch (cDesc.dtype())
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
					const float *A_ptr = aMem.data<float>();
					const float *B_ptr = bMem.data<float>();
					float *C_ptr = cMem.data<float>();

#if USE_BLIS
					bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<float*>(A_ptr), LDA, 1, const_cast<float*>(B_ptr), LDB, 1, &c_beta, C_ptr,
							LDC, 1);
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
					const double *A_ptr = aMem.data<double>();
					const double *B_ptr = bMem.data<double>();
					double *C_ptr = cMem.data<double>();
#if USE_BLIS
					bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<double*>(A_ptr), LDA, 1, const_cast<double*>(B_ptr), LDB, 1, &c_beta, C_ptr,
							LDC, 1);
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
					const std::complex<float> *A_ptr = aMem.data<std::complex<float>>();
					const std::complex<float> *B_ptr = bMem.data<std::complex<float>>();
					std::complex<float> *C_ptr = cMem.data<std::complex<float>>();
#if USE_BLIS
//					bli_cgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<float>*>(A_ptr), LDA, 1, const_cast<std::complex<float>*>(B_ptr),
//							LDB, 1, &c_beta, C_ptr, LDC, 1);
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
					const std::complex<double> *A_ptr = aMem.data<std::complex<double>>();
					const std::complex<double> *B_ptr = bMem.data<std::complex<double>>();
					std::complex<double> *C_ptr = cMem.data<std::complex<double>>();
#if USE_BLIS
//					bli_zgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<double>*>(A_ptr), LDA, 1,
//							const_cast<std::complex<double>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
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
		avStatus_t cpu_gemmBatched(const ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem)
		{
#if USE_BLIS
			trans_t op_A = is_transpose(aOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
			trans_t op_B = is_transpose(bOp) ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
#endif
#if USE_OPENBLAS
			CBLAS_TRANSPOSE op_A = is_transpose(aOp) ? CblasTrans : CblasNoTrans;
			CBLAS_TRANSPOSE op_B = is_transpose(bOp) ? CblasTrans : CblasNoTrans;
#endif
			int M = is_transpose(aOp) ? aDesc.dimension(2) : aDesc.dimension(1);
			int N = is_transpose(bOp) ? bDesc.dimension(1) : bDesc.dimension(2);
			int K = is_transpose(aOp) ? aDesc.dimension(1) : aDesc.dimension(2);

			int LDA = aDesc.dimension(2);
			int LDB = bDesc.dimension(2);
			int LDC = cDesc.dimension(2);

			switch (cDesc.dtype())
			{
//				case AVOCADO_DTYPE_BFLOAT16:
//				{
//					float c_alpha = getAlphaValue<float>(alpha);
//					float c_beta = getBetaValue<float>(beta);
//					for (int i = 0; i < aDesc).firstDim(); i++)
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
					for (int i = 0; i < aDesc.firstDim(); i++)
					{
						const float *A_ptr = aMem.data<float>() + i * M * K;
						const float *B_ptr = bMem.data<float>() + i * N * K;
						float *C_ptr = cMem.data<float>() + i * M * N;
#if USE_BLIS
						bli_sgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<float*>(A_ptr), LDA, 1, const_cast<float*>(B_ptr), LDB, 1, &c_beta, C_ptr,
								LDC, 1);
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
					for (int i = 0; i < aDesc.firstDim(); i++)
					{
						const double *A_ptr = aMem.data<double>() + i * M * K;
						const double *B_ptr = bMem.data<double>() + i * N * K;
						double *C_ptr = cMem.data<double>() + i * M * N;
#if USE_BLIS
						bli_dgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<double*>(A_ptr), LDA, 1, const_cast<double*>(B_ptr), LDB, 1, &c_beta,
								C_ptr, LDC, 1);
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
					for (int i = 0; i < aDesc.firstDim(); i++)
					{
						const std::complex<float> *A_ptr = aMem.data<std::complex<float>>() + i * M * K;
						const std::complex<float> *B_ptr = bMem.data<std::complex<float>>() + i * N * K;
						std::complex<float> *C_ptr = cMem.data<std::complex<float>>() + i * M * N;
#if USE_BLIS
//						bli_cgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<float>*>(A_ptr), LDA, 1,
//								const_cast<std::complex<float>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
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
					for (int i = 0; i < aDesc.firstDim(); i++)
					{
						const std::complex<double> *A_ptr = aMem.data<std::complex<double>>() + i * M * K;
						const std::complex<double> *B_ptr = bMem.data<std::complex<double>>() + i * N * K;
						std::complex<double> *C_ptr = cMem.data<std::complex<double>>() + i * M * N;
#if USE_BLIS
//						bli_zgemm(op_A, op_B, M, N, K, &c_alpha, const_cast<std::complex<double>*>(A_ptr), LDA, 1,
//								const_cast<std::complex<double>*>(B_ptr), LDB, 1, &c_beta, C_ptr, LDC, 1);
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

