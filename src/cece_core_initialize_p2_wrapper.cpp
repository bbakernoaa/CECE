/**
 * @file cece_core_initialize_p2_wrapper.cpp
 * @brief Wrapper for cece_core_initialize_p2 that matches the Fortran call signature.
 */

extern "C" {

/**
 * @brief Core initialize phase 2 function (implemented in cece_core_initialize_p2.cpp)
 */
void cece_core_initialize_p2_impl(void* data_ptr, int* nx, int* ny, int* nz, int* rc);

/**
 * @brief 5-parameter wrapper for cece_core_initialize_p2 (called by Fortran cap).
 *
 * This wrapper matches the signature expected by the Fortran cap.
 */
void cece_core_initialize_p2(void* data_ptr, int* nx, int* ny, int* nz, int* rc) {
    cece_core_initialize_p2_impl(data_ptr, nx, ny, nz, rc);
}

}  // extern "C"
