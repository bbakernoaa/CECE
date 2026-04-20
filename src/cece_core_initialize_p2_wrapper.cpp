/**
 * @file cece_core_initialize_p2_wrapper.cpp
 * @brief Wrapper for cece_core_initialize_p2_impl providing C linkage for Fortran.
 */

extern "C" {

/**
 * @brief Implementation function (in cece_core_initialize_p2.cpp)
 */
void cece_core_initialize_p2_impl(void* data_ptr, int* nx, int* ny, int* nz, int* rc);

/**
 * @brief C-linkage wrapper called by the Fortran cap.
 *
 * @param data_ptr Pointer to CeceInternalData
 * @param nx Pointer to grid X dimension
 * @param ny Pointer to grid Y dimension
 * @param nz Pointer to grid Z dimension
 * @param rc Return code
 */
void cece_core_initialize_p2(void* data_ptr, int* nx, int* ny, int* nz, int* rc) {
    cece_core_initialize_p2_impl(data_ptr, nx, ny, nz, rc);
}

}  // extern "C"
