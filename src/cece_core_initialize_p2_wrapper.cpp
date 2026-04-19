/**
 * @file cece_core_initialize_p2_wrapper.cpp
 * @brief Wrapper for cece_core_initialize_p2 that matches Fortran binding.
 */

extern "C" {

/**
 * @brief Core implementation (in cece_core_initialize_p2.cpp)
 */
void cece_core_initialize_p2_impl(void* data_ptr, int* nx, int* ny, int* nz, int* rc);

/**
 * @brief Fortran-compatible wrapper for cece_core_initialize_p2.
 *
 * Matches signature:
 * subroutine cece_core_initialize_p2(data_ptr, nx, ny, nz, rc) bind(C)
 *   type(c_ptr), value :: data_ptr
 *   integer(c_int), intent(in) :: nx, ny, nz
 *   integer(c_int), intent(out) :: rc
 * end subroutine
 */
void cece_core_initialize_p2(void* data_ptr, int* nx, int* ny, int* nz, int* rc) {
    cece_core_initialize_p2_impl(data_ptr, nx, ny, nz, rc);
}
}
