!> @file cece_fms_model.F90
!> @brief FMS Component Model cap for CECE
!>
!> This module implements a standard FMS component model interface for the
!> Community Emissions Computing Engine (CECE). It provides initialization,
!> update (run), and finalization routines that map to the underlying C++
!> CECE engine, allowing CECE to be run as a standalone component within the
!> Flexible Modeling System (FMS).

module cece_fms_model_mod
  use iso_c_binding

  ! Use FMS modules. If FMS is not available, compiling this will fail.
  ! This is expected for an FMS component cap.
  use time_manager_mod, only: time_type, get_time, get_date
  use fms_mod, only: mpp_pe, mpp_root_pe, stdlog, error_mesg, FATAL, NOTE, WARNING

  implicit none
  private

  public :: cece_fms_init, cece_fms_update, cece_fms_end

  !> @brief Module-level C++ data pointer (save ensures persistence across phases).
  type(c_ptr), save :: g_cece_data_ptr = c_null_ptr

  !> @brief Global flags
  logical, save :: module_is_initialized = .false.

  ! CECE Core C-API Interfaces
  interface
    subroutine cece_core_initialize_p1(data_ptr, rc) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), intent(out) :: data_ptr
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_core_initialize_p2(data_ptr, nx, ny, nz, rc) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: data_ptr
      integer(c_int), intent(in) :: nx, ny, nz
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_core_run(data_ptr, hour, day_of_week, rc) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: data_ptr
      integer(c_int), value :: hour, day_of_week
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_core_finalize(data_ptr, rc) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: data_ptr
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_core_write_step(data_ptr, time_seconds, step_index, rc) bind(C)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: data_ptr
      real(c_double), value :: time_seconds
      integer(c_int), value :: step_index
      integer(c_int), intent(out) :: rc
    end subroutine
  end interface

contains

  !> @brief Initialize the CECE FMS component.
  subroutine cece_fms_init(Time_init, Time, nx, ny, nz)
    type(time_type), intent(in) :: Time_init
    type(time_type), intent(in) :: Time
    integer, intent(in) :: nx, ny, nz

    integer(c_int) :: rc

    if (module_is_initialized) return

    call error_mesg('cece_fms_model', 'Initializing CECE Component', NOTE)

    ! Call CECE Phase 1 Initialization
    call cece_core_initialize_p1(g_cece_data_ptr, rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_initialize_p1 failed', FATAL)
    end if

    ! Call CECE Phase 2 Initialization with grid dimensions
    call cece_core_initialize_p2(g_cece_data_ptr, int(nx, c_int), int(ny, c_int), int(nz, c_int), rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_initialize_p2 failed', FATAL)
    end if

    module_is_initialized = .true.

  end subroutine cece_fms_init

  !> @brief Update (Advance) the CECE FMS component for a timestep.
  subroutine cece_fms_update(Time)
    type(time_type), intent(in) :: Time

    integer :: year, month, day, hour, minute, second
    integer :: day_of_week
    integer(c_int) :: rc
    real(c_double) :: time_seconds

    if (.not. module_is_initialized) then
      call error_mesg('cece_fms_model', 'cece_fms_update called before initialization', FATAL)
    end if

    ! Extract time information from FMS clock
    call get_date(Time, year, month, day, hour, minute, second)

    ! In a real FMS model, we would calculate the actual day of week based on the calendar.
    ! For CECE purposes, 0=Sunday, 6=Saturday. Using a dummy value for the example.
    day_of_week = 1 ! default to Monday

    ! Calculate total seconds for the writer (simplified)
    time_seconds = real(hour * 3600 + minute * 60 + second, c_double)

    ! Call the generic CECE run core
    call cece_core_run(g_cece_data_ptr, int(hour, c_int), int(day_of_week, c_int), rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_run failed', FATAL)
    end if

    ! If standalone writer is used, trigger output step
    call cece_core_write_step(g_cece_data_ptr, time_seconds, 1_c_int, rc)

  end subroutine cece_fms_update

  !> @brief Finalize and clean up the CECE FMS component.
  subroutine cece_fms_end()
    integer(c_int) :: rc

    if (.not. module_is_initialized) return

    call error_mesg('cece_fms_model', 'Finalizing CECE Component', NOTE)

    call cece_core_finalize(g_cece_data_ptr, rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_finalize failed', WARNING)
    end if

    g_cece_data_ptr = c_null_ptr
    module_is_initialized = .false.

  end subroutine cece_fms_end

end module cece_fms_model_mod