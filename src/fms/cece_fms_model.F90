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

  ! Use standard FMS modules for component implementation
  use time_manager_mod, only: time_type, get_time, get_date, get_calendar_type, NOLEAP
  use fms_mod, only: mpp_pe, mpp_root_pe, stdlog, error_mesg, FATAL, NOTE, WARNING
  use mpp_domains_mod, only: domain2d, mpp_get_compute_domain
  use coupler_types_mod, only: coupler_2d_bc_type
  use tracer_manager_mod, only: get_tracer_index

  implicit none
  private

  public :: cece_fms_init, cece_fms_update, cece_fms_end

  ! Forward declare helper functions
  private :: get_day_of_week

  !> @brief Module-level C++ data pointer (save ensures persistence across phases).
  type(c_ptr), save :: g_cece_data_ptr = c_null_ptr

  !> @brief Global flags
  logical, save :: module_is_initialized = .false.

  ! CECE Core C-API Interfaces
  interface
    subroutine cece_core_get_species_count(data_ptr, count, rc) bind(C)
      import :: c_ptr, c_int
      type(c_ptr), value :: data_ptr
      integer(c_int), intent(out) :: count
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_core_get_species_name(data_ptr, index, name, name_len, rc) bind(C)
      import :: c_ptr, c_int, c_char
      type(c_ptr), value :: data_ptr
      integer(c_int), value :: index
      character(kind=c_char), intent(out) :: name(*)
      integer(c_int), value :: name_len
      integer(c_int), intent(out) :: rc
    end subroutine

    subroutine cece_ccpp_get_export_field(data_ptr, name, name_len, &
                                          field_data, nx, ny, nz, rc) bind(C)
      import :: c_ptr, c_char, c_int, c_double
      type(c_ptr), value :: data_ptr
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), value :: name_len, nx, ny, nz
      real(c_double), intent(out) :: field_data(*)
      integer(c_int), intent(out) :: rc
    end subroutine

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
  !> Extracts the local grid dimensions from the provided FMS domain.
  subroutine cece_fms_init(Time_init, Time, domain, nz)
    type(time_type), intent(in) :: Time_init
    type(time_type), intent(in) :: Time
    type(domain2d),  intent(in) :: domain
    integer,         intent(in) :: nz

    integer :: is, ie, js, je
    integer :: nx, ny
    integer(c_int) :: rc

    if (module_is_initialized) return

    call error_mesg('cece_fms_model', 'Initializing CECE Component', NOTE)

    ! Extract local compute domain size from the FMS spatial domain
    call mpp_get_compute_domain(domain, is, ie, js, je)
    nx = ie - is + 1
    ny = je - js + 1

    ! Call CECE Phase 1 Initialization (reads configuration and initializes Kokkos/Physics)
    call cece_core_initialize_p1(g_cece_data_ptr, rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_initialize_p1 failed', FATAL)
    end if

    ! Call CECE Phase 2 Initialization with the computed grid dimensions
    call cece_core_initialize_p2(g_cece_data_ptr, int(nx, c_int), int(ny, c_int), int(nz, c_int), rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_initialize_p2 failed', FATAL)
    end if

    module_is_initialized = .true.

  end subroutine cece_fms_init

  !> @brief Update (Advance) the CECE FMS component for a timestep.
  !> Passes calculated emissions into the provided FMS coupler boundary type.
  subroutine cece_fms_update(Time, nx, ny, Atmos_bc)
    type(time_type), intent(in) :: Time
    integer, intent(in) :: nx, ny
    type(coupler_2d_bc_type), intent(inout) :: Atmos_bc

    integer :: year, month, day, hour, minute, second
    integer :: day_of_week
    integer(c_int) :: rc
    real(c_double) :: time_seconds

    integer(c_int) :: num_species, i, tr_idx
    character(len=256) :: species_name
    real(kind=8), allocatable :: export_field(:,:)

    if (.not. module_is_initialized) then
      call error_mesg('cece_fms_model', 'cece_fms_update called before initialization', FATAL)
    end if

    ! Extract time information from FMS clock
    call get_date(Time, year, month, day, hour, minute, second)

    ! Calculate the true day of the week (0=Sunday, ..., 6=Saturday) based on the current calendar date
    day_of_week = get_day_of_week(year, month, day)

    ! Calculate total elapsed seconds for the current day
    time_seconds = real(hour * 3600 + minute * 60 + second, c_double)

    ! Call the generic CECE run core. This advances physics schemes and interpolates diurnal cycles
    call cece_core_run(g_cece_data_ptr, int(hour, c_int), int(day_of_week, c_int), rc)
    if (rc /= 0) then
      call error_mesg('cece_fms_model', 'cece_core_run failed', FATAL)
    end if

    ! Extract configured species from CECE and populate the FMS coupler boundary object
    call cece_core_get_species_count(g_cece_data_ptr, num_species, rc)

    if (num_species > 0) then
      allocate(export_field(nx, ny))

      do i = 1, num_species
        species_name = ''
        call cece_core_get_species_name(g_cece_data_ptr, int(i-1, c_int), species_name, 256_c_int, rc)

        ! Pull the 2D emission slice from CECE
        call cece_ccpp_get_export_field(g_cece_data_ptr, trim(species_name)//c_null_char, &
             len_trim(species_name), export_field, int(nx, c_int), int(ny, c_int), 1_c_int, rc)

        if (rc == 0) then
          ! Find corresponding tracer index in the FMS tracer manager
          tr_idx = get_tracer_index('atmos_mod', trim(species_name))
          if (tr_idx > 0) then
            ! Load data into the FMS coupler boundary struct for upstream components (e.g. Atmosphere)
            Atmos_bc%flux(tr_idx)%field = export_field
          end if
        end if
      end do

      deallocate(export_field)
    end if

    ! If the CECE standalone diagnostic writer is enabled, trigger the output step
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

  !> @brief Calculates the day of the week given a Gregorian date.
  !> Returns an integer where 0=Sunday, 1=Monday, ..., 6=Saturday.
  integer function get_day_of_week(year, month, day)
    integer, intent(in) :: year, month, day
    integer :: a, y, m, jdn

    ! Calculate using Zeller's congruence adapted for Julian Day Number
    a = (14 - month) / 12
    y = year + 4800 - a
    m = month + 12 * a - 3

    ! Julian Day Number (JDN)
    jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045

    ! JDN modulo 7 yields 0 for Monday. We add 1 to shift 0 to Sunday.
    get_day_of_week = mod(jdn + 1, 7)

  end function get_day_of_week

end module cece_fms_model_mod