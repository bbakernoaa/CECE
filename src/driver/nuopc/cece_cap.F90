!> @file cece_cap.F90
!> @brief Production-grade decoupled NUOPC Model cap for CECE.
!>
!> The cap is a thin adapter over the shared simulation contract
!> (cece_sim_* C ABI): all lifecycle sequencing, export-field registration,
!> the time convention (ingest at step start, stamp at step end), and
!> teardown live inside the shared core so they cannot diverge from the
!> C++ standalone driver. The two drivers differ ONLY in how they obtain
!> the target grid; this cap resolves it from the CECE YAML through the
!> same code path the C++ driver runs.
module cece_cap_mod
  use iso_c_binding
  use ESMF
  use NUOPC
  use NUOPC_Model, modelSS => SetServices
  use NUOPC_Model, only: &
    label_Advertise, &
    label_RealizeProvided, &
    model_label_Advance => label_Advance, &
    model_label_Finalize => label_Finalize
  use cece_cap_grid_mod
  implicit none

  private

  public :: CECE_SetServices, CECE_SetConfigPath

  !> @brief Shared simulation handle (opaque cece::CeceSimulation*).
  type(c_ptr), save :: g_sim_ptr = c_null_ptr

  !> @brief Module-level config file path (save ensures persistence across phases).
  character(len=512), save :: g_config_file_path = "cece_control_mock.yaml"

  !> @brief Monotonic step counter; the shared writer counts steps 1-based.
  integer, save :: g_step_count = 0

  !> @brief Latched once the shared core reports completion; further
  !> advances then do no work (contract: hosts must stop stepping on the
  !> same signal as the C++ driver).
  logical, save :: g_complete = .false.

  ! C interfaces to the shared simulation facade (cece_driver library).
  ! These mirror include/cece/cece_sim_c_abi.h exactly.
  interface
    subroutine cece_set_config_file_path(config_path, path_len) &
                                         bind(C, name="cece_set_config_file_path")
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
    end subroutine

    subroutine cece_run_log_setup(config_path, path_len) &
                                  bind(C, name="cece_run_log_setup")
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
    end subroutine

    ! void cece_sim_create_from_yaml(const char* config_path, int path_len,
    !                                int mpi_comm_f, CeceSimulation** out_sim,
    !                                int* rc)
    subroutine cece_sim_create_from_yaml(config_path, path_len, mpi_comm_f, out_sim, rc) &
                                         bind(C, name="cece_sim_create_from_yaml")
      import :: c_char, c_int, c_ptr
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
      integer(c_int), value :: mpi_comm_f
      type(c_ptr), intent(out) :: out_sim
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_step(CeceSimulation* sim,
    !                    const char* step_start_iso, int step_start_len,
    !                    const char* step_end_iso, int step_end_len,
    !                    int step_index, int* complete_out, int* rc)
    subroutine cece_sim_step(sim, step_start_iso, step_start_len, &
                             step_end_iso, step_end_len, &
                             step_index, complete_out, rc) &
                             bind(C, name="cece_sim_step")
      import :: c_char, c_int, c_ptr
      type(c_ptr), value :: sim
      character(kind=c_char), intent(in) :: step_start_iso(*), step_end_iso(*)
      integer(c_int), value :: step_start_len, step_end_len
      integer(c_int), value :: step_index
      integer(c_int), intent(out) :: complete_out
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_finalize(CeceSimulation* sim, int* rc)
    subroutine cece_sim_finalize(sim, rc) &
                             bind(C, name="cece_sim_finalize")
      import :: c_ptr, c_int
      type(c_ptr), value :: sim
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_grid_info(const CeceSimulation* sim,
    !                         int* nx, int* ny, int* nz, int* topology,
    !                         double* lon_min, double* lon_max,
    !                         double* lat_min, double* lat_max, int* rc)
    subroutine cece_sim_grid_info(sim, nx, ny, nz, topology, &
                                  lon_min, lon_max, lat_min, lat_max, rc) &
                                  bind(C, name="cece_sim_grid_info")
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: sim
      integer(c_int), intent(out) :: nx, ny, nz, topology
      real(c_double), intent(out) :: lon_min, lon_max, lat_min, lat_max
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_nz_from_config(const char* config_path, int path_len,
    !                              int* nz_out, int* rc)
    subroutine cece_sim_nz_from_config(config_path, path_len, nz_out, rc) &
                                       bind(C, name="cece_sim_nz_from_config")
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
      integer(c_int), intent(out) :: nz_out
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_describe_yaml_grid(const char* config_path, int path_len,
    !                                  char* buf, int buf_max,
    !                                  int* buf_len_out, int* rc)
    subroutine cece_sim_describe_yaml_grid(config_path, path_len, buf, &
                                           buf_max, buf_len_out, rc) &
                                           bind(C, name="cece_sim_describe_yaml_grid")
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
      character(kind=c_char), intent(out) :: buf(*)
      integer(c_int), value :: buf_max
      integer(c_int), intent(out) :: buf_len_out
      integer(c_int), intent(out) :: rc
    end subroutine

    ! void cece_sim_create_from_esmf(const char* config_path, int path_len,
    !   int nx, int ny, int nz, int is_rad,
    !   const double* lon_coords, int lon_len,
    !   const double* lat_coords, int lat_len,
    !   int mpi_comm_f, CeceSimulation** out_sim, int* rc)
    subroutine cece_sim_create_from_esmf(config_path, path_len, nx, ny, nz, &
                                         is_rad, lon_coords, lon_len, &
                                         lat_coords, lat_len, mpi_comm_f, &
                                         out_sim, rc) &
                                         bind(C, name="cece_sim_create_from_esmf")
      import :: c_char, c_int, c_ptr, c_double
      character(kind=c_char), intent(in) :: config_path(*)
      integer(c_int), value :: path_len
      integer(c_int), value :: nx, ny, nz, is_rad
      real(c_double), intent(in) :: lon_coords(*)
      integer(c_int), value :: lon_len
      real(c_double), intent(in) :: lat_coords(*)
      integer(c_int), value :: lat_len
      integer(c_int), value :: mpi_comm_f
      type(c_ptr), intent(out) :: out_sim
      integer(c_int), intent(out) :: rc
    end subroutine
  end interface

contains

  !> @brief Set the YAML config file path dynamically from parent driver
  subroutine CECE_SetConfigPath(config_path, rc)
    character(len=*), intent(in) :: config_path
    integer, intent(out) :: rc
    g_config_file_path = config_path
    rc = ESMF_SUCCESS
  end subroutine CECE_SetConfigPath

  !> @brief SetServices routine for the production CECE component
  subroutine CECE_SetServices(gcomp, rc)
    type(ESMF_GridComp) :: gcomp
    integer, intent(out) :: rc

    write(*,'(A)') "INFO: [Cap] CECE_SetServices entered"
    rc = ESMF_SUCCESS

    ! 1. Inherit NUOPC Model base services
    write(*,'(A)') "INFO: [Cap] Calling NUOPC_CompDerive..."
    call NUOPC_CompDerive(gcomp, modelSS, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] NUOPC_CompDerive failed rc=", rc
      return
    end if

    ! 2. Register initialization phase 1 (Advertise)
    write(*,'(A)') "INFO: [Cap] Specializing Initialize phase 1 (Advertise)..."
    call NUOPC_CompSpecialize(gcomp, specLabel=label_Advertise, &
      specRoutine=InitializeAdvertise, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] CompSpecialize(Advertise) failed rc=", rc
      return
    end if

    ! 3. Register initialization phase 2 (Realize)
    write(*,'(A)') "INFO: [Cap] Specializing Initialize phase 2 (Realize)..."
    call NUOPC_CompSpecialize(gcomp, specLabel=label_RealizeProvided, &
      specRoutine=InitializeRealize, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] CompSpecialize(Realize) failed rc=", rc
      return
    end if

    ! 4. Register Run (Advance) phase
    write(*,'(A)') "INFO: [Cap] Specializing Run (Advance) phase..."
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_Advance, &
      specRoutine=Run, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] CompSpecialize(Advance) failed rc=", rc
      return
    end if

    ! 5. Register Finalize phase
    write(*,'(A)') "INFO: [Cap] Specializing Finalize phase..."
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_Finalize, &
      specRoutine=Finalize, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] CompSpecialize(Finalize) failed rc=", rc
      return
    end if

    write(*,'(A)') "INFO: [Cap] CECE_SetServices completed successfully"
  end subroutine CECE_SetServices

  !> @brief InitializeAdvertise (IPDv01p1)
  !>
  !> Records the configuration path and sets up run logging/banner only.
  !> The simulation itself is built in InitializeRealize, where the target
  !> grid is resolved — a parent component can only provide its grid once
  !> realization has begun, so construction must wait for that phase.
  subroutine InitializeAdvertise(comp, rc)
    type(ESMF_GridComp)  :: comp
    integer, intent(out) :: rc

    rc = ESMF_SUCCESS
    write(*,'(A)') "INFO: [Cap] InitializeAdvertise entered"

    ! Set YAML configuration path in the core C-API
    call cece_set_config_file_path(trim(g_config_file_path)//c_null_char, &
                                   int(len_trim(g_config_file_path), c_int))

    ! Configure run logging (optional log file, per-rank stdout suppression) and
    ! print the startup banner. Shared with the standalone driver so behavior is
    ! identical regardless of how CECE is launched. The shared simulation core
    ! re-invokes both during creation; the redirect, banner, and path are
    ! idempotent, but the banner must appear before grid resolution, so the cap
    ! calls them here too.
    call cece_run_log_setup(trim(g_config_file_path)//c_null_char, &
                            int(len_trim(g_config_file_path), c_int))

    write(*,'(A)') "INFO: [Cap] InitializeAdvertise completed successfully"
  end subroutine InitializeAdvertise

  !> @brief InitializeRealize (IPDv01p3)
  !>
  !> Resolves the target grid and builds the simulation through the shared
  !> facade. Grid resolution order: a parent-provided ESMF Grid/Mesh (a
  !> coupled run associates one with the component before realization) wins;
  !> otherwise the grid comes from the CECE YAML via the exact same code path
  !> as the C++ driver. When both exist, the parent grid is used and a single
  !> warning names the ignored YAML grid — the two sources are never merged.
  !> An unsupported parent topology fails loudly at realization; there is no
  !> fallback to a uniform grid. The vertical layer count always comes from
  !> the config, never from the flat 2-D grid.
  !>
  !> Standalone case (no parent grid): after the simulation is built, the
  !> component is associated with a uniform ESMF grid spanning the resolved
  !> coordinate extents, so metadata consumers see the correct geometry
  !> without the cap duplicating coordinate derivation.
  subroutine InitializeRealize(comp, rc)
    type(ESMF_GridComp) :: comp
    integer, intent(out) :: rc

    type(ESMF_Grid) :: grid
    type(ESMF_VM) :: vm
    integer :: mpi_comm_val
    integer(c_int) :: c_rc
    integer(c_int) :: nx_c, ny_c, nz_c, topology_c
    real(c_double) :: lon_min, lon_max, lat_min, lat_max

    ! Parent-grid discovery state
    logical :: grid_is_present
    logical :: mesh_is_present
    type(ESMF_Grid) :: parent_grid
    type(ESMF_Mesh) :: parent_mesh
    integer :: gx_nx, gx_ny, gx_is_rad, gx_rc
    real(ESMF_KIND_R8), allocatable :: gx_lon(:), gx_lat(:)
    integer(c_int) :: nz_cfg
    integer(c_int) :: desc_len
    character(len=512) :: yaml_desc
    logical :: vm_ok

    rc = ESMF_SUCCESS
    write(*,'(A)') "INFO: [Cap] InitializeRealize entered"

    ! Retrieve the raw ESMF VM MPI communicator (same handle the C++ driver
    ! passes: an MPI_Comm_c2f value; 0 lets the facade default to
    ! MPI_COMM_WORLD).
    mpi_comm_val = 0
    vm_ok = .false.
    call ESMF_GridCompGet(comp, vm=vm, rc=rc)
    if (rc == ESMF_SUCCESS) then
      vm_ok = .true.
      call ESMF_VMGet(vm, mpiCommunicator=mpi_comm_val, rc=rc)
      if (rc /= ESMF_SUCCESS) then
        write(*,'(A,I0)') 'WARNING: [Cap] ESMF_VMGet communicator failed rc=', rc
        mpi_comm_val = 0
        rc = ESMF_SUCCESS
      end if
    else
      write(*,'(A,I0)') 'WARNING: [Cap] ESMF_GridCompGet(vm) failed rc=', rc
      mpi_comm_val = 0
      rc = ESMF_SUCCESS
    end if

    ! Resolve the grid source. A parent grid/mesh already associated with the
    ! component wins over the config-built YAML grid. Query the presence
    ! flags first: retrieving an absent grid or mesh would itself be an
    ! error, so the objects are only fetched when flagged present.
    grid_is_present = .false.
    mesh_is_present = .false.
    call ESMF_GridCompGet(comp, gridIsPresent=grid_is_present, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') 'WARNING: [Cap] ESMF_GridCompGet(gridIsPresent) failed rc=', rc
      grid_is_present = .false.
      rc = ESMF_SUCCESS
    end if
    call ESMF_GridCompGet(comp, meshIsPresent=mesh_is_present, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') 'WARNING: [Cap] ESMF_GridCompGet(meshIsPresent) failed rc=', rc
      mesh_is_present = .false.
      rc = ESMF_SUCCESS
    end if
    if (grid_is_present) then
      call ESMF_GridCompGet(comp, grid=parent_grid, rc=rc)
      if (rc /= ESMF_SUCCESS) then
        write(*,'(A,I0)') 'ERROR: [Cap] Parent grid flagged present but retrieval failed rc=', rc
        return
      end if
    end if
    if (mesh_is_present) then
      call ESMF_GridCompGet(comp, mesh=parent_mesh, rc=rc)
      if (rc /= ESMF_SUCCESS) then
        write(*,'(A,I0)') 'ERROR: [Cap] Parent mesh flagged present but retrieval failed rc=', rc
        return
      end if
    end if

    if (grid_is_present .or. mesh_is_present) then
      ! Assembling the global coordinates is a collective operation on the
      ! component's VM, so the parent-grid path cannot proceed without it.
      if (.not. vm_ok) then
        write(*,'(A)') 'ERROR: [Cap] A parent ESMF grid is present but the', &
          ' component VM could not be retrieved; cannot assemble global', &
          ' coordinates (no fallback to a uniform grid).'
        rc = ESMF_FAILURE
        return
      end if
      ! Parent-provided grid path. The vertical layer count is always read
      ! from the config (driver.grid.nz): a flat 2-D grid carries no vertical
      ! dimension, so it can never supply nz.
      call cece_sim_nz_from_config(trim(g_config_file_path)//c_null_char, &
                                   int(len_trim(g_config_file_path), c_int), &
                                   nz_cfg, c_rc)
      if (c_rc /= 0 .or. nz_cfg <= 0) then
        write(*,'(A,I0)') 'ERROR: [Cap] Could not resolve vertical layer count from config rc=', int(c_rc)
        rc = ESMF_FAILURE
        return
      end if

      ! Precedence: when the YAML also describes a grid, name it as ignored
      ! (a single warning; the two sources are never merged). If the YAML
      ! grid cannot be derived at all there is nothing to warn about.
      yaml_desc = ' '
      desc_len = 0
      call cece_sim_describe_yaml_grid(trim(g_config_file_path)//c_null_char, &
                                       int(len_trim(g_config_file_path), c_int), &
                                       yaml_desc, int(len(yaml_desc), c_int), &
                                       desc_len, c_rc)
      if (c_rc == 0 .and. desc_len > 0) then
        if (desc_len <= len(yaml_desc)) then
          write(*,'(A,A)') 'WARNING: [Cap] Parent ESMF grid takes precedence;', &
            ' ignoring the grid defined in the CECE config: ', yaml_desc(1:desc_len)
        else
          write(*,'(A)') 'WARNING: [Cap] Parent ESMF grid takes precedence;', &
            ' ignoring the grid defined in the CECE config.'
        end if
      end if

      ! Extract the global coordinate arrays across PETs. ESMF coordinate
      ! access is Fortran-only, so the cap does the gathering and hands the
      ! C++ facade plain arrays; normalization, topology classification and
      ! validation happen in shared C++ code. An unsupported parent
      ! topology fails here with a named diagnostic — no uniform-grid
      ! fallback is attempted.
      if (grid_is_present) then
        call cece_cap_extract_parent_grid(grid=parent_grid, vm=vm, &
             nx=gx_nx, ny=gx_ny, is_rad=gx_is_rad, &
             lon=gx_lon, lat=gx_lat, rc=gx_rc)
      else
        call cece_cap_extract_parent_grid(mesh=parent_mesh, vm=vm, &
             nx=gx_nx, ny=gx_ny, is_rad=gx_is_rad, &
             lon=gx_lon, lat=gx_lat, rc=gx_rc)
      end if
      if (gx_rc /= ESMF_SUCCESS) then
        write(*,'(A)') 'ERROR: [Cap] Parent grid extraction failed; no fallback', &
          ' to a uniform grid is attempted.'
        rc = ESMF_FAILURE
        return
      end if

      call cece_sim_create_from_esmf(trim(g_config_file_path)//c_null_char, &
                                     int(len_trim(g_config_file_path), c_int), &
                                     int(gx_nx, c_int), int(gx_ny, c_int), &
                                     nz_cfg, int(gx_is_rad, c_int), &
                                     gx_lon, int(size(gx_lon), c_int), &
                                     gx_lat, int(size(gx_lat), c_int), &
                                     int(mpi_comm_val, c_int), g_sim_ptr, c_rc)
      if (allocated(gx_lon)) deallocate(gx_lon)
      if (allocated(gx_lat)) deallocate(gx_lat)
      if (c_rc /= 0 .or. .not. c_associated(g_sim_ptr)) then
        write(*,'(A,I0)') 'ERROR: [Cap] Failed to create CECE simulation on the', &
          ' parent-provided grid rc=', int(c_rc)
        g_sim_ptr = c_null_ptr
        rc = ESMF_FAILURE
        return
      end if

      write(*,'(A,I0,A,I0,A,I0)') 'INFO: [Cap] Simulation grid from parent ESMF object: ', &
        gx_nx, 'x', gx_ny, 'x', int(nz_cfg)
      ! The component is already associated with the parent grid/mesh, so no
      ! re-association is needed on this path.
      write(*,'(A)') "INFO: [Cap] InitializeRealize completed successfully"
      return
    end if

    ! Standalone case (no parent grid): build the simulation through the
    ! shared facade. Grid resolution (named grids, gridspec file,
    ! stream-inferred coordinates, uniform extents) runs inside C++ on the
    ! same code path as the standalone driver, together with core init,
    ! export-field registration, the driver orchestrator, and the output
    ! writer. If neither a parent grid nor a derivable YAML grid exists, the
    ! facade call fails loudly below — there is no silent substitution.
    call cece_sim_create_from_yaml(trim(g_config_file_path)//c_null_char, &
                                   int(len_trim(g_config_file_path), c_int), &
                                   int(mpi_comm_val, c_int), g_sim_ptr, c_rc)
    if (c_rc /= 0 .or. .not. c_associated(g_sim_ptr)) then
      write(*,'(A,I0)') "ERROR: [Cap] Failed to create CECE simulation rc=", int(c_rc)
      g_sim_ptr = c_null_ptr
      rc = ESMF_FAILURE
      return
    end if

    ! Read the resolved grid back so the component can be associated with a
    ! matching ESMF grid. Rectilinear grids carry one row of ny cells;
    ! flattened curvilinear/unstructured grids carry ny == 1 with nx nodes.
    call cece_sim_grid_info(g_sim_ptr, nx_c, ny_c, nz_c, topology_c, &
                            lon_min, lon_max, lat_min, lat_max, c_rc)
    if (c_rc /= 0) then
      write(*,'(A,I0)') "ERROR: [Cap] Failed to read resolved grid info rc=", int(c_rc)
      rc = ESMF_FAILURE
      return
    end if

    write(*,'(A,I0,A,I0,A,I0,A,I0)') "INFO: [Cap] Simulation grid: ", nx_c, "x", ny_c, "x", nz_c, &
      " topology=", topology_c

    grid = ESMF_GridCreateNoPeriDimUfrm(maxIndex=(/nx_c, ny_c/), &
      minCornerCoord=(/lon_min, lat_min/), &
      maxCornerCoord=(/lon_max, lat_max/), &
      coordSys=ESMF_COORDSYS_SPH_DEG, rc=rc)
    if (rc /= ESMF_SUCCESS) then
      write(*,'(A,I0)') "ERROR: [Cap] Failed to create ESMF grid rc=", rc
      return
    end if

    call ESMF_GridCompSet(comp, grid=grid, rc=rc)
    if (rc /= ESMF_SUCCESS) return

    write(*,'(A)') "INFO: [Cap] InitializeRealize completed successfully"
  end subroutine InitializeRealize

  !> @brief Run Advance step (Specialized via model_label_Advance)
  !>
  !> Drives one step of the shared simulation. The NUOPC clock's currTime
  !> is the step START instant (observed: the first Advance sees
  !> currTime == clock start), and the step END instant is the clock's
  !> next time. The shared core ingests at step start, computes, and stamps
  !> output at step end; completion is honored via the shared complete
  !> signal, so both drivers take identical step counts.
  subroutine Run(comp, rc)
    type(ESMF_GridComp) :: comp
    integer, intent(out) :: rc

    type(ESMF_Clock) :: clock
    type(ESMF_Time) :: currTime
    type(ESMF_Time) :: nextTime
    character(len=64) :: step_start_str, step_end_str
    integer(c_int) :: complete_c
    integer(c_int) :: c_rc

    rc = ESMF_SUCCESS

    ! Once the shared core reports completion, the host must stop stepping:
    ! do no further work on later advances (the harness may still call).
    if (g_complete) then
      write(*,'(A)') "INFO: [Cap] Simulation already complete; skipping advance"
      return
    end if

    if (.not. c_associated(g_sim_ptr)) then
      write(*,'(A)') "ERROR: [Cap] No live simulation at advance; was Realize skipped?"
      rc = ESMF_FAILURE
      return
    end if

    call ESMF_GridCompGet(comp, clock=clock, rc=rc)
    if (rc /= ESMF_SUCCESS) return

    call ESMF_ClockGet(clock, currTime=currTime, rc=rc)
    if (rc /= ESMF_SUCCESS) return

    call ESMF_ClockGetNextTime(clock, nextTime, rc=rc)
    if (rc /= ESMF_SUCCESS) return

    call ESMF_TimeGet(currTime, timeString=step_start_str, rc=rc)
    if (rc /= ESMF_SUCCESS) return
    call ESMF_TimeGet(nextTime, timeString=step_end_str, rc=rc)
    if (rc /= ESMF_SUCCESS) return

    ! The writer counts steps 1-based (output frequency is checked as
    ! step_index % output_freq), matching the standalone driver's counter.
    g_step_count = g_step_count + 1

    call cece_sim_step(g_sim_ptr, trim(step_start_str)//c_null_char, &
                       int(len_trim(step_start_str), c_int), &
                       trim(step_end_str)//c_null_char, &
                       int(len_trim(step_end_str), c_int), &
                       int(g_step_count, c_int), complete_c, c_rc)
    if (c_rc < 0) then
      write(*,'(A,I0)') "ERROR: [Cap] cece_sim_step failed rc=", int(c_rc)
      rc = ESMF_FAILURE
      return
    end if

    if (complete_c /= 0) then
      g_complete = .true.
      write(*,'(A)') "INFO: [Cap] Shared core reported simulation completion"
    end if
  end subroutine Run

  !> @brief Finalize and cleanup resources (Specialized via model_label_Finalize)
  subroutine Finalize(comp, rc)
    type(ESMF_GridComp) :: comp
    integer, intent(out) :: rc

    integer(c_int) :: c_rc

    rc = ESMF_SUCCESS
    write(*,'(A)') "INFO: [Cap] Finalizing CECE NUOPC Cap..."

    ! Tear down the shared simulation: driver orchestrator destroy, then
    ! core finalize. Non-zero rc is a warning only — output has already
    ! been flushed — matching the standalone driver's teardown semantics.
    call cece_sim_finalize(g_sim_ptr, c_rc)
    if (c_rc /= 0) then
      write(*,'(A,I0)') 'WARNING: [Cap] CECE teardown reported failures (rc=', int(c_rc)
    end if
    g_sim_ptr = c_null_ptr

    write(*,'(A)') "INFO: [Cap] CECE NUOPC Cap finalized successfully"
  end subroutine Finalize

end module cece_cap_mod
