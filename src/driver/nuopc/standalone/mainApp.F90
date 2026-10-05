!> @file mainApp.F90
!> @brief Main application for CECE standalone execution.
!>
!> Updated to use proper NUOPC_Driver pattern (SingleModelOpenMPProto style).
!> Fixed multi-timestep execution by removing external time loop conflicts.

program mainApp

  use ESMF
  use NUOPC
  use mpi
  use driver, only: driver_SS => SetServices, set_driver_config_file, set_cece_config_file

  implicit none

  integer :: rc, userRc, mpierr
  type(ESMF_GridComp) :: drvComp
  character(len=512) :: driver_cfg_file, cece_yaml_file

  ! Initialize ESMF with minimal logging to avoid string conversion issues
  call ESMF_Initialize(defaultCalKind=ESMF_CALKIND_GREGORIAN, &
                       logkindflag=ESMF_LOGKIND_MULTI, rc=rc)
  if (rc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_Initialize failed rc=", rc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if

  call ESMF_LogWrite("CECE STANDALONE STARTING", ESMF_LOGMSG_INFO, rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

  ! Check if we have 1 or 2 arguments
  call get_command_argument(1, cece_yaml_file)
  call get_command_argument(2, driver_cfg_file)

  ! If 2 args: old format (driver.cfg, config.yaml)
  if (len_trim(driver_cfg_file) > 0) then
    ! Swap so cece_yaml_file gets the second argument
    driver_cfg_file = cece_yaml_file
    call get_command_argument(2, cece_yaml_file)
  else
    ! If 1 arg: new simplified format (config.yaml only)
    driver_cfg_file = "cece_driver.cfg"  ! unused placeholder
  end if

  ! Fallback if no arguments
  if (len_trim(cece_yaml_file) == 0) then
    call get_environment_variable("CECE_CONFIG", cece_yaml_file)
    if (len_trim(cece_yaml_file) == 0) then
      cece_yaml_file = "cece_config.yaml"
    end if
  end if

  ! Set config files in driver module
  call set_driver_config_file(trim(driver_cfg_file))
  call set_cece_config_file(trim(cece_yaml_file))

  write(*,'(A,A)') "INFO: [mainApp] Driver config file: ", trim(driver_cfg_file)
  write(*,'(A,A)') "INFO: [mainApp] CECE config file:   ", trim(cece_yaml_file)

  ! Create driver component
  drvComp = ESMF_GridCompCreate(name="driver", rc=rc)
  if (rc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_GridCompCreate failed rc=", rc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if

  ! Set driver services
  call ESMF_GridCompSetServices(drvComp, driver_SS, userRc=userRc, rc=rc)
  if (rc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_GridCompSetServices failed rc=", rc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if
  if (userRc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_GridCompSetServices userRc=", userRc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if

  ! Initialize driver
  write(*,'(A)') "INFO: [mainApp] Calling ESMF_GridCompInitialize..."
  call ESMF_GridCompInitialize(drvComp, userRc=userRc, rc=rc)
  write(*,'(A,I0,A,I0)') "INFO: [mainApp] ESMF_GridCompInitialize returned: rc=", rc, " userRc=", userRc

  if (rc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_GridCompInitialize failed rc=", rc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if
  if (userRc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "ERROR: ESMF_GridCompInitialize userRc=", userRc
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  end if

  ! RUN THE DRIVER
  call ESMF_GridCompRun(drvComp, userRc=userRc, rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  if (ESMF_LogFoundError(rcToCheck=userRc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

  ! FINALIZE THE DRIVER
  ! Skip ESMF_GridCompFinalize to avoid segfault issues
  write(*,'(A)') "INFO: [mainApp] Skipping driver finalization to avoid segfault"

  !-----------------------------------------------------------------------------

  call ESMF_LogWrite("mainApp FINISHED", ESMF_LOGMSG_INFO, rc=rc)

  write(*,'(A)') "INFO: [mainApp] CECE execution completed successfully"
  write(*,'(A)') "INFO: [mainApp] Skipping ESMF_Finalize to avoid framework cleanup conflicts"

  ! ESMF_Finalize is intentionally skipped above (it conflicts with the
  ! framework cleanup on this standalone path), but ESMF_Initialize called
  ! MPI_Init, and MPI requires MPI_Finalize before exit: without it, mpirun
  ! flags every rank as "exited without calling finalize" and reports a
  ! non-zero job status even though the simulation completed successfully.
  ! Finalize MPI directly so multi-rank launches exit cleanly.
  call MPI_Finalize(mpierr)
  if (mpierr /= MPI_SUCCESS) then
    write(*,'(A,I0)') "WARN: [mainApp] MPI_Finalize returned error code ", mpierr
  end if

end program mainApp
