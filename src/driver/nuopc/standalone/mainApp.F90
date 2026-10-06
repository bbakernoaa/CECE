!> @file mainApp.F90
!> @brief Main application for CECE standalone execution.
!>
!> Updated to use proper NUOPC_Driver pattern (SingleModelOpenMPProto style).
!> Fixed multi-timestep execution by removing external time loop conflicts.

program mainApp

  use ESMF
  use NUOPC
  use mpi
  use driver, only: driver_SS => SetServices, set_cece_config_file

  implicit none

  integer :: rc, userRc, mpierr
  type(ESMF_GridComp) :: drvComp
  character(len=512) :: second_arg, cece_yaml_file

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
  call get_command_argument(2, second_arg)

  ! Legacy two-argument form (ignored.cfg config.yaml): the YAML is the
  ! second positional argument and the first is a driver config file that
  ! the standalone path never reads. Accept and ignore it so documented
  ! invocations keep working.
  if (len_trim(second_arg) > 0) then
    call get_command_argument(2, cece_yaml_file)
  end if

  ! Fallback if no arguments
  if (len_trim(cece_yaml_file) == 0) then
    call get_environment_variable("CECE_CONFIG", cece_yaml_file)
    if (len_trim(cece_yaml_file) == 0) then
      cece_yaml_file = "cece_config.yaml"
    end if
  end if

  ! Set the CECE YAML config path in the driver module
  call set_cece_config_file(trim(cece_yaml_file))

  write(*,'(A,A)') "INFO: [mainApp] CECE config file:   ", trim(cece_yaml_file)

  ! Create driver component
  drvComp = ESMF_GridCompCreate(name="driver", rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

  ! Set driver services
  call ESMF_GridCompSetServices(drvComp, driver_SS, userRc=userRc, rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  if (ESMF_LogFoundError(rcToCheck=userRc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

  ! Initialize driver
  call ESMF_LogWrite("[mainApp] Calling ESMF_GridCompInitialize...", &
    ESMF_LOGMSG_INFO, rc=rc)
  call ESMF_GridCompInitialize(drvComp, userRc=userRc, rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  if (ESMF_LogFoundError(rcToCheck=userRc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

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
  ! Run the component finalization phase so the CECE cap tears down the
  ! shared simulation through its model_label_Finalize specialization
  ! (flushing and closing the output writer) before the framework exits.
  write(*,'(A)') "INFO: [mainApp] Calling ESMF_GridCompFinalize..."
  call ESMF_GridCompFinalize(drvComp, userRc=userRc, rc=rc)
  if (ESMF_LogFoundError(rcToCheck=rc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)
  if (ESMF_LogFoundError(rcToCheck=userRc, msg=ESMF_LOGERR_PASSTHRU, &
    line=__LINE__, &
    file=__FILE__)) &
    call ESMF_Finalize(endflag=ESMF_END_ABORT)

  !-----------------------------------------------------------------------------

  call ESMF_LogWrite("mainApp FINISHED", ESMF_LOGMSG_INFO, rc=rc)

  write(*,'(A)') "INFO: [mainApp] CECE execution completed successfully"

  ! Tear down the ESMF framework while keeping MPI alive, then finalize MPI
  ! explicitly. ESMF_Finalize must be called once on each PET before the
  ! application exits; ESMF_END_KEEPMPI is the supported endflag for hosts
  ! (like this standalone app) that own the MPI lifecycle and call
  ! MPI_Finalize themselves afterwards.
  call ESMF_Finalize(endflag=ESMF_END_KEEPMPI, rc=rc)
  if (rc /= ESMF_SUCCESS) then
    write(*,'(A,I0)') "WARN: [mainApp] ESMF_Finalize reported rc=", rc
  end if

  call MPI_Finalize(mpierr)
  if (mpierr /= MPI_SUCCESS) then
    write(*,'(A,I0)') "WARN: [mainApp] MPI_Finalize returned error code ", mpierr
  end if

end program mainApp
