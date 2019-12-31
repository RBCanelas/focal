program nbody
!! Focal example program: simple gravitational n-body simulator
!! Based on: https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/mini-nbody/cuda/nbody-orig.cu
!!
!! This program demonstrates the following Focal capabilities
!!  - Creating a default context and selecting a device
!!  - Loading kernels from compiled resource
!!  - Transferring data to the device
!!  - Out-of-order command queue with barriers and event-dependencies
!!  - Profiling kernels and buffer transfers
!!
use Focal
use iso_c_binding, longint=>c_int64_t
implicit none

! --------- Program configuration ---------
integer, parameter :: N = 5000 !3E5                 ! No. of bodies
integer, parameter :: blockSize = 16               ! Local work group size
real, parameter :: dt = 1                           ! Global time-step
integer, parameter :: Niter = 1000                  ! Number of iterations to perform
character(*), parameter :: cl_vendor = 'intel'     ! Vendor for which to create OpenCL context

! ---------Program variables  ---------
integer :: i, nBlock
integer(longint) :: kern1T, kern2T
real :: Tavg, perf
character(:), allocatable :: kernelSrc              ! Kernel source string
type(fclDevice), allocatable :: devices(:)          ! List of focal devices
type(fclProgram) :: prog                            ! Focal program object
type(fclKernel) :: kern1, kern2                     ! Focal kernel object
type(fclEvent) :: e

real, dimension(N) :: px, py, pz, vx, vy, vz
type(fclDeviceFloat) :: pxd, pyd, pzd, vxd, vyd, vzd

! Write header
write(*,*) ('-',i=1,72)
write(*,*) '                Focal example program: nbody simulator'
write(*,'(A,I6,A,I6,A)') '                 ( NBody: ',N,'    NIteration: ',Niter,')'
write(*,*) ('-',i=1,72)
write(*,*)

! Create context with nvidia platform
call fclSetDefaultContext(fclCreateContext(vendor=cl_vendor))

! Select device with most cores and create command queue
devices = fclFindDevices(sortBy='cores')

call fclSetDefaultCommandQ(fclCreateCommandQ(devices(1),enableProfiling=.true., &
           outOfOrderExec=.true.,blockingWrite=.false.))

write(*,*) '  Created OpenCL command queue on device: "',devices(1)%name,'"'
write(*,'(A,I6,A,I6,A,I4,A,A,A)') '    (', devices(1)%nComputeUnits,' cores, ', &
    devices(1)%global_memory/1024/1024,'MB, ', &
    devices(1)%clock_freq, 'MHz, ',&
    devices(1)%version,')'
write(*,*) ''

! Load kernels from file and compile
call fclGetKernelResource(kernelSrc)
prog = fclCompileProgram(kernelSrc)

call fclDumpBuildLog(prog,devices(1))

! Get kernel objects and set local/global work sizes
nBlock = (N+blockSize-1)/blockSize
kern1 = fclGetProgramKernel(prog,'bodyForces2',[nBlock*blockSize],[blockSize],&
                              profileSize=Niter)
kern2 = fclGetProgramKernel(prog,'integrateBodies',[nBlock*blockSize],[blockSize],&
                              profileSize=Niter)

! Initialise host array data
call random_number(vx)
call random_number(vy)
call random_number(vz)

call random_number(px)
call random_number(py)
call random_number(pz)

! Initialise device arrays
pxd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='pxd')
pyd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='pyd')
pzd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='pzd')
vxd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='vxd')
vyd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='vyd')
vzd = fclBufferFloat(N,read=.true.,write=.true.,profileSize=1,profileName='vzd')

! Copy data to device
pxd = px
pyd = py
pzd = pz
vxd = vx
vyd = vy
vzd = vz
call fclBarrier()
e = fclLastBarrierEvent

! Set kernel arguments once
call kern1%setArgs(N,dt,pxd,pyd,pzd,&
   fclLocalFloat(blockSize),fclLocalFloat(blockSize),fclLocalFloat(blockSize),&
   vxd,vyd,vzd)
call kern2%setArgs(N,dt,pxd,pyd,pzd,vxd,vyd,vzd)

! Main time-stepping loop
write(*,'(A)',advance='no') '  Time-stepping...'
do i=1,nIter

  call kern1%launchAfter(e)
  call kern2%launchAfter(fclLastKernelEvent)
  e = fclLastKernelEvent

end do

call fclWait()
write(*,*) ' done.'


! Calculate and print out profiling data
call fclDumpProfileData([kern1,kern2],devices(1))
call fclDumpProfileData([pxd,pyd,pzd,vxd,vyd,vzd])

! Calculate performance metric
kern1T = sum(fclGetEventDurations(kern1%profileEvents(1:Niter)))
kern2T = sum(fclGetEventDurations(kern2%profileEvents(1:Niter)))
Tavg = (kern1T+kern2T)/NIter

perf = N*N/Tavg

write(*,*) perf,' billion interactions per second'

end program nbody
! -----------------------------------------------------------------------------
