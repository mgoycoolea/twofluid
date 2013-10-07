






! All the constants in the program go here, will be useful for equations
module constants
  implicit none
  private
  public dp
  ! dp is the modern way of assigning double precision
  ! you declare them as real(dp) :: a
  ! also if you input a number always follow by _dp
  ! e.g. a = 1_dp
  integer, parameter :: dp = kind(0.d0)
end module constants


! All equations go here
module equations
  use constants, only: dp
  implicit none
  private
  public 

contains

end module equations

module read_input
  use constants, only: dp
  implicit none
  private

  open(9,file='input_twofluid',status='old')
  read(9,*)
  read(9,*) N0
  read(9,*)
  read(9,*) tau
  read(9,*)
  read(9,*) vol
  read(9,*)
  read(9,*) ENin0
  read(9,*)
  read(9,*) dt
  read(9,*)
  read(9,*) rsteps
  read(9,*)
  read(9,*) timetot
  read(9,*)
  read(9,*) R0
  read(9,*)
  read(9,*) Rdomain
  read(9,*)
  read(9,*) savepoint
  read(9,*)
  read(9,*) savefreq
  read(9,*)
  read(9,*) Bext
  close(9)

  OPEN(30,file='input_print',STATUS='replace')
  write(30,*) N0
  write(30,*) tau
  write(30,*) vol
  write(30,*) ENin0
  write(30,*) dt
  write(30,*) rsteps
  write(30,*) timetot
  write(30,*) R0
  write(30,*) Rdomain
  write(30,*) savepoint
  write(30,*) savefreq
  write(30,*) Bext
  close(30)
  


end module read_input


module variables
  use constants, only: dp
  implicit none
  private
  public 

  real(dp), allocatable, dimension(:) ::& 
       Ne1, Ne2, Te1, Te2, Vre1, Vre2 &
       Ni1, Ni2, Ti1, Ti2, Vri1, Vri2 &


contains

end module equations
