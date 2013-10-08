program twofluid
  use read_input_mod
  use init_variables
  use equations_gas_conservative
  use TVDLF
  implicit none


end program twofluid


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


module init_variables
  use constants, only: dp
  implicit none
  public

  real(dp), allocatable, dimension(:) ::&
       Ne1, Ne2, Te1, Te2, Vre1, Vre2, &
       Ni1, Ni2, Ti1, Ti2, Vri1, Vri2

contains

  subroutine initial_conditions()
    write(*,*) 'I do something'
  end subroutine initial_conditions
end module init_variables



! All equations go here
module equations_gas_conservative
  use constants
  implicit none
  private
  !public 

contains
  real(dp) function dNe_dt(var_string, R, VRE, DLOGE, GRAD_VRE, GRAD_DLOGE) result(output)
    real(dp), intent(in) :: R, VRE, DLOGE, GRAD_VRE, GRAD_DLOGE
    character(len=*), intent(in) :: var_string
  end function dNe_dt

end module equations_gas_conservative


module TVDLF
  use constants, only: dp
  implicit none
  private
  !public 
  
end module TVDLF


module read_input_mod
  use constants, only: dp
  implicit none
  public

contains
  
  subroutine read_input()
    real(dp) :: N0, tau, vol, ENin0, dt, rsteps, timetot, R0,&
         Rdomain, savepoint, savefreq, Bext

    open(9,file='input_twofluid',status='old')
    read(9,*) N0
    read(9,*) tau
    read(9,*) vol
    read(9,*) ENin0
    read(9,*) dt
    read(9,*) rsteps
    read(9,*) timetot
    read(9,*) R0
    read(9,*) Rdomain
    read(9,*) savepoint
    read(9,*) savefreq
    read(9,*) Bext
    close(9)

    OPEN(30,file='print_input',STATUS='replace')
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
  end subroutine read_input

end module read_input_mod







