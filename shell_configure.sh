#!/bin/bash

hdf5path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
fftwpath=/usr/lib/x86_64-linux-gnu/fftw

#python3 configure.py --prob=disk --coord=cylindrical --eos=adiabatic --flux=hllc --ndustfluids=3 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

python3 configure.py --prob=disk_3d_mhd_dust -b --coord=spherical_polar --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd -sts

#python3 configure.py --prob=disk_3d_mhd_dust -b --coord=spherical_polar --ndustfluids=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python3 configure.py --prob=disk_RWI_2D --coord=cylindrical --ndustfluids=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_VSI --coord=spherical_polar --ndustfluids=1 --nghost=2 --nscalars=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_dust_diffusion -mpi --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd --coord=cylindrical

#python3 configure.py --prob=disk_dust_drift --coord=cylindrical --ndustfluids=1 --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_planet_dust_cylindrical --coord=cylindrical --ndustfluids=1 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_planet_dust_spherical --coord=spherical_polar --ndustfluids=1 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_multiple_planets_dust_cylindrical --coord=cylindrical --ndustfluids=1 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_streaming_cylindrical --ndustfluids=1 --coord=cylindrical --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=disk_streaming_spherical --coord=spherical_polar --ndustfluids=1 --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dmr_dust --ndustfluids=3 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd --nscalars=0

#python3 configure.py --prob=dust_NSH --ndustfluids=2 --eos=isothermal --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python3 configure.py --prob=dust_collision --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dust_collision_different_Ts --ndustfluids=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dust_diffusion -mpi --eos=isothermal --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dust_diffusion -mpi --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double --coord=cylindrical -omp --cxx=g++-simd

#python3 configure.py --prob=dust_inelastic_collision --ndustfluids=2 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dust_squaredrag --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dusty_shock --ndustfluids=3 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=dusty_soundwave --ndustfluids=1 --nghost=2 --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=gr_torus -g -b --coord=kerr-schild --flux=hlle --nghost 4 -hdf5 --hdf5_path=${hdf5path} -mpi -omp --cxx=g++-simd

#python3 configure.py --prob=hb3 -b --eos=isothermal -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --ndustfluids=2 -omp --cxx=g++-simd

#python3 configure.py --prob=kh_dust --ndustfluids=0 --nscalars=1 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=rt --ndustfluids=0 --nscalars=0 --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=ssheet --flux=hlle --eos=isothermal --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=ssheet_planet_dust --eos=isothermal --nghost=2 --ndustfluids=0 --nscalars=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=ssheet_planet_dust_selfgravity --eos=isothermal --nghost=2 --ndustfluids=0 --nscalars=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd -fft --fftw_path=${fftwpath} --grav=fft

#python3 configure.py --prob=ssheet_RWI_dust --eos=isothermal --nghost=2 --ndustfluids=0 --nscalars=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=ssheet_RWI_dust_gap --eos=isothermal --nghost=2 --ndustfluids=0 --nscalars=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double -omp --cxx=g++-simd

#python3 configure.py --prob=streaming_eigen --ndustfluids=1 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python3 configure.py --prob=streaming_eigen_2dust --ndustfluids=2 --eos=isothermal --nghost=4 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=hlle -omp --cxx=g++-simd

#python3 configure.py --prob=streaming_nonlinear --ndustfluids=1 --eos=isothermal --nghost=3 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python3 configure.py --prob=streaming_stratified --ndustfluids=1 --eos=isothermal --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python3 configure.py --prob=streaming_stratified_cartesian --ndustfluids=1 --eos=isothermal --nghost=2 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double --flux=roe -omp --cxx=g++-simd

#python3 configure.py --prob=turb -hdf5 --hdf5_path=${hdf5path} -fft --fftw_path=${fftwpath} -omp --cxx=g++-simd

#python3 configure.py --prob=poisson -hdf5 --hdf5_path=${hdf5path} -h5double -fft --fftw_path=${fftwpath} --grav=fft --eos=isothermal -mpi -omp --cxx=g++-simd

python3 configure.py --prob=hgb -b -hdf5 --hdf5_path=${hdf5path} -h5double --eos=isothermal -mpi -omp --cxx=g++-simd

make clean
make -j 10
