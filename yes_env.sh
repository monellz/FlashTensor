spack load gcc@12.2.0
spack load cuda@12.1.1/odp466h
spack load llvm@18.1.2
spack load cmake@3.27.9/w5bq4at
spack load ninja@1.11.1/antp5sf
spack load openmpi@4.1.5%gcc@12.2.0/cext65e
spack load python@3.10.13/blgzgdt

# force using g++ instead of clang in llvm
export CC=$(which gcc)
export CXX=$(which g++)

# python -m venv asuka_venv
source asuka_venv/bin/activate

export OCTAVE="srun -p octave --gres=gpu:1 -A public"
export TWILLS="srun -p twills --gres=gpu:h100:1"
