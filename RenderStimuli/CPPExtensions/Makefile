PYBIN=/home/kk/P3.10/bin/
CC=/usr/lib64/ccache/clang++
NVCC=/usr/local/cuda-12/bin/nvcc -allow-unsupported-compiler

O_DIRS = o/

PARAMETERS_O_CPU = -O3 -std=c++14 -fPIC -Wall -fopenmp=libomp
PARAMETERS_Linker_CPU = -shared -lm -lomp -lstdc++ -Wall

PARAMETERS_O_GPU= -O3 -std=c++14 -ccbin=$(CC) \
    -Xcompiler "-fPIC -Wall -fopenmp=libomp"
PARAMETERS_Linker_GPU=-Xcompiler "-shared -lm -lomp -lstdc++ -Wall"

name = TCopy
type = CPU

PYPOSTFIX := $(shell $(PYBIN)python3-config --extension-suffix)
PYBIND11INCLUDE := $(shell $(PYBIN)python3 -m pybind11 --includes)
PARAMETERS_O = $(PARAMETERS_O_CPU) $(PYBIND11INCLUDE) 
PARAMETERS_Linker = $(PARAMETERS_Linker_CPU)

so_file = Py$(name)$(type)$(PYPOSTFIX)
pyi_file = Py$(name)$(type).pyi
all: $(so_file)

$(O_DIRS)$(name)$(type).o: $(name)$(type).h $(name)$(type).cpp
	mkdir -p $(O_DIRS)
	$(CC) $(PARAMETERS_O) -c $(name)$(type).cpp -o $(O_DIRS)$(name)$(type).o

$(O_DIRS)Py$(name)$(type).o: $(name)$(type).h Py$(name)$(type).cpp
	mkdir -p $(O_DIRS)
	$(CC) $(PARAMETERS_O) -c Py$(name)$(type).cpp -o $(O_DIRS)Py$(name)$(type).o

$(so_file): $(O_DIRS)$(name)$(type).o $(O_DIRS)Py$(name)$(type).o
	$(CC) $(PARAMETERS_Linker) -o $(so_file) $(O_DIRS)$(name)$(type).o $(O_DIRS)Py$(name)$(type).o


#######################
clean:
	rm -rf $(O_DIRS)
	rm -f $(so_file)
	rm -f $(pyi_file)

