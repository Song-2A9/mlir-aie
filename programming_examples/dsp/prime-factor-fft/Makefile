##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

device = npu
targetname = fft
#VPATH := ${srcdir}/../../../aie_kernels/generic
data_size = 4096
trace_size = 8192
PASSTHROUGH_SIZE = ${data_size}

aie_py_src=${targetname}.py
use_alt?=0

.PHONY: all dft9 template clean

all: build/final_${data_size}.xclbin

build/fft_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${data_size} 0 > $@

build/fft_trace_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${data_size} ${trace_size} > $@

# List of AIE single-core kernel sources (without extension)
DFT9_MMUL_SRCS := dft9_mmul0 dft9_mmul1 dft9_mmul2 dft9_mmul3

# Build object list from these sources
DFT9_MMUL_OBJS := $(patsubst %,build/%.cpp.o,$(DFT9_MMUL_SRCS))

dft9: $(DFT9_MMUL_OBJS)

build/%.cpp.o: aie_kernels/dft9/%.cpp
	mkdir -p $(@D)
ifeq ($(device),npu)
	cd $(@D) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2_FLAGS) -c $(abspath $<) -o $(@F)
else
	echo "Device type not supported"
endif
	
build/final_${data_size}.xclbin: build/fft_${data_size}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.txt $(<:%=../%)

build/final_trace_${data_size}.xclbin: build/fft_trace_${data_size}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.txt $(<:%=../%)


${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE 

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size}
	../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > trace_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean:
	rm -rf build _build ${targetname}*.exe
