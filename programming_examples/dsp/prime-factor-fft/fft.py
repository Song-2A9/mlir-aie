# passthrough_kernel/passthrough_kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2

try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    vector_size = int(sys.argv[2])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")

# Define tensor types
line_size = vector_size // 4
# line_type = np.ndarray[(line_size,), np.dtype[np.uint8]]
line_type = np.ndarray[(line_size,), np.dtype[np.int16]]
vector_type = np.ndarray[(vector_size,), np.dtype[np.uint8]]

# Dataflow with ObjectFifos
of_in = ObjectFifo(line_type, name="in")
of_0to1 = ObjectFifo(line_type, name="b0to1")
of_1to2 = ObjectFifo(line_type, name="b1to2")
of_2to3 = ObjectFifo(line_type, name="b2to3")
of_out = ObjectFifo(line_type, name="out")

# External, binary kernel definition
passthrough_fn = Kernel(
    "passThroughLine",
    "passThrough.cc.o",
    [line_type, line_type, np.int32],
)

dft9_mmul0_fn = Kernel(
    "dft9_0",
    "dft9_mmul0.cc.o",
    [line_type],
)

dft9_mmul1_fn = Kernel(
    "dft9_1",
    "dft9_mmul1.cc.o",
    [line_type],
)

dft9_mmul2_fn = Kernel(
    "dft9_2",
    "dft9_mmul2.cc.o",
    [line_type],
)

dft9_mmul3_fn = Kernel(
    "dft9_3",
    "dft9_mmul3.cc.o",
    [line_type],
)

# Task for the core to perform
def core_fn(of_in, of_out, passThroughLine):
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    passThroughLine(elemIn, elemOut, line_size)
    of_in.release(1)
    of_out.release(1)

def core_dft9_mmul0_fn(of_in, of_out, dft9_0): # of_out is placeholder
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    dft9_0(elemIn)
    of_in.release(1)
    of_out.release(1)

def core_dft9_mmul1_fn(of_in, of_in2, of_out, dft9_1): # of_in and of_out are placeholders
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    elemIn2 = of_in2.acquire(1)
    dft9_1(elemIn2)
    of_in2.release(1)
    of_in.release(1)
    of_out.release(1)

def core_dft9_mmul2_fn(of_in, of_in2, of_out, dft9_2):  # of_in and of_out are placeholders
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    elemIn2 = of_in2.acquire(1)
    dft9_2(elemIn2)
    of_in2.release(1)
    of_in.release(1)
    of_out.release(1)

def core_dft9_mmul3_fn(of_in, of_out, dft9_3): # of_in is a placeholder
    elemIn = of_in.acquire(1)
    elemOut = of_out.acquire(1)
    dft9_3(elemOut)
    of_in.release(1)
    of_out.release(1)

def core_fn2(of_in, of_in2, of_out, passThroughLine):
    elemOut = of_out.acquire(1)
    elemIn = of_in.acquire(1)
    elemIn2 = of_in2.acquire(1)
    passThroughLine(elemIn, elemOut, line_size)
    of_in2.release(1)
    of_in.release(1)
    of_out.release(1)


# Create workers to perform the tasks
# dtf9_kk0 = Worker(core_fn, [of_in.cons(), of_0to1.prod(), passthrough_fn])
dtf9_kk0 = Worker(core_dft9_mmul0_fn, [of_in.cons(), of_0to1.prod(), dft9_mmul0_fn])
# dtf9_kk1 = Worker(core_fn2, [of_0to1.cons(), of_in.cons(), of_1to2.prod(), passthrough_fn])
dtf9_kk1 = Worker(core_dft9_mmul1_fn, [of_0to1.cons(), of_in.cons(), of_1to2.prod(), dft9_mmul1_fn])
# dtf9_kk2 = Worker(core_fn2, [of_1to2.cons(), of_in.cons(), of_2to3.prod(), passthrough_fn])
dtf9_kk2 = Worker(core_dft9_mmul2_fn, [of_1to2.cons(), of_in.cons(), of_2to3.prod(), dft9_mmul2_fn])
# dtf9_kk3 = Worker(core_fn, [of_2to3.cons(), of_out.prod(), passthrough_fn])
dtf9_kk3 = Worker(core_dft9_mmul3_fn, [of_2to3.cons(), of_out.prod(), dft9_mmul3_fn])
my_workers = [dtf9_kk0, dtf9_kk1, dtf9_kk2, dtf9_kk3]

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
    rt.start(*my_workers)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
