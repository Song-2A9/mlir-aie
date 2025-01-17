# dma_transpose/dma_transpose.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern




# this resembles the buffer A data layout and transformations
def my_passthrough(m, k, K):

    # large K must be divisible by small k 
    assert K % k == 0

    # those define the API sizes for the MatMul kernels
    r = 4
    s = 5

    # assertions for m and k which should be divisible by the API sizes
    assert m % r == 0
    assert k % s == 0
    
        
    # compute tile is m x k (small tile)
    comp_tile_ty = np.ndarray[(m, k), np.dtype[np.int32]]

    # memory tile is m x K (larger tile)
    mem_tile_ty = np.ndarray[(m, K), np.dtype[np.int32]]
    

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)


            # AIE-array data movement with object fifos

            # Input 
            of_in_shim_to_mem = object_fifo(
                "shim_to_mem",
                ShimTile,
                MemTile,
                2,
                mem_tile_ty,
                None,
                # Emulate the pre-tiling of m*k in software (test.cpp)
                # because data will conme from DDR in row-major format.
                # Thus, this is comment out for our testing.
                # [
                #     [
                #         (m, k), 
                #         (K // k, m*k), 
                #         (k, 1),
                #     ]
                # ],
            )

            of_in_mem_to_comp = object_fifo(
                "mem_to_comp",
                MemTile,
                ComputeTile,
                2,
                comp_tile_ty,
                # 4D transformation in MemTile (MM2S)
                # Assumes that the "higher" MemTile size
                # defines the 4D transformation
                [ 
                    (K // k, m*k),
                    (k // s, s),
                    (m, k), 
                    (s, 1),
                ],
                # 3D transformation in CompTile (S2MM)
                [
                    [
                        (k//s, r*s), 
                        (m//r, r*k), 
                        (r*s, 1),
                    ]
                ],
                
            )


            # links mem to comp
            object_fifo_link(of_in_shim_to_mem, of_in_mem_to_comp)


            # Output
            of_out_comp_to_mem = object_fifo(
                "comp_to_mem",
                ComputeTile,
                MemTile,
                2,
                comp_tile_ty,
            )

            of_out_mem_to_shim = object_fifo(
                "mem_to_shim",
                MemTile,
                ShimTile,
                2, 
                mem_tile_ty
            )

            # links comp to mem
            object_fifo_link(of_out_comp_to_mem, of_out_mem_to_shim)


            # <<<<<<<<<<<<<<<<< This one didn't really work!! >>>>>>>>>>>>>>>>>>
            # links mem to comp (in) and comp to mem (out)
            # such that compute tile can pass (see below)
            # object_fifo_link(of_in_mem_to_comp, of_out_comp_to_mem)



            # Compute tile just passes, doesn't do any operation
            @core(ComputeTile)
            def core_body():
                for _ in range_(sys.maxsize):
                    
                    for _ in range_(K//k):
                        elem_in = of_in_mem_to_comp.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = of_out_comp_to_mem.acquire(ObjectFifoPort.Produce, 1)
                        for i in range_(m):
                            for j in range_(k):
                                elem_out[i, j] = elem_in[i, j]
                                # pass

                        of_in_mem_to_comp.release(ObjectFifoPort.Consume, 1)
                        of_out_comp_to_mem.release(ObjectFifoPort.Produce, 1)


            # set the runtime type as 1D array
            runtime_ty = np.ndarray[(m*K,), np.dtype[np.int32]]

            # To/from AIE-array data movement
            @runtime_sequence(runtime_ty, runtime_ty, runtime_ty)
            def sequence(A, B, C):
                
                npu_dma_memcpy_nd(
                    metadata=of_in_shim_to_mem,
                    bd_id=1,
                    mem=A,
                    sizes=[1, 1, 1, m*K],
                )

                npu_dma_memcpy_nd(
                    metadata=of_out_mem_to_shim, 
                    bd_id=0, 
                    mem=C, 
                    sizes=[1, 1, 1, m*K])
                
                # wait only on output since input will have completed before output
                dma_wait(of_out_mem_to_shim)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="m, k, K", type=int, nargs="*", default=[32, 64, 256])
    args = p.parse_args()

    if len(args.dims) != 3:
        print(
            "ERROR: Must provide all 3 dimensions", file=sys.stderr
        )
        exit(-1)

    my_passthrough(
        m=args.dims[0],
        k=args.dims[1],
        K=args.dims[2],
    )
