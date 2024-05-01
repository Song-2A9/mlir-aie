//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license, !hsa
// RUN: xchesscc_wrapper aie -c %S/kernel.cc
// RUN: %PYTHON aiecc.py --aiesim --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module @test_chess_04_deprecated_shim_dma_precompiled_kernel{
  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t71 = aie.tile(7, 1)
  %t70 = aie.tile(7, 0)

  %buf_a_ping = aie.buffer(%t73) {sym_name = "a_ping" } : memref<256xi32>
  %buf_a_pong = aie.buffer(%t73) {sym_name = "a_pong" } : memref<256xi32>
  %buf_b_ping = aie.buffer(%t73) {sym_name = "b_ping" } : memref<256xi32>
  %buf_b_pong = aie.buffer(%t73) {sym_name = "b_pong" } : memref<256xi32>

  %lock_a_ping = aie.lock(%t73, 3) // a_ping
  %lock_a_pong = aie.lock(%t73, 4) // a_pong
  %lock_b_ping = aie.lock(%t73, 5) // b_ping
  %lock_b_pong = aie.lock(%t73, 6) // b_pong

  func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> ()

  %c13 = aie.core(%t73) {
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    
    scf.for %iv = %lb to %ub step %step {
      
      aie.use_lock(%lock_a_ping, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_b_ping, "Acquire", 0) // acquire for write
      func.call @func(%buf_a_ping, %buf_b_ping) : (memref<256xi32>, memref<256xi32>) -> ()
      aie.use_lock(%lock_a_ping, "Release", 0) // release for write
      aie.use_lock(%lock_b_ping, "Release", 1) // release for read

      aie.use_lock(%lock_a_pong, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_b_pong, "Acquire", 0) // acquire for write
      func.call @func(%buf_a_pong, %buf_b_pong) : (memref<256xi32>, memref<256xi32>) -> ()
      aie.use_lock(%lock_a_pong, "Release", 0) // release for write
      aie.use_lock(%lock_b_pong, "Release", 1) // release for read
      
    }

    aie.end
  } { link_with="kernel.o" }

  // Tile DMA
  %m73 = aie.mem(%t73) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%lock_a_ping, "Acquire", 0)
      aie.dma_bd(%buf_a_ping : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_a_ping, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_a_pong, "Acquire", 0)
      aie.dma_bd(%buf_a_pong : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_a_pong, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%lock_b_ping, "Acquire", 1)
      aie.dma_bd(%buf_b_ping : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_b_ping, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_b_pong, "Acquire", 1)
      aie.dma_bd(%buf_b_pong : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_b_pong, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // DDR buffer
  %buffer_in  = aie.external_buffer {sym_name = "input_buffer" } : memref<512 x i32>
  %buffer_out = aie.external_buffer {sym_name = "output_buffer" } : memref<512 x i32>
  %lock1 = aie.lock(%t70, 1) {sym_name = "input_lock" }
  %lock2 = aie.lock(%t70, 2) {sym_name = "output_lock" }

  // Shim DMA connection to kernel
  aie.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
  aie.flow(%t73, "DMA" : 1, %t70, "DMA" : 0)

  // Shim DMA loads large buffer to local memory
  %dma = aie.shim_dma(%t70) {
      aie.dma_start(MM2S, 0, ^bd0, ^dma)
    ^dma:
      aie.dma_start(S2MM, 0, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_in : memref<512 x i32>, 0, 512)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^bd1:
      aie.use_lock(%lock2, Acquire, 1)
      aie.dma_bd(%buffer_out : memref<512 x i32>, 0, 512)
      aie.use_lock(%lock2, Release, 0)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }


}
