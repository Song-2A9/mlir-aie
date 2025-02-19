//===- shimtileDMA_test_bad.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK:   error: 'aie.tile' op number of output DMA channel exceeded!

module @shimtileDMA_channels {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.external_buffer : memref<16xi32>
        %lock0 = aie.lock(%tile20, 0)
        %buff1 = aie.external_buffer : memref<16xi32>
        %lock1 = aie.lock(%tile20, 1)
        %buff2 = aie.external_buffer : memref<16xi32>
        %lock2 = aie.lock(%tile20, 2)

        aie.objectfifo @objfifo (%tile20, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem12 = aie.shim_dma(%tile20) {
            %dma1 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
        ^bb1:
            aie.use_lock(%lock0, Acquire, 1)
            aie.dma_bd(%buff0 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock0, Release, 0)
            aie.next_bd ^bb2
        ^bb2:
            aie.use_lock(%lock1, Acquire, 1)
            aie.dma_bd(%buff1 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock1, Release, 0)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(MM2S, 1, ^bb4, ^bb5)
        ^bb4:
            aie.use_lock(%lock2, Acquire, 0)
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock2, Release, 1)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
