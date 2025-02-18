//===- link_repeat_count_test.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @memtileRepeat {
// CHECK:   aie.device(npu1) {
// CHECK:     memref.global "public" @of3_cons : memref<16xi32>
// CHECK:     memref.global "public" @of3 : memref<16xi32>
// CHECK:     memref.global "public" @of2_cons : memref<32xi32>
// CHECK:     memref.global "public" @of2 : memref<32xi32>
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     memref.global "public" @of0_cons : memref<32xi32>
// CHECK:     memref.global "public" @of0 : memref<32xi32>
// CHECK:     %{{.*}}tile_1_0 = aie.tile(1, 0)
// CHECK:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK:     %{{.*}}tile_2_1 = aie.tile(2, 1)
// CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK:     %of3_cons_prod_lock = aie.lock(%{{.*}}tile_1_0, 2) {init = 1 : i32, sym_name = "of3_cons_prod_lock"}
// CHECK:     %of3_cons_cons_lock = aie.lock(%{{.*}}tile_1_0, 3) {init = 0 : i32, sym_name = "of3_cons_cons_lock"}
// CHECK:     %of2_cons_buff_0 = aie.buffer(%{{.*}}tile_2_1) {sym_name = "of2_cons_buff_0"} : memref<32xi32> 
// CHECK:     %of2_cons_prod_lock = aie.lock(%{{.*}}tile_2_1, 0) {init = 1 : i32, sym_name = "of2_cons_prod_lock"}
// CHECK:     %of2_cons_cons_lock = aie.lock(%{{.*}}tile_2_1, 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock"}
// CHECK:     %of2_buff_0 = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of2_buff_0"} : memref<32xi32> 
// CHECK:     %of2_prod_lock = aie.lock(%{{.*}}tile_3_3, 0) {init = 3 : i32, sym_name = "of2_prod_lock"}
// CHECK:     %of2_cons_lock = aie.lock(%{{.*}}tile_3_3, 1) {init = 0 : i32, sym_name = "of2_cons_lock"}
// CHECK:     %of1_cons_buff_0 = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_prod_lock = aie.lock(%{{.*}}tile_1_2, 0) {init = 1 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %of1_cons_cons_lock = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %of0_cons_buff_0 = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of0_cons_buff_0"} : memref<32xi32> 
// CHECK:     %of0_cons_prod_lock = aie.lock(%{{.*}}tile_1_1, 0) {init = 3 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:     %of0_cons_cons_lock = aie.lock(%{{.*}}tile_1_1, 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:     %of0_prod_lock = aie.lock(%{{.*}}tile_1_0, 0) {init = 1 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %of0_cons_lock = aie.lock(%{{.*}}tile_1_0, 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:     aie.flow(%{{.*}}tile_1_0, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_2_1, DMA : 0, %{{.*}}tile_1_0, DMA : 0)
// CHECK:     aie.shim_dma_allocation @of0(MM2S, 0, 1)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of0_cons_prod_lock, AcquireGreaterEqual, 3)
// CHECK:       aie.dma_bd(%of0_cons_buff_0 : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%of0_cons_cons_lock, Release, 3)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 2)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%of0_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_cons_buff_0 : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%of0_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of2_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_buff_0 : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%of2_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_cons_buff_0 : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%of2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%of2_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_cons_buff_0 : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%of2_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of3(S2MM, 0, 1)
// CHECK:   }
// CHECK: }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile21 = aie.tile(2, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile10, {%tile11}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of1 (%tile11, {%tile12}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of0] -> [@of1] ([] [])

    aie.objectfifo @of2 (%tile33, {%tile21}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of3 (%tile21, {%tile10}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of2] -> [@of3] ([] [])
 }
}
