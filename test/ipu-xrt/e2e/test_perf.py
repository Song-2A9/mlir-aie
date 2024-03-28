# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import math
from pathlib import Path
import sys
import time
import warnings

from aie.compiler.aiecc.main import emit_design_kernel_json
from aie.compiler.util import (
    compile_with_vectorization,
    compile_without_vectorization,
    make_xclbin,
)
from aie.dialects import aie, aiex, builtin, pdl
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
from aie.dialects.aiex import TileArray
from aie.dialects.transform import (
    any_op_t,
    apply_registered_pass,
    get_parent_op,
    interpreter as interp,
)
from aie.dialects.transform.extras import named_sequence
from aie.dialects.transform.loop import loop_unroll
from aie.dialects.transform.structured import structured_match
from aie.extras.context import ExplicitlyManagedModule

# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref, scf, vector
from aie.extras.runtime.passes import Pipeline, run_pipeline

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
from aie.extras.util import find_ops, mlir_type_to_np_dtype
from aie.ir import StringAttr, UnitAttr
from aie.util import _to_js, extract_patches
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np
import pytest
from bfloat16 import bfloat16

from util import sliding_window

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


range_ = scf.range_
yield_ = scf.yield_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def test_tiled_nonsquare_tile_matrix_mult_vectorized_sugar(
    ctx: MLIRContext, workdir: Path
):
    M, K, N = 32, 32, 32

    @func.func(sym_visibility="private")
    def matmul_f32_f32(
        A: T.memref(M, K, T.f32()),
        B: T.memref(K, N, T.f32()),
        C: T.memref(M, N, T.f32()),
    ):
        linalg.matmul(A, B, C)

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul_f32_f32.emit(decl=True)
        tile_0_0 = aie.tile(0, 0)
        tile_0_2 = aie.tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(tile_0_2, (M, K), T.f32())
        buffer_0_2_b = aie.buffer(tile_0_2, (K, N), T.f32())
        # out
        buffer_0_2_c = aie.buffer(tile_0_2, (M, N), T.f32())

        # input
        lock_0_2_read_in_a = aie.lock(tile_0_2, init=1)
        lock_0_2_read_in_b = aie.lock(tile_0_2, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, init=0)
        lock_0_2_use_b = aie.lock(tile_0_2, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, init=0)

        aie.flow(tile_0_0, DMA, 0, tile_0_2, DMA, 0)
        aie.flow(tile_0_0, DMA, 1, tile_0_2, DMA, 1)
        aie.flow(tile_0_2, DMA, 0, tile_0_0, DMA, 0)

        @aie.mem(tile_0_2)
        def mem_0_2():
            aiex.receive_bd(0, lock_0_2_read_in_a, buffer_0_2_a, lock_0_2_use_a)
            aiex.receive_bd(1, lock_0_2_read_in_b, buffer_0_2_b, lock_0_2_use_b)
            aiex.send_bd(0, lock_0_2_write_out_c, buffer_0_2_c, lock_0_2_use_c)
            aie.end()

        @aie.core(tile_0_2)
        def core():
            with (
                aiex.hold_lock(lock_0_2_use_a, lock_0_2_read_in_a),
                aiex.hold_lock(lock_0_2_use_b, lock_0_2_read_in_b),
                aiex.hold_lock(lock_0_2_use_c, lock_0_2_write_out_c),
            ):
                linalg.fill(0, buffer_0_2_c)
                matmul_f32_f32(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_f32_f32.emit(force=True)

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addf"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, 16)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()

    affine_loops = run_pipeline(
        mod_aievec,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_unroll",
            debug_payload_root_tag="payload",
        )
        .canonicalize()
        .cse(),
    )

    super_vec = run_pipeline(
        affine_loops,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_super_vectorize",
            debug_payload_root_tag="payload",
        )
        .lower_affine(),
    )

    mod_aievec = find_ops(
        super_vec.operation,
        lambda x: "transform.target_tag" in x.attributes,
        single=True,
    )

    compile_with_vectorization(mod_aie, mod_aievec, workdir)
    # compile_without_vectorization(mod_aie, workdir)

    ipu_insts = aiex.ipu.get_prolog()
    xclbin_path = make_xclbin(mod_aie, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(M, K), (K, M), (M, N)], np.float32)

        col = 0
        lengths = [M * K, K * N, M * N]
        bd_id_direction = [MM2S, MM2S, S2MM]
        shim_channels = [0, 1, 0]
        for i, (len, bd_id_dir, shim_ch) in enumerate(
            zip(lengths, bd_id_direction, shim_channels)
        ):
            bd_id = buffer_idx = i
            writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                column=col, bd_id=bd_id, length=len
            )
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(
                    channel_dir=bd_id_dir,
                    channel_index=shim_ch,
                    column=col,
                    bd_id=bd_id,
                )
            )

        ipu_insts.extend(
            aiex.ipu.sync(column=col, channel=shim_channels[2], direction=S2MM)
        )

        xclbin.load_ipu_instructions(ipu_insts)

        wrap_A = np.asarray(views[0])
        wrap_B = np.asarray(views[1])
        wrap_C = np.asarray(views[2])

        A = np.ones((M, K)).astype(np.float32)
        B = np.ones((K, N)).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
                assert False


def test_4x4_broadcast_outer_product_matmult(ctx: MLIRContext, workdir: Path):

    cols = [0, 1, 2, 3]
    core_rows = [2, 3, 4, 5]
    rows = [0, 1, *core_rows]

    n_cols = n_rows = 4
    time_slices = 16

    M, K, N = 128, 128, 128
    m, k, n = M // n_cols, K // time_slices, N // n_rows

    shim_channels = {}

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        for col in cols:
            # broadcast out to the row
            tiles[col, 1].flow(tiles[col, 2:], source_annot="a", dest_annot="a")

        for col in cols:
            # broadcast out to the col
            tiles[col, 1].flow(tiles[:, col + 2], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            tiles[col, 1].rflow(tiles[col, 2:], source_annot="c", dest_annot="c")

        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            shim_channels[int(t.tile.col), 0] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), 1] = int(out_b_fl.source_channel)

            in_c_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), 2] = int(in_c_fl.dest_channel)

        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_flows = t.flows(filter_dest=True, dest_annot="c")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            A = aie.buffer(t.tile, (m, K), dtype=T.f32())
            B = aie.buffer(t.tile, (K, n), dtype=T.f32())
            C = aie.buffer(t.tile, (m, N), dtype=T.f32())

            in_A_lock = aie.lock(t.tile, init=1)
            out_A_lock = aie.lock(t.tile, init=0)
            in_B_lock = aie.lock(t.tile, init=1)
            out_B_lock = aie.lock(t.tile, init=0)

            in_C_locks = (
                aie.lock(t.tile, init=1),
                aie.lock(t.tile, init=0),
                aie.lock(t.tile, init=0),
                aie.lock(t.tile, init=0),
            )
            out_C_lock = aie.lock(t.tile, init=0)

            iter_A, *dims_A = extract_patches(A, patch_shape=(m, k))[1:]
            # use all 4 because this one is column order
            iter_B, *dims_B = extract_patches(B, patch_shape=(k, n))
            iter_C, *dims_C = extract_patches(C, patch_shape=(m, n))[1:]

            # rel_val=0 does some weird shit... either makes the stream run through all repeat_counts
            # or whatever i don't know ... but you get a stall?
            @aie.memtile_dma(t.tile)
            def mem():
                # fmt: off
                aiex.receive_bd(in_a_fl.dest_channel, in_A_lock, A, out_A_lock, loop=False)
                # acq_ge decrements, then rel_val reincrements and around around we go
                aiex.send_bd(out_a_fl.source_channel, out_A_lock, A,
                    acq_action=AcquireGreaterEqual, acq_val=1, rel_val=1,
                    dims=dims_A, len=m * k, iter=iter_A, repeat_count=time_slices - 1,
                )

                aiex.receive_bd(in_b_fl.dest_channel, in_B_lock, B, out_B_lock, loop=False)
                # acq_ge decrements, then rel_val reincrements and around around we go
                aiex.send_bd(out_b_fl.source_channel, out_B_lock, B,
                    acq_action=AcquireGreaterEqual, acq_val=1, rel_val=1,
                    dims=dims_B, len=k * n, iter=iter_B, repeat_count=time_slices - 1
                )

                for i, (in_c_fl, (in_lock, out_lock)) in enumerate(
                    zip(in_c_flows, sliding_window([*in_C_locks, out_C_lock], 2))
                ):
                    aiex.receive_bd(
                        in_c_fl.dest_channel, in_lock, C, out_lock,
                        acq_action=AcquireGreaterEqual,
                        dims=dims_C, len=m * n, offset=i * m, repeat_count=time_slices - 1,
                    )
                # fmt: on

                aiex.send_bd(
                    out_c_fl.source_channel,
                    out_C_lock,
                    C,
                    in_C_locks[0],
                    repeat_count=time_slices - 1,
                )

                aie.end()

        for t in list(tiles[cols, 2:]):
            in_a_fl = t.flows(filter_dest=True, dest_annot="a")
            in_b_fl = t.flows(filter_dest=True, dest_annot="b")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            in_a_prod_lock = aie.lock(t.tile, init=1)
            in_a_cons_lock = aie.lock(t.tile, init=0)
            in_b_prod_lock = aie.lock(t.tile, init=1)
            in_b_cons_lock = aie.lock(t.tile, init=0)
            out_c_prod_lock = aie.lock(t.tile, init=1)
            out_c_cons_lock = aie.lock(t.tile, init=0)

            a_buffer = t.buffer([(m, k)], T.f32(), annot="a")
            b_buffer = t.buffer([(k, n)], T.f32(), annot="b")
            c_buffer = t.buffer([(m, n)], T.f32(), annot="c")

            @aie.mem(t.tile)
            def mem():
                # fmt: off
                aiex.receive_bd(int(in_a_fl.dest_channel), in_a_prod_lock, a_buffer, in_a_cons_lock, repeat_count=time_slices - 1)
                aiex.receive_bd(int(in_b_fl.dest_channel), in_b_prod_lock, b_buffer, in_b_cons_lock, repeat_count=time_slices - 1)
                aiex.send_bd(int(out_c_fl.source_channel), out_c_cons_lock, c_buffer, out_c_prod_lock, repeat_count=time_slices - 1)
                # fmt: on

                aie.end()

            @aie.core(t.tile, elf_file="core_0_2.elf")
            def core():
                linalg.fill(0, c_buffer)
                for _ in range_(time_slices):
                    with (
                        aiex.hold_lock(in_a_cons_lock, in_a_prod_lock),
                        aiex.hold_lock(in_b_cons_lock, in_b_prod_lock),
                        aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                    ):
                        linalg.matmul(a_buffer, b_buffer, c_buffer)
                    yield_()

    # print(ctx.module)

    compile_without_vectorization(ctx.module, workdir, template_core=(0, 2))
    buffer_args = list(
        zip(
            [f"col_{c}_a" for c in cols],
            [f"col_{c}_b" for c in cols],
            [f"col_{c}_c" for c in cols],
        )
    )
    buffer_args = [a for col in buffer_args for a in col]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    ipu_insts = aiex.ipu.get_prolog()
    bd_id_direction = {0: MM2S, 1: MM2S, 2: S2MM}
    buffer_lengths = []
    for a in buffer_args:
        if "_a" in a:
            buffer_lengths.append((m, K))
        elif "_b" in a:
            buffer_lengths.append((K, n))
        elif "_c" in a:
            buffer_lengths.append((m, N))

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(buffer_lengths, np.float32)
        buffer_idx = -1
        for col in cols:
            for bd_id in [0, 1, 2]:
                buffer_idx += 1
                writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                    column=col,
                    bd_id=bd_id,
                    length=np.prod(buffer_lengths[buffer_idx]),
                )
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=bd_id_direction[bd_id],
                        channel_index=shim_channels[col, bd_id],
                        column=col,
                        bd_id=bd_id,
                        repeats=time_slices - 1,
                    )
                )

        for col in cols:
            bd_id = 2
            dest_channel = shim_channels[col, bd_id]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            As = np.random.randint(0, 10, (M, K)).astype(np.float32)
            Bs = np.random.randint(0, 3, (K, N)).astype(np.float32)
            Cs = np.zeros((m, N), dtype=np.float32)
            wraps = list(map(np.asarray, views))
            for col in cols:
                np.copyto(
                    wraps[3 * col + 0], As[m * col : m * (col + 1), :], casting="no"
                )
                np.copyto(
                    wraps[3 * col + 1], Bs[:, n * col : n * (col + 1)], casting="no"
                )
                np.copyto(wraps[3 * col + 2], Cs, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            correct = As @ Bs
            result = np.vstack([wraps[3 * col + 2] for col in cols])
            assert np.allclose(result, correct)


def test_4x4_broadcast_outer_product_matmult_vectorized(
    ctx: MLIRContext, workdir: Path
):

    cols = [0, 1, 2, 3]
    core_rows = [2, 3, 4, 5]
    rows = [0, 1, *core_rows]

    n_cols = n_rows = 4
    M, K, N = 32, 32, 32
    time_slices = 16

    m, k, n = M // n_cols, K // time_slices, N // n_rows
    while ((m * k) + (k * n) + (m * n)) * 4 >= (64 << 10):
        time_slices *= 2
        m, k, n = M // n_cols, K // time_slices, N // n_rows

    shim_channels = {}

    @func.func(sym_visibility="private")
    def matmul_f32_f32(
        A: T.memref(m, k, T.f32()),
        B: T.memref(k, n, T.f32()),
        C: T.memref(m, n, T.f32()),
    ):
        linalg.matmul(A, B, C)

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul_f32_f32.emit(decl=True)

        tiles = TileArray(cols, rows)
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        for col in cols:
            # broadcast out to the row
            tiles[col, 1].flow(tiles[col, 2:], source_annot="a", dest_annot="a")

        for col in cols:
            # broadcast out to the col
            tiles[col, 1].flow(tiles[:, col + 2], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            tiles[col, 1].rflow(tiles[col, 2:], source_annot="c", dest_annot="c")

        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            shim_channels[int(t.tile.col), 0] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), 1] = int(out_b_fl.source_channel)

            in_c_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), 2] = int(in_c_fl.dest_channel)

        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_flows = t.flows(filter_dest=True, dest_annot="c")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            A = aie.buffer(t.tile, (m, K), dtype=T.f32(), name=f"A_{int(t.tile.col)}")
            B = aie.buffer(t.tile, (K, n), dtype=T.f32(), name=f"B_{int(t.tile.col)}")
            C = aie.buffer(t.tile, (m, N), dtype=T.f32(), name=f"C_{int(t.tile.col)}")

            in_A_lock = aie.lock(t.tile, init=1)
            out_A_lock = aie.lock(t.tile, init=0)
            in_B_lock = aie.lock(t.tile, init=1)
            out_B_lock = aie.lock(t.tile, init=0)

            in_C_locks = (
                aie.lock(t.tile, init=1),
                aie.lock(t.tile, init=0),
                aie.lock(t.tile, init=0),
                aie.lock(t.tile, init=0),
            )
            out_C_lock = aie.lock(t.tile, init=0)

            iter_A, *dims_A = extract_patches(
                (m, K), patch_shape=(m, k), trailing_dims=3
            )
            # use all 4 because this one is column order
            iter_B, *dims_B = extract_patches(
                (K, n), patch_shape=(k, n), trailing_dims=4
            )
            iter_C, *dims_C = extract_patches(
                (m, N), patch_shape=(m, n), trailing_dims=3
            )

            # rel_val=0 does some weird shit... either makes the stream run through all repeat_counts
            # or whatever i don't know ... but you get a stall?
            @aie.memtile_dma(t.tile)
            def mem():
                # fmt: off
                aiex.receive_bd(in_a_fl.dest_channel, in_A_lock, A, out_A_lock, loop=False)
                # acq_ge decrements, then rel_val reincrements and around around we go
                aiex.send_bd(out_a_fl.source_channel, out_A_lock, A,
                             acq_action=AcquireGreaterEqual, acq_val=1, rel_val=1,
                             dims=dims_A, len=m * k, iter=iter_A, repeat_count=time_slices - 1,
                             )

                aiex.receive_bd(in_b_fl.dest_channel, in_B_lock, B, out_B_lock, loop=False)
                # acq_ge decrements, then rel_val reincrements and around around we go
                aiex.send_bd(out_b_fl.source_channel, out_B_lock, B,
                             acq_action=AcquireGreaterEqual, acq_val=1, rel_val=1,
                             dims=dims_B, len=k * n, iter=iter_B, repeat_count=time_slices - 1
                             )

                for i, (in_c_fl, (in_lock, out_lock)) in enumerate(
                        zip(in_c_flows, sliding_window([*in_C_locks, out_C_lock], 2))
                ):
                    aiex.receive_bd(
                        in_c_fl.dest_channel, in_lock, C, out_lock,
                        acq_action=AcquireGreaterEqual,
                        dims=dims_C, len=m * n, offset=i * m, repeat_count=time_slices - 1,
                    )
                # fmt: on

                aiex.send_bd(
                    out_c_fl.source_channel,
                    out_C_lock,
                    C,
                    in_C_locks[0],
                    repeat_count=time_slices - 1,
                )

                aie.end()

        for t in list(tiles[cols, 2:]):
            in_a_fl = t.flows(filter_dest=True, dest_annot="a")
            in_b_fl = t.flows(filter_dest=True, dest_annot="b")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            in_a_prod_lock = aie.lock(t.tile, init=1)
            in_a_cons_lock = aie.lock(t.tile, init=0)
            in_b_prod_lock = aie.lock(t.tile, init=1)
            in_b_cons_lock = aie.lock(t.tile, init=0)
            out_c_prod_lock = aie.lock(t.tile, init=1)
            out_c_cons_lock = aie.lock(t.tile, init=0)

            a_buffer = t.buffer(
                [(m, k)],
                T.f32(),
                annot="a",
                name=f"a_{int(t.tile.col)}_{int(t.tile.row)}",
            )
            b_buffer = t.buffer(
                [(k, n)],
                T.f32(),
                annot="b",
                name=f"b_{int(t.tile.col)}_{int(t.tile.row)}",
            )
            c_buffer = t.buffer(
                [(m, n)],
                T.f32(),
                annot="c",
                name=f"c_{int(t.tile.col)}_{int(t.tile.row)}",
            )

            @aie.mem(t.tile)
            def mem():
                # fmt: off
                aiex.receive_bd(int(in_a_fl.dest_channel), in_a_prod_lock, a_buffer, in_a_cons_lock, repeat_count=time_slices - 1)
                aiex.receive_bd(int(in_b_fl.dest_channel), in_b_prod_lock, b_buffer, in_b_cons_lock, repeat_count=time_slices - 1)
                aiex.send_bd(int(out_c_fl.source_channel), out_c_cons_lock, c_buffer, out_c_prod_lock, repeat_count=time_slices - 1)
                # fmt: on

                aie.end()

            @aie.core(t.tile, elf_file="core_0_2.elf")
            def core():
                linalg.fill(0, c_buffer)
                for _ in range_(time_slices):
                    with (
                        aiex.hold_lock(in_a_cons_lock, in_a_prod_lock),
                        aiex.hold_lock(in_b_cons_lock, in_b_prod_lock),
                        aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                    ):
                        matmul_f32_f32(a_buffer, b_buffer, c_buffer)
                    yield_()

    mod_aie.finish()

    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_f32_f32.emit(force=True)

    affine_unroll, super_vectorize = None, None

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():

        nonlocal affine_unroll, super_vectorize

        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addf"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, k)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            func = apply_registered_pass(any_op_t(), func, "lower-affine")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()
    interp.apply_named_sequence(payload, affine_unroll, mod_aievec.module)
    interp.apply_named_sequence(payload, super_vectorize, mod_aievec.module)

    compile_with_vectorization(mod_aie, payload, workdir, template_core=(0, 2))
    buffer_args = list(
        zip(
            [f"col_{c}_a" for c in cols],
            [f"col_{c}_b" for c in cols],
            [f"col_{c}_c" for c in cols],
        )
    )
    buffer_args = [a for col in buffer_args for a in col]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(mod_aie, workdir, kernel_json=kernel_json)

    ipu_insts = aiex.ipu.get_prolog()
    bd_id_direction = {0: MM2S, 1: MM2S, 2: S2MM}
    buffer_lengths = []
    for a in buffer_args:
        if "_a" in a:
            buffer_lengths.append((m, K))
        elif "_b" in a:
            buffer_lengths.append((K, n))
        elif "_c" in a:
            buffer_lengths.append((m, N))

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(buffer_lengths, np.float32)
        buffer_idx = -1
        for col in cols:
            for bd_id in [0, 1, 2]:
                buffer_idx += 1
                writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                    column=col,
                    bd_id=bd_id,
                    length=np.prod(buffer_lengths[buffer_idx]),
                )
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=bd_id_direction[bd_id],
                        channel_index=shim_channels[col, bd_id],
                        column=col,
                        bd_id=bd_id,
                        repeats=time_slices - 1,
                    )
                )

        for col in cols:
            bd_id = 2
            dest_channel = shim_channels[col, bd_id]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            As = np.random.randint(0, 10, (M, K)).astype(np.float32)
            Bs = np.random.randint(0, 3, (K, N)).astype(np.float32)
            Cs = np.zeros((m, N), dtype=np.float32)
            wraps = list(map(np.asarray, views))
            for col in cols:
                np.copyto(
                    wraps[3 * col + 0], As[m * col : m * (col + 1), :], casting="no"
                )
                np.copyto(
                    wraps[3 * col + 1], Bs[:, n * col : n * (col + 1)], casting="no"
                )
                np.copyto(wraps[3 * col + 2], Cs, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            correct = As @ Bs
            result = np.vstack([wraps[3 * col + 2] for col in cols])
            if not np.allclose(result, correct):
                print(correct - result)


def num_rows_cols_per_tile(
    m=16, k=16, n=16, dtype_size=4, data_memory_size=(512 << 10)
):
    row_size = m * k * dtype_size
    col_size = k * n * dtype_size
    inner_product_size = m * n * dtype_size

    def total_size(a, b):
        return row_size * a + col_size * b + (a * b) * inner_product_size

    a = b = 0
    # a more correct treatment would enable reuse on the memtile...
    # i.e. a and b not equal
    while total_size(a + 1, b + 1) <= data_memory_size:
        a += 1
        b += 1

    if a == 0 or b == 0:
        raise ValueError(
            f"couldn't fit even one {m},{k},{n} row/col on tile: {total_size(1, 1)}"
        )
    return a, b


def test_4x4_row_product_matmult(ctx: MLIRContext, workdir: Path):

    cols = [0, 1, 2, 3]
    core_rows = [2, 3, 4, 5]
    rows = [0, 1, *core_rows]

    M, K, N = 2048, 2048, 2048
    m, k, n = 16, 16, 16

    dtype_size = 4
    fat_A_rows_per_memtile, fat_B_cols_per_memtile = num_rows_cols_per_tile(
        m, K, n, dtype_size
    )
    if fat_A_rows_per_memtile < len(core_rows):
        warnings.warn(f"reducing row cores to match {fat_A_rows_per_memtile=}")
        core_rows = core_rows[:fat_A_rows_per_memtile]

    fat_A_rows_per_core_in_col = fat_A_rows_per_memtile // len(core_rows)
    fat_A_rows_per_memtile = fat_B_cols_per_memtile = fat_A_rows_per_core_in_col * len(
        core_rows
    )
    if fat_A_rows_per_memtile == 0:
        raise ValueError(f"not enough rows per memtile: {fat_A_rows_per_memtile=}")

    n_thin_slices_per_core, _ = num_rows_cols_per_tile(
        m, k, n, data_memory_size=(64 << 10)
    )
    thin_slice_stride = min(k * 2 ** math.floor(math.log2(n_thin_slices_per_core)), K)
    K_slices = K // thin_slice_stride

    total_n_fat_A_rows = M // m
    total_n_fat_B_cols = N // n
    n_rounds = int(
        math.ceil(
            total_n_fat_A_rows
            * total_n_fat_B_cols
            / (fat_A_rows_per_memtile * len(cols))
        )
    )

    core_tile_fat_A_group_offset = lambda i: fat_A_rows_per_core_in_col * i * (m * K)

    # since we're sending to each core in parallel (times 4 will equal total sends, which should equal total B sends)
    n_receive_thin_A_per_round = n_send_thin_A_per_round = (
        fat_A_rows_per_core_in_col * K_slices
    )
    n_receive_thin_B_per_round = n_send_thin_B_per_round = (
        fat_B_cols_per_memtile * K_slices
    )
    n_send_thin_C_per_round = n_receive_thin_C_per_round = (
        fat_A_rows_per_memtile * fat_B_cols_per_memtile
    )
    assert len(core_rows) * n_receive_thin_A_per_round == n_receive_thin_B_per_round

    # iterate through all the thin slices of all fat_A_rows
    # note iteration stride is not pushed through dims, it's linear addressing...
    if n_send_thin_A_per_round > 64:
        raise ValueError(f"{n_send_thin_A_per_round=} must <64")
    A_thin_slice_iter = (n_send_thin_A_per_round, m * thin_slice_stride)
    # gotta walk through fat_B_cols_per_memtile worth of columns, each with K_slices, each with k rows and n columns
    if n_send_thin_B_per_round > 64:
        raise ValueError(f"{n_send_thin_B_per_round=} must <64")
    B_thin_slice_iter = (n_send_thin_B_per_round, thin_slice_stride * n)

    shim_channels = {}
    arg_name_bd_id = {"a": 0, "b": 1, "c": 2}

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        for col in cols:
            # unique a per row on each col
            for row in core_rows:
                tiles[col, 1].flow(tiles[col, row], source_annot="a", dest_annot="a")
            # broadcast out b to the row
            tiles[col, 1].flow(tiles[col, core_rows], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            tiles[col, 1].rflow(tiles[col, core_rows], source_annot="c", dest_annot="c")

        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            shim_channels[int(t.tile.col), "a"] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), "b"] = int(out_b_fl.source_channel)

            in_c_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), "c"] = int(in_c_fl.dest_channel)

        # memtiles
        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_a_flows = t.flows(filter_source=True, source_annot="a")
            if len(core_rows) == 1:
                out_a_flows = [out_a_flows]
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_flows = t.flows(filter_dest=True, dest_annot="c")
            if len(core_rows) == 1:
                in_c_flows = [in_c_flows]
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            # stretch so that you can use extract_patches with the same dims below
            A = aie.buffer(t.tile, (fat_A_rows_per_memtile * m, K), dtype=T.f32())
            # send columns because we're sending columns from the host...
            B = aie.buffer(t.tile, (fat_B_cols_per_memtile * K, n), dtype=T.f32())
            C = aie.buffer(
                t.tile,
                (fat_A_rows_per_memtile * m, fat_B_cols_per_memtile * n),
                dtype=T.f32(),
            )

            in_A_lock = aie.lock(t.tile, init=1)
            out_A_lock = aie.lock(t.tile, init=0)
            in_B_lock = aie.lock(t.tile, init=1)
            out_B_lock = aie.lock(t.tile, init=0)

            in_C_locks = [aie.lock(t.tile, init=1)]
            for _ in core_rows[1:]:
                in_C_locks.append(aie.lock(t.tile, init=0))
            out_C_lock = aie.lock(t.tile, init=0)

            dims_A = extract_patches(A, patch_shape=(m, thin_slice_stride))
            # send columns because we're sending columns from the host...
            dims_B = extract_patches(B, patch_shape=(thin_slice_stride, n))
            dims_C = extract_patches(C, patch_shape=(m, n))

            @aie.memtile_dma(t.tile)
            def mem():
                aiex.receive_bd(
                    in_a_fl.dest_channel,
                    in_A_lock,
                    A,
                    out_A_lock,
                    acq_action=AcquireGreaterEqual,
                    acq_val=1,
                    # gonna send out to each core in the col
                    rel_val=len(core_rows) * n_send_thin_A_per_round,
                    # repeat_count=n_rounds - 1,
                )
                # this is gonna deadlock....
                for i, out_a_fl in enumerate(out_a_flows):
                    aiex.send_bd(
                        out_a_fl.source_channel,
                        out_A_lock,
                        A,
                        acq_action=AcquireGreaterEqual,
                        acq_val=1,
                        rel_val=1,
                        dims=dims_A,
                        len=m * thin_slice_stride,
                        offset=core_tile_fat_A_group_offset(i),
                        iter=A_thin_slice_iter,
                        # repeat_count=n_rounds * n_send_thin_A_per_round - 1,
                    )

                aiex.receive_bd(
                    in_b_fl.dest_channel,
                    in_B_lock,
                    B,
                    out_B_lock,
                    # repeat_count=n_rounds - 1,
                )
                # gonna broadcast to each core in the col
                aiex.send_bd(
                    out_b_fl.source_channel,
                    out_B_lock,
                    B,
                    acq_action=AcquireGreaterEqual,
                    acq_val=1,
                    rel_val=1,
                    dims=dims_B,
                    len=thin_slice_stride * n,
                    iter=B_thin_slice_iter,
                    # repeat_count=n_rounds * n_send_thin_B_per_round - 1,
                )

                for i, (in_c_fl, (in_lock, out_lock)) in enumerate(
                    zip(in_c_flows, sliding_window([*in_C_locks, out_C_lock], 2))
                ):
                    aiex.receive_bd(
                        in_c_fl.dest_channel,
                        in_lock,
                        C,
                        out_lock,
                        acq_action=AcquireGreaterEqual,
                        dims=dims_C,
                        len=m * n,
                        offset=i * m,
                        # repeat_count=n_rounds * n_receive_thin_C_per_round - 1,
                    )

                aiex.send_bd(
                    out_c_fl.source_channel,
                    out_C_lock,
                    C,
                    in_C_locks[0],
                    # repeat_count=n_rounds - 1,
                )

                aie.end()

        for t in list(tiles[cols, core_rows]):
            in_a_fl = t.flows(filter_dest=True, dest_annot="a")
            in_b_fl = t.flows(filter_dest=True, dest_annot="b")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            in_a_prod_lock = aie.lock(t.tile, init=1)
            in_a_cons_lock = aie.lock(t.tile, init=0)
            in_b_prod_lock = aie.lock(t.tile, init=1)
            in_b_cons_lock = aie.lock(t.tile, init=0)
            out_c_prod_lock = aie.lock(t.tile, init=1)
            out_c_cons_lock = aie.lock(t.tile, init=0)

            a_buffer = t.buffer([(m, thin_slice_stride)], T.f32(), annot="a")
            b_buffer = t.buffer([(thin_slice_stride, n)], T.f32(), annot="b")
            c_buffer = t.buffer([(m, n)], T.f32(), annot="c")

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_a_fl.dest_channel),
                    in_a_prod_lock,
                    a_buffer,
                    in_a_cons_lock,
                    # repeat_count=n_rounds * n_receive_thin_A_per_round - 1,
                )
                aiex.receive_bd(
                    int(in_b_fl.dest_channel),
                    in_b_prod_lock,
                    b_buffer,
                    in_b_cons_lock,
                    # repeat_count=n_rounds * n_receive_thin_B_per_round - 1,
                )
                aiex.send_bd(
                    int(out_c_fl.source_channel),
                    out_c_cons_lock,
                    c_buffer,
                    out_c_prod_lock,
                    # repeat_count=n_rounds * n_send_thin_C_per_round - 1,
                )

                aie.end()

            @aie.core(t.tile, elf_file="core_0_2.elf")
            def core():
                for _ in range_(
                    # n_rounds * fat_A_rows_per_core_in_col * fat_B_cols_per_memtile
                    2
                    << 20
                ):
                    linalg.fill(0, c_buffer)
                    for _ in range_(K_slices):
                        with (
                            aiex.hold_lock(in_a_cons_lock, in_a_prod_lock),
                            aiex.hold_lock(in_b_cons_lock, in_b_prod_lock),
                            aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                        ):
                            linalg.matmul(a_buffer, b_buffer, c_buffer)
                        yield_()
                    yield_()

    compile_without_vectorization(ctx.module, workdir, template_core=(0, 2))

    A = np.random.randint(0, 10, (M, K)).astype(np.float32)
    B = np.random.randint(0, 3, (K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    patches_A = extract_patches(A, patch_shape=(m, K))
    patches_B = extract_patches(B, patch_shape=(K, n))
    patches_C = extract_patches(C, patch_shape=(m, n))
    assert patches_A.shape[0] == M // m == total_n_fat_A_rows
    assert patches_B.shape[1] == N // n == total_n_fat_B_cols

    patches_A = np.broadcast_to(
        patches_A, (total_n_fat_A_rows, total_n_fat_B_cols, m, K)
    )
    patches_B = np.broadcast_to(
        patches_B, (total_n_fat_A_rows, total_n_fat_B_cols, K, n)
    )
    patches_A = patches_A.reshape(-1, m, K)
    patches_B = patches_B.reshape(-1, K, n)
    patches_C = patches_C.reshape(-1, m, n)

    assert fat_A_rows_per_memtile == fat_B_cols_per_memtile
    pad_patches_width = (
        math.ceil(patches_A.shape[0] / (fat_A_rows_per_memtile * len(cols)))
        * fat_A_rows_per_memtile
        * len(cols)
    )
    assert (pad_patches_width / (fat_A_rows_per_memtile * len(cols))) == n_rounds

    pad_patches_width -= patches_A.shape[0]
    patches_A = np.pad(
        patches_A,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_B = np.pad(
        patches_B,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_C = np.pad(
        patches_C,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )

    buffer_args = list(
        zip(
            [(f"{c}_a", (fat_A_rows_per_memtile, m, K)) for c in cols],
            [(f"{c}_b", (fat_B_cols_per_memtile, K, n)) for c in cols],
            [(f"{c}_c", (fat_A_rows_per_memtile, m, n)) for c in cols],
        )
    )
    buffer_args = dict([a for col in buffer_args for a in col])
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    arg_name_direction = {"a": MM2S, "b": MM2S, "c": S2MM}
    ipu_insts = aiex.ipu.get_prolog()
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np.float32)
        buffer_idx = -1
        for col in cols:
            for arg_name, bd_id in arg_name_bd_id.items():
                writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                    column=col,
                    bd_id=bd_id,
                    length=np.prod(buffer_args[f"{col}_{arg_name}"]),
                )
                buffer_idx += 1
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=arg_name_direction[arg_name],
                        channel_index=shim_channels[col, arg_name],
                        column=col,
                        bd_id=bd_id,
                        repeats=n_rounds - 1,
                    )
                )

        for col in cols:
            dest_channel = shim_channels[col, arg_name]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            wraps = list(map(np.asarray, views))
            for i, col in enumerate(cols):
                np.copyto(
                    wraps[3 * i + 0],
                    patches_A[
                        fat_A_rows_per_memtile * i : fat_A_rows_per_memtile * (i + 1)
                    ],
                    casting="no",
                )
                np.copyto(
                    wraps[3 * i + 1],
                    patches_B[
                        fat_B_cols_per_memtile * i : fat_B_cols_per_memtile * (i + 1)
                    ],
                    casting="no",
                )
                np.copyto(
                    wraps[3 * i + 2],
                    patches_C[
                        fat_A_rows_per_memtile * i : fat_A_rows_per_memtile * (i + 1)
                    ],
                    casting="no",
                )

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            start_time = time.monotonic_ns()
            xclbin.wait(30)
            end_time = time.monotonic_ns()
            xclbin.sync_buffers_from_device()
            total_time = (end_time - start_time) / 1e3
            print(f"{total_time=}us")

            correct = A @ B
            result = np.hstack([wraps[3 * i + 2] for i in range(len(cols))])
            print(result)


def test_4x4_row_product_matmult_vectorized(ctx: MLIRContext, workdir: Path):

    cols = [0, 1, 2, 3]
    core_rows = [2, 3, 4, 5]
    rows = [0, 1, *core_rows]

    M, K, N = 2048, 2048, 2048
    m, k, n = 16, 16, 16

    dtype_size = 4
    fat_A_rows_per_memtile, fat_B_cols_per_memtile = num_rows_cols_per_tile(
        m, K, n, dtype_size
    )
    if fat_A_rows_per_memtile < len(core_rows):
        warnings.warn(f"reducing row cores to match {fat_A_rows_per_memtile=}")
        core_rows = core_rows[:fat_A_rows_per_memtile]

    fat_A_rows_per_core_in_col = fat_A_rows_per_memtile // len(core_rows)
    fat_A_rows_per_memtile = fat_B_cols_per_memtile = fat_A_rows_per_core_in_col * len(
        core_rows
    )
    if fat_A_rows_per_memtile == 0:
        raise ValueError(f"not enough rows per memtile: {fat_A_rows_per_memtile=}")

    n_thin_slices_per_core, _ = num_rows_cols_per_tile(
        m, k, n, data_memory_size=(64 << 10)
    )
    thin_slice_stride = min(k * 2 ** math.floor(math.log2(n_thin_slices_per_core)), K)
    K_slices = K // thin_slice_stride

    total_n_fat_A_rows = M // m
    total_n_fat_B_cols = N // n
    n_rounds = int(
        math.ceil(
            total_n_fat_A_rows * total_n_fat_B_cols / (len(cols) * len(core_rows))
        )
    )

    core_tile_fat_A_group_offset = lambda i: fat_A_rows_per_core_in_col * i * (m * K)

    # since we're sending to each core in parallel (times 4 will equal total sends, which should equal total B sends)
    n_receive_thin_A_per_round = n_send_thin_A_per_round = (
        fat_A_rows_per_core_in_col * K_slices
    )
    n_receive_thin_B_per_round = n_send_thin_B_per_round = (
        fat_B_cols_per_memtile * K_slices
    )
    n_send_thin_C_per_round = n_receive_thin_C_per_round = (
        fat_A_rows_per_memtile * fat_B_cols_per_memtile
    )
    assert len(core_rows) * n_receive_thin_A_per_round == n_receive_thin_B_per_round

    # iterate through all the thin slices of all fat_A_rows
    # note iteration stride is not pushed through dims, it's linear addressing...
    if n_send_thin_A_per_round > 64:
        raise ValueError(f"{n_send_thin_A_per_round=} must <64")
    A_thin_slice_iter = (n_send_thin_A_per_round, m * thin_slice_stride)
    # gotta walk through fat_B_cols_per_memtile worth of columns, each with K_slices, each with k rows and n columns
    if n_send_thin_B_per_round > 64:
        raise ValueError(f"{n_send_thin_B_per_round=} must <64")
    B_thin_slice_iter = (n_send_thin_B_per_round, thin_slice_stride * n)

    shim_channels = {}
    arg_name_bd_id = {"a": 0, "b": 1, "c": 2}

    mod_aie = ExplicitlyManagedModule()

    @func.func(sym_visibility="private")
    def matmul(
        A: T.memref(m, thin_slice_stride, T.i32()),
        B: T.memref(thin_slice_stride, n, T.i32()),
        C: T.memref(m, n, T.i32()),
    ):
        linalg.matmul(A, B, C)

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul.emit(decl=True)
        tiles = TileArray(cols, rows)
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        for col in cols:
            # unique a per row on each col
            for row in core_rows:
                tiles[col, 1].flow(tiles[col, row], source_annot="a", dest_annot="a")
            # broadcast out b to the row
            tiles[col, 1].flow(tiles[col, core_rows], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            tiles[col, 1].rflow(tiles[col, core_rows], source_annot="c", dest_annot="c")

        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            shim_channels[int(t.tile.col), "a"] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), "b"] = int(out_b_fl.source_channel)

            in_c_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), "c"] = int(in_c_fl.dest_channel)

        # memtiles
        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_a_flows = t.flows(filter_source=True, source_annot="a")
            if len(core_rows) == 1:
                out_a_flows = [out_a_flows]
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_flows = t.flows(filter_dest=True, dest_annot="c")
            if len(core_rows) == 1:
                in_c_flows = [in_c_flows]
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            # stretch so that you can use extract_patches with the same dims below
            A = aie.buffer(t.tile, (fat_A_rows_per_memtile * m, K), dtype=T.i32())
            # send columns because we're sending columns from the host...
            B = aie.buffer(t.tile, (fat_B_cols_per_memtile * K, n), dtype=T.i32())
            C = aie.buffer(
                t.tile,
                (fat_A_rows_per_memtile * m, fat_B_cols_per_memtile * n),
                dtype=T.i32(),
            )

            in_A_lock = aie.lock(t.tile, init=1)
            out_A_lock = aie.lock(t.tile, init=0)
            in_B_lock = aie.lock(t.tile, init=1)
            out_B_lock = aie.lock(t.tile, init=0)

            in_C_locks = [aie.lock(t.tile, init=1)]
            for _ in core_rows[1:]:
                in_C_locks.append(aie.lock(t.tile, init=0))
            out_C_lock = aie.lock(t.tile, init=0)

            dims_A = extract_patches(A, patch_shape=(m, thin_slice_stride))
            # send columns because we're sending columns from the host...
            dims_B = extract_patches(B, patch_shape=(thin_slice_stride, n))
            dims_C = extract_patches(C, patch_shape=(m, n))

            @aie.memtile_dma(t.tile)
            def mem():
                aiex.receive_bd(
                    in_a_fl.dest_channel,
                    in_A_lock,
                    A,
                    out_A_lock,
                    acq_action=AcquireGreaterEqual,
                    acq_val=1,
                    # gonna send out to each core in the col
                    rel_val=len(core_rows) * n_send_thin_A_per_round,
                    # repeat_count=n_rounds - 1,
                )
                # this is gonna deadlock....
                for i, out_a_fl in enumerate(out_a_flows):
                    aiex.send_bd(
                        out_a_fl.source_channel,
                        out_A_lock,
                        A,
                        acq_action=AcquireGreaterEqual,
                        acq_val=1,
                        rel_val=1,
                        dims=dims_A,
                        len=m * thin_slice_stride,
                        offset=core_tile_fat_A_group_offset(i),
                        iter=A_thin_slice_iter,
                        # repeat_count=n_rounds * n_send_thin_A_per_round - 1,
                    )

                aiex.receive_bd(
                    in_b_fl.dest_channel,
                    in_B_lock,
                    B,
                    out_B_lock,
                    # repeat_count=n_rounds - 1,
                )
                # gonna broadcast to each core in the col
                aiex.send_bd(
                    out_b_fl.source_channel,
                    out_B_lock,
                    B,
                    acq_action=AcquireGreaterEqual,
                    acq_val=1,
                    rel_val=1,
                    dims=dims_B,
                    len=thin_slice_stride * n,
                    iter=B_thin_slice_iter,
                    # repeat_count=n_rounds * n_send_thin_B_per_round - 1,
                )

                for i, (in_c_fl, (in_lock, out_lock)) in enumerate(
                    zip(in_c_flows, sliding_window([*in_C_locks, out_C_lock], 2))
                ):
                    aiex.receive_bd(
                        in_c_fl.dest_channel,
                        in_lock,
                        C,
                        out_lock,
                        acq_action=AcquireGreaterEqual,
                        dims=dims_C,
                        len=m * n,
                        offset=i * m,
                        # repeat_count=n_rounds * n_receive_thin_C_per_round - 1,
                    )

                aiex.send_bd(
                    out_c_fl.source_channel,
                    out_C_lock,
                    C,
                    in_C_locks[0],
                    # repeat_count=n_rounds - 1,
                )

                aie.end()

        for t in list(tiles[cols, core_rows]):
            in_a_fl = t.flows(filter_dest=True, dest_annot="a")
            in_b_fl = t.flows(filter_dest=True, dest_annot="b")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            in_a_prod_lock = aie.lock(t.tile, init=1)
            in_a_cons_lock = aie.lock(t.tile, init=0)
            in_b_prod_lock = aie.lock(t.tile, init=1)
            in_b_cons_lock = aie.lock(t.tile, init=0)
            out_c_prod_lock = aie.lock(t.tile, init=1)
            out_c_cons_lock = aie.lock(t.tile, init=0)

            a_buffer = t.buffer([(m, thin_slice_stride)], T.i32(), annot="a")
            b_buffer = t.buffer([(thin_slice_stride, n)], T.i32(), annot="b")
            c_buffer = t.buffer([(m, n)], T.i32(), annot="c")

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_a_fl.dest_channel),
                    in_a_prod_lock,
                    a_buffer,
                    in_a_cons_lock,
                    # repeat_count=n_rounds * n_receive_thin_A_per_round - 1,
                )
                aiex.receive_bd(
                    int(in_b_fl.dest_channel),
                    in_b_prod_lock,
                    b_buffer,
                    in_b_cons_lock,
                    # repeat_count=n_rounds * n_receive_thin_B_per_round - 1,
                )
                aiex.send_bd(
                    int(out_c_fl.source_channel),
                    out_c_cons_lock,
                    c_buffer,
                    out_c_prod_lock,
                    # repeat_count=n_rounds * n_send_thin_C_per_round - 1,
                )

                aie.end()

            @aie.core(t.tile, elf_file="core_0_2.elf")
            def core():
                for _ in range_(
                    # n_rounds * fat_A_rows_per_core_in_col * fat_B_cols_per_memtile
                    2
                    << 20
                ):
                    linalg.fill(0, c_buffer)
                    for _ in range_(K_slices):
                        with (
                            aiex.hold_lock(in_a_cons_lock, in_a_prod_lock),
                            aiex.hold_lock(in_b_cons_lock, in_b_prod_lock),
                            aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                        ):
                            # linalg.matmul(a_buffer, b_buffer, c_buffer)
                            matmul(a_buffer, b_buffer, c_buffer)
                        yield_()
                    yield_()

    mod_aie.finish()

    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul.emit(force=True)

    affine_unroll, super_vectorize = None, None

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():

        nonlocal affine_unroll, super_vectorize

        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addf"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, k)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            func = apply_registered_pass(any_op_t(), func, "lower-affine")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()
    interp.apply_named_sequence(payload, affine_unroll, mod_aievec.module)
    interp.apply_named_sequence(payload, super_vectorize, mod_aievec.module)

    compile_with_vectorization(mod_aie, payload, workdir, template_core=(0, 2))

    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 3, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    patches_A = extract_patches(A, patch_shape=(m, K))
    patches_B = extract_patches(B, patch_shape=(K, n))
    patches_C = extract_patches(C, patch_shape=(m, n))
    assert patches_A.shape[0] == M // m == total_n_fat_A_rows
    assert patches_B.shape[1] == N // n == total_n_fat_B_cols

    patches_A = np.broadcast_to(
        patches_A, (total_n_fat_A_rows, total_n_fat_B_cols, m, K)
    )
    patches_B = np.broadcast_to(
        patches_B, (total_n_fat_A_rows, total_n_fat_B_cols, K, n)
    )
    patches_A = patches_A.reshape(-1, m, K)
    patches_B = patches_B.reshape(-1, K, n)
    patches_C = patches_C.reshape(-1, m, n)

    assert fat_A_rows_per_memtile == fat_B_cols_per_memtile
    pad_patches_width = (
        math.ceil(patches_A.shape[0] / (fat_A_rows_per_memtile * len(cols)))
        * fat_A_rows_per_memtile
        * len(cols)
    )
    assert (pad_patches_width / (fat_A_rows_per_memtile * len(cols))) == n_rounds

    pad_patches_width -= patches_A.shape[0]
    patches_A = np.pad(
        patches_A,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_B = np.pad(
        patches_B,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_C = np.pad(
        patches_C,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )

    buffer_args = list(
        zip(
            [(f"{c}_a", (fat_A_rows_per_memtile, m, K)) for c in cols],
            [(f"{c}_b", (fat_B_cols_per_memtile, K, n)) for c in cols],
            [(f"{c}_c", (fat_A_rows_per_memtile, m, n)) for c in cols],
        )
    )
    buffer_args = dict([a for col in buffer_args for a in col])
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(mod_aie, workdir, kernel_json=kernel_json)

    arg_name_direction = {"a": MM2S, "b": MM2S, "c": S2MM}
    ipu_insts = aiex.ipu.get_prolog()
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np.int32)
        buffer_idx = -1
        for col in cols:
            for arg_name, bd_id in arg_name_bd_id.items():
                writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                    column=col,
                    bd_id=bd_id,
                    length=np.prod(buffer_args[f"{col}_{arg_name}"]),
                )
                buffer_idx += 1
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=arg_name_direction[arg_name],
                        channel_index=shim_channels[col, arg_name],
                        column=col,
                        bd_id=bd_id,
                        repeats=n_rounds - 1,
                    )
                )

        for col in cols:
            dest_channel = shim_channels[col, arg_name]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            wraps = list(map(np.asarray, views))
            for i, col in enumerate(cols):
                np.copyto(
                    wraps[3 * i + 0],
                    patches_A[
                        fat_A_rows_per_memtile * i : fat_A_rows_per_memtile * (i + 1)
                    ],
                    casting="no",
                )
                np.copyto(
                    wraps[3 * i + 1],
                    patches_B[
                        fat_B_cols_per_memtile * i : fat_B_cols_per_memtile * (i + 1)
                    ],
                    casting="no",
                )
                np.copyto(
                    wraps[3 * i + 2],
                    patches_C[
                        fat_A_rows_per_memtile * i : fat_A_rows_per_memtile * (i + 1)
                    ],
                    casting="no",
                )

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            start_time = time.monotonic_ns()
            xclbin.wait(30)
            end_time = time.monotonic_ns()
            xclbin.sync_buffers_from_device()
            total_time = (end_time - start_time) / 1e3
            print(f"{total_time=}us")

            # correct = A @ B
            # result = np.hstack([wraps[3 * i + 2] for i in range(len(cols))])
            # print(result)


def view(source, shape, dtype=None, shift=0):
    if dtype is None:
        dtype = source.type.element_type
    byte_width_dtype = dtype.width // 8
    byte_shift = shift * byte_width_dtype
    byte_shift = arith.constant(byte_shift, index=True)
    return memref.view(T.memref(*shape, dtype), source, byte_shift, [])


def test_5x4_inner_product_matmult_vectorized(ctx: MLIRContext, workdir: Path):

    M, K, N = 512, 512, 512
    m, k, n = 16, 16, 16

    cols = [0, 1, 2, 3, 4]
    memtile_cols = [1, 2, 3, 4]
    partition_start_col = cols[0]
    start_columns = [partition_start_col]

    core_rows = [2, 3, 4, 5]
    rows = [0, 1, *core_rows]

    dtype = T.bf16()
    byte_width_dtype = dtype.width // 8
    (
        fat_A_rows_per_memtile_per_round,
        fat_B_cols_per_memtile_per_round,
    ) = num_rows_cols_per_tile(m, K, n, byte_width_dtype)
    if fat_A_rows_per_memtile_per_round < len(core_rows):
        warnings.warn(
            f"reducing row cores to match {fat_A_rows_per_memtile_per_round=}"
        )
        core_rows = core_rows[:fat_A_rows_per_memtile_per_round]
    assert fat_A_rows_per_memtile_per_round == fat_B_cols_per_memtile_per_round

    # since this is a 5x4
    n_memtiles = len(memtile_cols)
    n_cores_per_memtile = len(cols)
    fat_A_rows_per_core_in_col_per_round = (
        fat_A_rows_per_memtile_per_round // n_cores_per_memtile
    )
    fat_A_rows_per_memtile_per_round = fat_B_cols_per_memtile_per_round = (
        fat_A_rows_per_core_in_col_per_round * n_cores_per_memtile
    )
    if fat_A_rows_per_memtile_per_round == 0:
        raise ValueError(
            f"not enough rows per memtile: {fat_A_rows_per_memtile_per_round=}"
        )
    total_n_fat_A_rows = M // m
    total_n_fat_B_cols = N // n
    n_rounds_send_from_shim_to_memtile = int(
        math.ceil(total_n_fat_A_rows / (n_memtiles * fat_A_rows_per_memtile_per_round))
    )

    # these are the same because 1xfat A + 1xfat B == 1xfat_AB
    fat_AB_rows_per_memtile_per_round = fat_A_rows_per_memtile_per_round
    assert (
        fat_AB_rows_per_memtile_per_round % len(cols) == 0
    ), f"not enough work for all cores per memtile: {fat_AB_rows_per_memtile_per_round=}"
    fat_AB_rows_per_core_in_col_per_round = fat_A_rows_per_core_in_col_per_round
    total_fat_AB_rows_per_memtile = (
        fat_AB_rows_per_memtile_per_round * n_rounds_send_from_shim_to_memtile
    )

    K_slices = K // k

    core_tile_fat_AB_group_offset = lambda i: i * (m * k + k * n)

    n_receive_thin_AB_per_round = n_send_thin_AB_per_round = (
        fat_AB_rows_per_core_in_col_per_round * K_slices
    )
    n_send_thin_C_per_round = n_receive_thin_C_per_round = n_receive_thin_AB_per_round
    assert len(core_rows) * n_receive_thin_AB_per_round

    # iterate through all the thin slices of all fat_A_rows
    # note iteration stride is not pushed through dims, it's linear addressing...
    if n_send_thin_AB_per_round > 64:
        raise ValueError(f"{n_send_thin_AB_per_round=} must <64")
    AB_thin_slice_iter = (n_send_thin_AB_per_round, m * k + k * n)

    shim_channels = {}
    arg_name_bd_id = {"ab": 0, "c": 1}
    arg_name_direction = {"ab": MM2S, "c": S2MM}

    mod_aie = ExplicitlyManagedModule()

    @func.func(sym_visibility="private")
    def matmul(
        A: T.memref(m, k, dtype),
        B: T.memref(k, n, dtype),
        C: T.memref(m, n, dtype),
    ):
        linalg.matmul(A, B, C)

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul.emit(decl=True)
        tiles = TileArray(cols, rows)
        # shim to mem flows (except col 0 which doesn't have dma)
        tiles[memtile_cols, 0].flow(
            tiles[memtile_cols, 1], source_annot="ab", dest_annot="ab"
        )
        tiles[memtile_cols, 1].flow(
            tiles[memtile_cols, 0], source_annot="c", dest_annot="c"
        )

        # col1 -> 5 connections, 1 to each col in row 2
        # col2 -> 5 connections, 1 to each col in row 3
        # ...
        for i, col in enumerate(memtile_cols):
            for t in tiles[cols, 2 + i]:
                tiles[col, 1].flow(t, source_annot="ab", dest_annot="ab")
                tiles[col, 1].rflow(t, source_annot="c", dest_annot="c")

        # just so the names don't leak
        @lambda f: f()
        def _():
            for t in tiles[memtile_cols, 0]:
                out_ab_fl = t.flows(filter_source=True, source_annot="ab", single=True)
                shim_channels[int(t.tile.col), "ab"] = int(out_ab_fl.source_channel)
                in_c_fl = t.flows(filter_dest=True, dest_annot="c")
                shim_channels[int(t.tile.col), "c"] = int(in_c_fl.dest_channel)

        # just so the names don't leak
        @lambda f: f()
        def _():
            for t in tiles[memtile_cols, 1]:
                in_ab_fl = t.flows(filter_dest=True, dest_annot="ab", single=True)
                out_ab_flows = t.flows(filter_source=True, source_annot="ab")
                if len(core_rows) == 1:
                    out_ab_flows = [out_ab_flows]
                in_c_flows = t.flows(filter_dest=True, dest_annot="c")
                if len(core_rows) == 1:
                    in_c_flows = [in_c_flows]
                out_c_fl = t.flows(filter_source=True, source_annot="c")

                # interleave A and B
                AB = aie.buffer(
                    t.tile,
                    (fat_AB_rows_per_memtile_per_round, (m * K + K * n)),
                    dtype=dtype,
                    name=f"AB_{int(t.tile.col)}",
                )
                C = aie.buffer(
                    t.tile,
                    # bad reuse - would be A_tiles_per_memtile_per_round * B_tiles_per_memtile_per_round
                    # if i did all-to-all
                    # as-is (this is just inner-product), we just bom
                    (fat_AB_rows_per_memtile_per_round, (m * n)),
                    dtype=dtype,
                    name=f"C_{int(t.tile.col)}",
                )

                # read the full patches_AB and then stride
                n_AB_consumers = len(out_ab_flows)
                in_AB_lock = aie.lock(
                    t.tile,
                    init=n_AB_consumers,
                    sym_name=f"in_AB_lock_{int(t.tile.col)}",
                )
                out_AB_lock = aie.lock(
                    t.tile, init=0, sym_name=f"out_AB_lock_{int(t.tile.col)}"
                )
                in_C_locks = [
                    aie.lock(
                        t.tile, init=1, sym_name=f"in_C_lock_{int(t.tile.col)}_from_{c}"
                    )
                    for c in cols
                ]
                out_C_lock = aie.lock(
                    t.tile, init=0, sym_name=f"out_C_lock_{int(t.tile.col)}"
                )

                @aie.memtile_dma(t.tile)
                def mem():
                    assert (
                        fat_AB_rows_per_memtile_per_round % len(out_ab_flows) == 0
                    ), f"not enough work for all cores per memtile: {fat_AB_rows_per_memtile_per_round=}"
                    aiex.receive_bd(
                        in_ab_fl.dest_channel,
                        in_AB_lock,
                        AB,
                        out_AB_lock,
                        acq_action=AcquireGreaterEqual,
                        acq_val=n_AB_consumers,
                        rel_val=2 * n_AB_consumers,
                    )
                    for i, out_ab_fl in enumerate(out_ab_flows):
                        bd = aiex.send_bd(
                            out_ab_fl.source_channel,
                            out_AB_lock,
                            AB,
                            in_AB_lock,
                            acq_action=AcquireGreaterEqual,
                            acq_val=2,
                            # don't release anything (this is a no-op)
                            rel_val=0,
                            len=(m * k + k * n),
                            offset=core_tile_fat_AB_group_offset(i),
                            iter=AB_thin_slice_iter,
                            num_bds=2,
                        )

                        @aie.another_bd(bd)
                        def sync_bd():
                            # acquire after all consumers have started (ie when the sema is down to 0 after all have decremented)
                            aie.use_lock(out_AB_lock, Acquire, value=0)
                            aie.dma_bd(AB, len=0)
                            aie.use_lock(in_AB_lock, Release, value=1)

                    for i, (in_c_fl, (in_lock, out_lock)) in enumerate(
                        zip(in_c_flows, sliding_window([*in_C_locks, out_C_lock], 2))
                    ):
                        aiex.receive_bd(
                            in_c_fl.dest_channel,
                            in_lock,
                            C,
                            out_lock,
                            acq_action=AcquireGreaterEqual,
                            len=m * n,
                            offset=i * m * n,
                            iter=(n_receive_thin_C_per_round, m * n),
                        )

                    aiex.send_bd(
                        out_c_fl.source_channel,
                        out_C_lock,
                        C,
                        in_C_locks[0],
                    )

                    aie.end()

        for t in list(tiles[cols, core_rows]):
            col, row = int(t.tile.col), int(t.tile.row)
            in_ab_fl = t.flows(filter_dest=True, dest_annot="ab")
            out_c_fl = t.flows(filter_source=True, source_annot="c")

            in_ab_prod_lock = aie.lock(
                t.tile, init=1, sym_name=f"in_ab_prod_lock_{col}_{row}"
            )
            in_ab_cons_lock = aie.lock(
                t.tile, init=0, sym_name=f"in_ab_cons_lock_{col}_{row}"
            )
            out_c_prod_lock = aie.lock(
                t.tile, init=1, sym_name=f"out_c_prod_lock_{col}_{row}"
            )
            out_c_cons_lock = aie.lock(
                t.tile, init=0, sym_name=f"out_c_cons_lock_{col}_{row}"
            )

            # weird type just for taking memref.view below
            ab_buffer = t.buffer(
                [((m * k + k * n) * byte_width_dtype,)],
                T.i8(),
                annot="ab",
            )
            c_buffer = t.buffer([(m, n)], dtype, annot="c")

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_ab_fl.dest_channel),
                    in_ab_prod_lock,
                    ab_buffer,
                    in_ab_cons_lock,
                )
                aiex.send_bd(
                    int(out_c_fl.source_channel),
                    out_c_cons_lock,
                    c_buffer,
                    out_c_prod_lock,
                )

                aie.end()

            @aie.core(t.tile, elf_file="core_0_2.elf")
            def core():
                a_buffer = view(ab_buffer, (m, k), dtype=dtype)
                b_buffer = view(ab_buffer, (k, n), dtype=dtype, shift=m * k)
                for _ in range_(2 << 20):
                    linalg.fill(0, c_buffer)
                    for _ in range_(K_slices):
                        with (
                            aiex.hold_lock(in_ab_cons_lock, in_ab_prod_lock),
                            aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                        ):
                            matmul(a_buffer, b_buffer, c_buffer)
                        yield_()
                    yield_()

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul.emit(force=True)

    affine_unroll, super_vectorize = None, None

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():

        nonlocal affine_unroll, super_vectorize

        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addf"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, k)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            func = apply_registered_pass(any_op_t(), func, "lower-affine")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()
    interp.apply_named_sequence(payload, affine_unroll, mod_aievec.module)
    interp.apply_named_sequence(payload, super_vectorize, mod_aievec.module)

    compile_with_vectorization(
        mod_aie,
        payload,
        workdir,
        template_core=(0, 2),
        partition_start_col=partition_start_col,
    )

    np_dtype = np.float16
    A = np.random.randint(0, 10, (M, K)).astype(np_dtype)
    B = np.random.randint(0, 3, (K, N)).astype(np_dtype)
    C = np.zeros((M, N), dtype=np_dtype)

    patches_A = extract_patches(A, patch_shape=(m, K))
    patches_B = extract_patches(B, patch_shape=(K, n))
    patches_C = extract_patches(C, patch_shape=(m, n))
    assert patches_A.shape[0] == M // m == total_n_fat_A_rows
    assert patches_B.shape[1] == N // n == total_n_fat_B_cols

    patches_A = np.broadcast_to(
        patches_A, (total_n_fat_A_rows, total_n_fat_B_cols, m, K)
    )
    patches_B = np.broadcast_to(
        patches_B, (total_n_fat_A_rows, total_n_fat_B_cols, K, n)
    )
    patches_A = patches_A.reshape(-1, m, K)
    patches_B = patches_B.reshape(-1, K, n)
    patches_C = patches_C.reshape(-1, m, n)

    assert fat_A_rows_per_memtile_per_round == fat_B_cols_per_memtile_per_round
    pad_patches_width = (
        math.ceil(patches_A.shape[0] / (fat_A_rows_per_memtile_per_round * n_memtiles))
        * fat_A_rows_per_memtile_per_round
        * n_memtiles
    )

    pad_patches_width -= patches_A.shape[0]
    patches_A = np.pad(
        patches_A,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_B = np.pad(
        patches_B,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )
    patches_C = np.pad(
        patches_C,
        pad_width=[
            (0, pad_patches_width),
            (0, 0),
            (0, 0),
        ],
    )

    # stack then column ordering (i guess stacking extends???)
    patches_AB = np.vstack((patches_A, patches_B.swapaxes(-2, -1))).reshape(
        (-1,), order="F"
    )
    patches_AB = patches_AB.reshape(-1, m * K + K * n)

    buffer_args = list(
        zip(
            [
                (
                    f"{c}_ab",
                    (
                        patches_AB.shape[0] // n_memtiles,
                        m * K + K * n,
                    ),
                )
                for c in memtile_cols
            ],
            [
                (
                    f"{c}_c",
                    (patches_C.shape[0] // n_memtiles, m, n),
                )
                for c in memtile_cols
            ],
        )
    )
    buffer_args = dict([a for col in buffer_args for a in col])
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(
        mod_aie, workdir, kernel_json=kernel_json, start_columns=start_columns
    )

    ipu_insts = aiex.ipu.get_prolog()
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np_dtype)
        buffer_idx = -1
        for col in memtile_cols:
            for arg_name, bd_id in arg_name_bd_id.items():
                writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                    column=col,
                    bd_id=bd_id,
                    length=np.prod(buffer_args[f"{col}_{arg_name}"]),
                )
                buffer_idx += 1
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=arg_name_direction[arg_name],
                        channel_index=shim_channels[col, arg_name],
                        column=col,
                        bd_id=bd_id,
                        repeats=n_rounds_send_from_shim_to_memtile - 1,
                    )
                )

        for col in memtile_cols:
            dest_channel = shim_channels[col, arg_name]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)
        n_arg_types = len(arg_name_direction)

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            # wraps = list(map(np.asarray, views))
            # for i, _ in enumerate(memtile_cols):
            #     np.copyto(
            #         wraps[n_arg_types * i + 0],
            #         padded_AB[
            #             total_fat_AB_rows_per_memtile
            #             * i : total_fat_AB_rows_per_memtile
            #             * (i + 1)
            #         ],
            #         casting="no",
            #     )
            #     np.copyto(
            #         wraps[n_arg_types * i + 1],
            #         patches_C[
            #             total_fat_AB_rows_per_memtile
            #             * i : total_fat_AB_rows_per_memtile
            #             * (i + 1)
            #         ],
            #         casting="no",
            #     )

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            start_time = time.monotonic_ns()
            xclbin.wait(30)
            end_time = time.monotonic_ns()
            xclbin.sync_buffers_from_device()
            total_time = (end_time - start_time) / 1e3
            print(f"{total_time=}us")
