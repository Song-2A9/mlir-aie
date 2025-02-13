
//===- dft9_mmul0.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

typedef cint16 TT_DATA;
typedef cint16 TT_TWID;
typedef cacc64 TT_ACC;

static constexpr unsigned NUM_FFT = 4*7*16;
static constexpr unsigned NSAMP_I = 9*NUM_FFT; // 9 samples per transform
static constexpr unsigned DNSHIFT = 15;
static constexpr unsigned COEFF_DEPTH = 4*16;

alignas(16) TT_TWID coeff[COEFF_DEPTH] = {
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{32767,0},
{25102,-21063},
{5690,-32270},
{-16384,-28378},
{-30792,-11207},
{-30792,11207},
{-16384,28378},
{5690,32270},
{32767,0},
{5690,-32270},
{-30792,-11207},
{-16384,28378},
{25102,21063},
{25102,-21063},
{-16384,-28378},
{-30792,11207},
{32767,0},
{-16384,-28378},
{-16384,28378},
{32767,0},
{-16384,-28378},
{-16384,28378},
{32767,0},
{-16384,-28378},
{32767,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{25102,21063},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{5690,32270},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{-16384,28378},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0},
{0,0}
};

extern "C" {
    // void dft9_0(TT_DATA *input)
void dft9_0(int16_t *input)
{

    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
  
    // Define matrix multiplication:
    using MMUL = aie::mmul<1,4,8,TT_DATA,TT_TWID>;
    MMUL compute0;
    MMUL compute1;
    MMUL compute2;
    MMUL compute3;
    MMUL compute4;
    MMUL compute5;
    MMUL compute6;
    MMUL compute7;

    //cascade<TT_ACC>* __restrict acc_o;
    aie::accum<TT_ACC, 8> acc_o;
    
    // Vector registers for DFT coefficients:
    aie::vector<TT_TWID,32> vc0;
    aie::vector<TT_TWID,32> vc1;

    // Twiddles for first half of samples:
    vc0.insert<8>(0,aie::load_v<8>(&coeff[ 0]));
    vc0.insert<8>(1,aie::load_v<8>(&coeff[ 8]));
    vc0.insert<8>(2,aie::load_v<8>(&coeff[16]));
    vc0.insert<8>(3,aie::load_v<8>(&coeff[24]));

    vc1.insert<8>(0,aie::load_v<8>(&coeff[32]));
    vc1.insert<8>(1,aie::load_v<8>(&coeff[40]));
    vc1.insert<8>(2,aie::load_v<8>(&coeff[48]));
    vc1.insert<8>(3,aie::load_v<8>(&coeff[56]));

    // Lane buffers
    aie::vector<TT_DATA,32> data0;
    aie::vector<TT_DATA,32> data1;
    aie::vector<TT_DATA,32> data2;

    // Input buffer pointer:
    TT_DATA* __restrict ptr = (cint16_t *)input;

    // Loop over complete set of DFT-9 required for PFA-1008 transform:
    // --> We will run 7*16 = 112 transforms in total for a complete PFA-1008 (one kernel invocation)
    //
    // The loop body will perform 8 transforms since 9 x 8 = 72 is the smallest multiple of 8

    for (unsigned rr=0; rr < NUM_FFT/8; rr++)
        chess_prepare_for_pipelining
    {
        data0.insert<8>(0,aie::load_v<8>((ptr   )));
        data0.insert<8>(1,aie::load_v<8>((ptr+ 8)));
        data0.insert<8>(2,aie::load_v<8>((ptr+16)));
        data0.insert<8>(3,aie::load_v<8>((ptr+24)));
        data1.insert<8>(0,aie::load_v<8>((ptr+32)));
        data1.insert<8>(1,aie::load_v<8>((ptr+40)));
        data1.insert<8>(2,aie::load_v<8>((ptr+48)));
        data1.insert<8>(3,aie::load_v<8>((ptr+56)));
        data2.insert<8>(0,aie::load_v<8>((ptr+56)));
        data2.insert<8>(1,aie::load_v<8>((ptr+64)));

        compute0.mul(                        data0.extract<4>(0), vc0 );
        compute1.mul(                        data0.extract<4>(0), vc1 );
        compute2.mul( (aie::shuffle_down(data0,1)).extract<4>(2), vc0 );
        compute3.mul( (aie::shuffle_down(data0,1)).extract<4>(2), vc1 );
        compute4.mul( (aie::shuffle_down(data0,2)).extract<4>(4), vc0 );
        compute5.mul( (aie::shuffle_down(data0,2)).extract<4>(4), vc1 );
        compute6.mul( (aie::shuffle_down(data0,3)).extract<4>(6), vc0 );
        compute7.mul( (aie::shuffle_down(data0,3)).extract<4>(6), vc1 );

        put_mcd(v8cacc64(compute0.to_accum()));
        put_mcd(v8cacc64(compute1.to_accum()));
        put_mcd(v8cacc64(compute2.to_accum()));
        put_mcd(v8cacc64(compute3.to_accum()));
        put_mcd(v8cacc64(compute4.to_accum()));
        put_mcd(v8cacc64(compute5.to_accum()));
        put_mcd(v8cacc64(compute6.to_accum()));
        put_mcd(v8cacc64(compute7.to_accum()));

        compute0.mul(                        data1.extract<4>(1), vc0 );
        compute1.mul(                        data1.extract<4>(1), vc1 );
        compute2.mul( (aie::shuffle_down(data1,1)).extract<4>(3), vc0 );
        compute3.mul( (aie::shuffle_down(data1,1)).extract<4>(3), vc1 );
        compute4.mul( (aie::shuffle_down(data1,2)).extract<4>(5), vc0 );
        compute5.mul( (aie::shuffle_down(data1,2)).extract<4>(5), vc1 );
        compute6.mul( (aie::shuffle_down(data2,3)).extract<4>(1), vc0 );
        compute7.mul( (aie::shuffle_down(data2,3)).extract<4>(1), vc1 );

        put_mcd(v8cacc64(compute0.to_accum()));
        put_mcd(v8cacc64(compute1.to_accum()));
        put_mcd(v8cacc64(compute2.to_accum()));
        put_mcd(v8cacc64(compute3.to_accum()));
        put_mcd(v8cacc64(compute4.to_accum()));
        put_mcd(v8cacc64(compute5.to_accum()));
        put_mcd(v8cacc64(compute6.to_accum()));
        put_mcd(v8cacc64(compute7.to_accum()));

        // Update pointer:
        ptr += 72;
    }  
}

} // extern "C"


