//===- dft9_mmul3.cc -------------------------------------------------*- C++
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

void dft9_3(TT_DATA *output)
{

    aie::set_rounding(aie::rounding_mode::symmetric_inf);
    aie::set_saturation(aie::saturation_mode::saturate);
  
    //cascade<TT_ACC>* __restrict acc_o;
    aie::accum<TT_ACC, 8> acc_i;
    
    // Output buffer pointer:
    // auto itw = aie::begin_vector<8>(sig_o);
    TT_DATA* __restrict itw = output;

    aie::vector<TT_DATA,8> data0;
    aie::vector<TT_DATA,8> data1;
    aie::vector<TT_DATA,8> data2;
    aie::vector<TT_DATA,8> data3;
    aie::vector<TT_DATA,8> data4;
    aie::vector<TT_DATA,8> data5;
    aie::vector<TT_DATA,8> data6;
    aie::vector<TT_DATA,8> data7;
    aie::vector<TT_DATA,8> data8;
    aie::vector<TT_DATA,8> data9;
    aie::vector<TT_DATA,8> dataA;
    aie::vector<TT_DATA,8> dataB;
    aie::vector<TT_DATA,8> dataC;
    aie::vector<TT_DATA,8> dataD;
    aie::vector<TT_DATA,8> dataE;
    aie::vector<TT_DATA,8> dataF;
  
    for (unsigned rr=0; rr < NUM_FFT/8; rr++)
        chess_prepare_for_pipelining
    {
        data0 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data1 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data2 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data3 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data4 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data5 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data6 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data7 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);

        data8 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        data9 = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataA = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataB = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataC = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataD = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataE = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);
        dataF = ((aie::accum<TT_ACC,8>)get_scd_v8cacc64()).to_vector<TT_DATA>(DNSHIFT);

        // data0 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data1 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data2 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data3 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data4 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data5 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data6 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data7 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
    
        // data8 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // data9 = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataA = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataB = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataC = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataD = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataE = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);
        // dataF = readincr_v<8>(acc_i).to_vector<TT_DATA>(DNSHIFT);

        aie::store_v(itw++, data0);
        aie::store_v(itw++, aie::shuffle_up_fill(data2,aie::shuffle_down_fill(data0,data1,1),1));
        aie::store_v(itw++, aie::shuffle_up_fill(data4,aie::shuffle_down_fill(data2,data3,1),2));
        aie::store_v(itw++, aie::shuffle_up_fill(data6,aie::shuffle_down_fill(data4,data5,1),3));
        aie::store_v(itw++, aie::shuffle_up_fill(data8,aie::shuffle_down_fill(data6,data7,1),4));
        aie::store_v(itw++, aie::shuffle_up_fill(dataA,aie::shuffle_down_fill(data8,data9,1),5));
        aie::store_v(itw++, aie::shuffle_up_fill(dataC,aie::shuffle_down_fill(dataA,dataB,1),6));
        aie::store_v(itw++, aie::shuffle_up_fill(dataE,aie::shuffle_down_fill(dataC,dataD,1),7));
        aie::store_v(itw++, aie::shuffle_down_fill(dataE,dataF,1));
        
        // *itw++ = data0;                                                               //  0 to  7
        // *itw++ = aie::shuffle_up_fill(data2,aie::shuffle_down_fill(data0,data1,1),1); //  8 to 15
        // *itw++ = aie::shuffle_up_fill(data4,aie::shuffle_down_fill(data2,data3,1),2); // 16 to 23
        // *itw++ = aie::shuffle_up_fill(data6,aie::shuffle_down_fill(data4,data5,1),3); // 24 to 31
        // *itw++ = aie::shuffle_up_fill(data8,aie::shuffle_down_fill(data6,data7,1),4); // 32 to 39
        // *itw++ = aie::shuffle_up_fill(dataA,aie::shuffle_down_fill(data8,data9,1),5); // 40 to 47
        // *itw++ = aie::shuffle_up_fill(dataC,aie::shuffle_down_fill(dataA,dataB,1),6); // 48 to 56
        // *itw++ = aie::shuffle_up_fill(dataE,aie::shuffle_down_fill(dataC,dataD,1),7); // 56 to 63
        // *itw++ = aie::shuffle_down_fill(dataE,dataF,1);                               // 64 to 71   
    }  
}



