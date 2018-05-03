#include "cuda_sha3.h"

//
// Hashburner Enhancements for COSMiC: LtTofu (Mag517)
// Date: April 24, 2018
//

// 
// Authors: Mikers, Azlehria
// Date: March 4 - April x, 2018 for 0xbitcoin dev
//

//
// based on https://github.com/Dunhili/SHA3-gpu-brute-force-cracker/blob/master/sha3.cu
// Author: Brian Bowden
// Date: 5/12/14
//

// Optimizations List (incomplete)
// ---

#define  NUM_HBOPT_ROTLBACK32    0
#define  NUM_HBOPT_SEL32BITXOR   1
#define  NUM_HBOPT_NOUNCEARITH   2
bool h_HBoptimizations[8] = { 0 };

// SEL32BITXOR : When state elements are XOR'd with a Round Constant where there are 8
//               or more leading zeroes, do a 32-bit XOR (via PTX) on only relevant half.
//               Seems to improve performance on Maxwell G2, not tested on lower. Might slightly
//               hurt performance on Pascal.

// ROTLBACK32  : When possible, instead of rotating a state element by a magnitude as large as 32,
//              swap the high and low bytes (shortcut to ROTL 32/ROTR 32) and then use funnel shifter
//              to get the rest of the way. Under performance testing.

// NOUNCEAROTH : ... TODO fill this out
//
// 

int32_t intensity;
int32_t cuda_device;
int32_t clock_speed;
int32_t compute_version;
struct timeb start, end;

uint64_t cnt;
uint64_t printable_hashrate_cnt;
uint64_t print_counter;

bool gpu_initialized;
bool new_input;

uint8_t solution[32] = { 0 };

uint64_t* h_message;
uint8_t init_message[84];

uint64_t* d_solution;

uint8_t* d_challenge;
uint8_t* d_hash_prefix;
__constant__ uint64_t d_mid[25];
__constant__ uint64_t d_target;
__constant__ uint32_t threads;
__constant__ bool d_HBoptimizations[8];

__device__ __forceinline__
uint64_t bswap_64( uint64_t input )
{
  asm( "{"
       "  .reg .u32 oh, ol;"
       "  mov.b64 {oh,ol}, %0;"
       "  prmt.b32 oh, oh, 0, 0x0123;"
       "  prmt.b32 ol, ol, 0, 0x0123;"
       "  mov.b64 %0, {ol,oh};"
       "}" : "+l"(input) );
  return input;
}

// try doing this with two offsettings of output operand instead
__device__ __forceinline__
uint64_t ROTL64asm (uint64_t input, uint32_t magnitude)
{
#if __CUDA_ARCH__ >= 320
    asm ("{"
         ".reg .b32 hi, lo, mag, scr;"
         "mov.b32 mag, %1;"
         "mov.b64 {hi,lo}, %0;"
         "shf.l.wrap.b32 scr, lo, hi, mag;"
         "shf.l.wrap.b32 lo, hi, lo, mag;"
         "mov.b64 %0, {scr,lo};"
         "}" : "+l"(input) : "r"(magnitude) );
    return input;
#else
    return ROTL64(input, magnitude);
#endif
}

// try doing this with two offsettings of output operand instead
__device__ __forceinline__
uint64_t ROTR64asm (uint64_t input, uint32_t magnitude)
{
  // TODO/FIXME: verify correct version is running on multiple arches
#if __CUDA_ARCH__ >= 320
    asm ("{"
         ".reg .b32 hi, lo, mag, scr;"
         "mov.b32 mag, %1;"
         "mov.b64 {hi,lo}, %0;"
         "shf.r.wrap.b32 scr, hi, lo, mag;"
         "shf.r.wrap.b32 lo, lo, hi, mag;"
         "mov.b64 %0, {scr,lo};"
         "}" : "+l"(input) : "r"(magnitude) );
    return input;
#else
    return ROTR64(input, magnitude);
#endif
}

__device__ __forceinline__
uint64_t xor5( uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e )
{
  asm( "  xor.b64 %0, %0, %1;" : "+l"(a) : "l"(b) );
  asm( "  xor.b64 %0, %0, %1;" : "+l"(a) : "l"(c) );
  asm( "  xor.b64 %0, %0, %1;" : "+l"(a) : "l"(d) );
  asm( "  xor.b64 %0, %0, %1;" : "+l"(a) : "l"(e) );
  return a;
}

// shortcut to rotation by 32 (flip halves), then rotate left by `mag`
__device__ __forceinline__
uint64_t ROTLfrom32 (uint64_t rtdby32, uint32_t magnitude)
{
    asm ("{"
         "    .reg .b32 hi, lo, scr, mag;       "
         "    mov.b64 {lo,hi}, %0;              "      // halves reversed since rotl'd by 32
         "    mov.b32 mag, %1;                  "
         "    shf.l.wrap.b32 scr, lo, hi, mag;  "
         "    shf.l.wrap.b32 lo, hi, lo, mag;   "
         "    mov.b64 %0, {scr,lo};             "
         "}" : "+l"(rtdby32) : "r"(magnitude) );    // see if this is faster w/ uint2 .x and .y
                                                  // for saving shf results out
    return rtdby32;   // return rotation from the rotation by 32
}

// shortcut to rotation by 32 (flip halves), then rotate right by `mag`
__device__ __forceinline__
uint64_t ROTRfrom32 (uint64_t rtdby32, uint32_t magnitude)
{
    asm ("{"
         "    .reg .b32 hi, lo, scr, mag;       "
         "    mov.b64 {lo,hi}, %0;              "      // halves reversed since rotl'd by 32
         "    mov.b32 mag, %1;                  "
         "    shf.r.wrap.b32 scr, hi, lo, mag;  "
         "    shf.r.wrap.b32 lo, lo, hi, mag;   "
         "    mov.b64 %0, {scr,lo};             "
         "}" : "+l"(rtdby32) : "r"(magnitude) );    // see if this is faster w/ uint2 .x and .y
                                                  // for saving shf results out
    return rtdby32;   // return rotation from the rotation by 32
}

__device__ __forceinline__
uint64_t ROTLby16 (uint64_t input)
{
    asm ( "{"
          "   .reg .b32 hi, lo, scr;"
          "   mov.b64 {hi,lo}, %0;"
          "   prmt.b32 scr, hi, lo, 0x5432;"
          "   prmt.b32 lo, hi, lo, 0x1076;"
          "   mov.b64 %0, {lo,scr};"
          " }" : "+l"(input) );
    return input;
}

// see if this is faster using x and y vectors, no extra regs
__device__ __forceinline__
uint64_t ROTLby8 (uint64_t input)
{
    asm ( "{"
          ".reg .b32 hi, lo, scr;"
          "mov.b64 {hi,lo}, %0;"
          "prmt.b32 scr, hi, lo, 0x2107;"
          "prmt.b32 lo, hi, lo, 0x6543;"
          "mov.b64 %0, {scr,lo};"
          "}"
          : "+l"(input) );
    return input;
}

__device__ __forceinline__
uint64_t ROTRby8 (uint64_t input)
{
      asm ( "{"
          ".reg .b32 hi, lo, scr;"
          "mov.b64 {hi,lo}, %0;"
          "prmt.b32 scr, lo, hi, 0x0765;"
          "prmt.b32 lo, lo, hi, 0x4321;"
          "mov.b64 %0, {scr,lo};"
          "}"
          : "+l"(input) );
    return input;
}

// TODO: Look for a snappier way to do this. Should still be slightly
//       faster than the variable-magnitude version above.
__device__ __forceinline__
uint64_t ROTLby1 (uint64_t input)
{
#if __CUDA_ARCH__ >= 320
    asm ("{"
         ".reg .b32 hi, lo, scr;"
         "mov.b64 {hi,lo}, %0;"
         "shf.l.wrap.b32 scr, lo, hi, 1;"   // magnitude replaced w/ immediate
         "shf.l.wrap.b32 lo, hi, lo, 1;"    // magnitude replaced w/ immediate
         "mov.b64 %0, {scr,lo};"
         "}" : "+l"(input) );
    return input;
#else
    return ROTL64(input, 1);
#endif
}

// try doing this with two offsettings of output operand instead
__device__ __forceinline__
uint64_t ROTRby1 (uint64_t input)
{
#if __CUDA_ARCH__ >= 320
    asm ("{"
         ".reg .b32 hi, lo, scr;"
         "mov.b64 {hi,lo}, %0;"
         "shf.r.wrap.b32 scr, hi, lo, 1;"
         "shf.r.wrap.b32 lo, lo, hi, 1;"
         "mov.b64 %0, {scr,lo};"
         "}" : "+l"(input) );
    return input;
#else
    return ROTR64(input, 1);
#endif
}

__device__ __forceinline__
uint64_t xor3( uint64_t a, uint64_t b, uint64_t c )
{
  uint64_t output{ 0 };
  asm( "{"
       "  xor.b64 %0, %1, %2;"
       "  xor.b64 %0, %0, %3;"
       "}" : "+l"(output) : "l"(a), "l"(b), "l"(c) );
  return output;
}

// FIXME: Assuming SM 5.x+
__device__ __forceinline__
uint64_t lop3_0xD2( uint64_t a, uint64_t b, uint64_t c )
{   // FIXME/TODO: make SURE that the correct version is running on Maxwell Gen2, Pascal!
#if __CUDA_ARCH__ >= 500
  asm( "{"
       "  .reg .b32 ah, al, bh, bl, ch, cl;"
       "  mov.b64 {ah,al}, %0;"
       "  mov.b64 {bh,bl}, %1;"
       "  mov.b64 {ch,cl}, %2;"

       "  lop3.b32 ah, ah, bh, ch, 0xD2;"
       "  lop3.b32 al, al, bl, cl, 0xD2;"
       "  mov.b64 %0, {ah,al};"
       "}" : "+l"(a) : "l"(b), "l"(c) );
  return a;
#else
  return a ^ ((~b) & c);
#endif
}

__device__
bool keccak( uint64_t nonce, uint32_t thread, uint64_t i_mid[], bool optimizations[] )
{
  uint64_t state[25], C[5], D[5], scratch;

  //if (thread == 543210)
  //    PermTest (0);
 uint64_t RClocal[24] =
  {
    /* Element     (elements which are '32bit': 1, 4-5, 8, 9-12, 18, 22)      */
    /* -------     ------------------  ------------------  ------------------ */
    /* 00..02  */  0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    /* 03..05  */  0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    /* 06..08  */  0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    /* 09..11  */  0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    /* 12..14  */  0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    /* 15..17  */  0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    /* 18..20  */  0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    /* 21..23  */  0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    /* -------     ------------------  ------------------  ------------------ */
  };

  // TODO: consider defining these conditionally, preprocessor
  //if (HB_SELECTIVE_32BIT_XOR)
  //uint2* RCvec = (uint2*)&RClocal;     // vectorized access to RClocal[]
  //uint2* stateVec = (uint2*)&state;    // vectorized access to state[]

  //if (thread == 543210)
  //    PermTest (0);

  C[0] = i_mid[ 2] ^ ROTR64asm(nonce, 20);
  C[1] = i_mid[ 4] ^ ROTL64asm(nonce, 14);

  state[ 0] = lop3_0xD2( i_mid[ 0], i_mid[ 1], C[ 0] );   //^ 0x0000000000000001;
#if HB_SELECTIVE_32BIT_XOR                          // shouldn't it be .y? only XOR low end of RC 0
    //stateVec[0].x = stateVec[0].x ^ 0x00000001;   // because the rest is leading zeroes :)
    asm ( "xor.b32 %0, %0, 0x00000001;" : "+r"(stateVec[0].x) );
#else
    state[0] = state[0] ^ 0x0000000000000001;     // was RC[0]
#endif

  state[ 1] = lop3_0xD2( i_mid[ 1], C[ 0], i_mid[ 3] );
  state[ 2] = lop3_0xD2( C[ 0], i_mid[ 3], C[ 1] );
  state[ 3] = lop3_0xD2( i_mid[ 3], C[ 1], i_mid[ 0] );
  state[ 4] = lop3_0xD2( C[ 1], i_mid[ 0], i_mid[ 1] );

  C[0] = i_mid[ 6] ^ ROTL64asm(nonce, 20);          // nonce*1048576;
  C[1] = i_mid[ 9] ^ ROTR64(nonce, 2);
  state[ 5] = lop3_0xD2( i_mid[ 5], C[ 0], i_mid[7] );
  state[ 6] = lop3_0xD2( C[0], i_mid[ 7], i_mid[8] );
  state[ 7] = lop3_0xD2( i_mid[ 7], i_mid[ 8], C[1] );
  state[ 8] = lop3_0xD2( i_mid[ 8], C[1], i_mid[5] );
  state[ 9] = lop3_0xD2( C[1], i_mid[ 5], C[0] );

  // experimental rotation replacement
  scratch = nonce*128;
  C[0] = i_mid[11] ^ scratch;                    // ROTL by 7
  C[1] = i_mid[13] ^ scratch*2;                  // ROTL by 8
  state[10] = lop3_0xD2( i_mid[10], C[0], i_mid[12] );
  state[11] = lop3_0xD2( C[0], i_mid[12], C[1] );
  state[12] = lop3_0xD2( i_mid[12], C[1], i_mid[14] );
  state[13] = lop3_0xD2( C[1], i_mid[14], i_mid[10] );
  state[14] = lop3_0xD2( i_mid[14], i_mid[10], C[0] );

  C[0] = i_mid[15] ^ ROTL64asm(nonce, 27);   //nonce*134217728;
  C[1] = i_mid[18] ^ ROTLby16 (nonce);
  state[15] = lop3_0xD2( C[0], i_mid[16], i_mid[17] );
  state[16] = lop3_0xD2( i_mid[16], i_mid[17], C[1] );
  state[17] = lop3_0xD2( i_mid[17], C[1], i_mid[19] );
  state[18] = lop3_0xD2( C[1], i_mid[19], C[0] );
  state[19] = lop3_0xD2( i_mid[19], C[0], i_mid[16] );

  C[0] = i_mid[20] ^ ROTRby1(nonce);
  C[1] = i_mid[21] ^ ROTR64 (nonce, 9);      //idea: ROTRby1(ROTRby8(nonce));
  C[2] = i_mid[22] ^ ROTR64(nonce, 25);
  state[20] = lop3_0xD2( C[0], C[1], C[2] );
  state[21] = lop3_0xD2( C[1], C[2], i_mid[23] );
  state[22] = lop3_0xD2( C[2], i_mid[23], i_mid[24] );
  state[23] = lop3_0xD2( i_mid[23], i_mid[24], C[0] );
  state[24] = lop3_0xD2( i_mid[24], C[0], C[1] );


#if __CUDA_ARCH__ >= 350
#  pragma unroll
#endif
  for( int32_t i{ 1 }; i < 23; ++i )
  {
    // Theta
    for( uint32_t x{ 0 }; x < 5; ++x )
    {
      C[(x + 6) % 5] = xor5( state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20] );
    }

#if __CUDA_ARCH__ >= 350
    for( uint32_t x{ 0 }; x < 5; ++x )
    {
      D[x] = ROTL64(C[(x + 2) % 5], 1);
      state[x]      = xor3( state[x]     , D[x], C[x] );
      state[x +  5] = xor3( state[x +  5], D[x], C[x] );
      state[x + 10] = xor3( state[x + 10], D[x], C[x] );
      state[x + 15] = xor3( state[x + 15], D[x], C[x] );
      state[x + 20] = xor3( state[x + 20], D[x], C[x] );
    }
#else
    for( uint32_t x{ 0 }; x < 5; ++x )
    {
      D[x] = ROTL64(C[(x + 2) % 5], 1) ^ C[x];
      state[x]      = state[x]      ^ D[x];
      state[x +  5] = state[x +  5] ^ D[x];
      state[x + 10] = state[x + 10] ^ D[x];
      state[x + 15] = state[x + 15] ^ D[x];
      state[x + 20] = state[x + 20] ^ D[x];
    }
#endif

    // Rho Pi
    C[0] = state[1];
    state[ 1] = ROTR64asm( state[ 6], 20 );
    state[ 6] = ROTL64asm( state[ 9], 20 );
    state[ 9] = ROTR64( state[22],  3 );
    state[22] = ROTR64asm( state[14], 25 );
    state[14] = ROTL64asm( state[20], 18 );
    state[20] = ROTR64( state[ 2],  2 );
    state[ 2] = ROTR64asm( state[12], 21 );
    state[12] = ROTL64asm( state[13], 25 );
    state[13] = ROTLby8( state[19] );
    state[19] = ROTRby8( state[23] );
    state[23] = ROTR64asm( state[15], 23 );
    
    // FIXME: watch out for thread divergence within warp. But this should be fine.
    //if (optimizations[NUM_HBOPT_ROTLBACK32])
#if __CUDA_ARCH__ >= 320
    state[15] = ROTRfrom32 (state[4],  5);        // (make toggleable- forced on for now)
#else
    state[15] = ROTL64(state[4], 27);
#endif

    /*if (thread==1236667)
    {
        printf ("st15: %" PRIx64 " \n", state[15]);
        printf ("tstR: %" PRIx64 " \n", ROTRfrom32(state[4], 5) );
        printf ("tstL: %" PRIx64 " \n\n", ROTLfrom32(state[4], 5) );
    }*/

    state[ 4] = ROTL64asm( state[24], 14 );
    state[24] = ROTL64( state[21],  2 );
    state[21] = ROTR64( state[ 8], 9 );            // R9
    state[ 8] = ROTR64asm( state[16], 19 );

    // FIXME: watch out for thread divergence. Slowdown here. Figure out why
    //if (optimizations[NUM_HBOPT_ROTLBACK32])    // forced on
    //{
#if __CUDA_ARCH__ >= 320
        state[16] = ROTLfrom32 (state[5], 4);
        state[ 5] = ROTRfrom32 (state[3], 4);
#else
        state[16] = ROTR64(state[5], 28);
        state[ 5] = ROTL64(state[3], 28);
#endif
    state[ 3] = ROTL64asm( state[18], 21 );
    state[18] = ROTL64asm( state[17], 15 );
    state[17] = ROTL64asm( state[11], 10 );
    state[11] = ROTL64( state[ 7],  6 );
    state[ 7] = ROTL64( state[10],  3 );
    state[10] = ROTLby1( C[0] );

    // lop3_0xD2
    for( uint32_t x{ 0 }; x < 25; x += 5 )
    {
      C[0] = state[x];
      C[1] = state[x + 1];
      C[2] = state[x + 2];
      C[3] = state[x + 3];
      C[4] = state[x + 4];
      state[x]     = lop3_0xD2( C[0], C[1], C[2] );
      state[x + 1] = lop3_0xD2( C[1], C[2], C[3] );
      state[x + 2] = lop3_0xD2( C[2], C[3], C[4] );
      state[x + 3] = lop3_0xD2( C[3], C[4], C[0] );
      state[x + 4] = lop3_0xD2( C[4], C[0], C[1] );
    }

    // Iota
    state[0] = state[0] ^ RClocal[i];
  }

  for( uint32_t x{ 0 }; x < 5; ++x )
  {
    C[(x + 6) % 5 ] = xor5( state[x], state[x + 5], state[x + 10], state[x + 15], state[x + 20] );
  }

  D[0] = ROTLby1(C[2] );
  D[1] = ROTLby1(C[3] );
  D[2] = ROTLby1(C[4] );

  state[ 0] = xor3( state[ 0], D[0], C[0] );
  state[ 6] = ROTR64asm(xor3( state[ 6], D[1], C[1] ), 20);
  state[12] = ROTR64(xor3( state[12], D[2], C[2] ), 21);

  state[ 0] = lop3_0xD2( state[ 0], state[ 6], state[12] ) ^ 0x8000000080008008;    // was RC[23];

  return bswap_64( state[0] ) <= d_target;
}

KERNEL_LAUNCH_PARAMS
void gpu_mine( uint64_t* solution, uint64_t cnt )
{
  uint64_t thread = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t nonce{ cnt + thread };
  uint64_t passingMid[25];
  bool passingOptimizations[8];
  uint8_t i;

  // TODO: use cuda/memcpy instead
  for (i=0; i<8; ++i)
    passingOptimizations[i] = d_HBoptimizations[i];

  // TODO: use cuda/memcpy instead
  for (i=0; i<25; ++i)
      passingMid[i] = d_mid[i];

  if( keccak(nonce, thread, passingMid, passingOptimizations) )
  {
    *solution = nonce;
    return;
  }
}

__host__
void stop_solving()
{
  // h_done[0] = -2;
}

__host__
uint64_t getHashCount()
{
  return cnt;
}

__host__
void resetHashCount()
{
  cudaSetDevice( cuda_device );

  *h_message = UINT64_MAX;
  cudaMemcpy( d_solution, h_message, 8, cudaMemcpyHostToDevice );   // sizeof(uint64_t), 8 bytes

  //cnt = 0;
  printable_hashrate_cnt = 0;
  print_counter = 0;

  ftime( &start );
}

__host__
void send_to_device( uint64_t target, uint64_t* message )
{
  cudaSetDevice( cuda_device );

  uint64_t C[4], D[5], mid[25];
  C[0] = message[0] ^ message[5] ^ message[10] ^ 0x100000000ull;
  C[1] = message[1] ^ message[6] ^ 0x8000000000000000ull;
  C[2] = message[2] ^ message[7];
  C[3] = message[4] ^ message[9];

  D[0] = ROTL64(C[1], 1) ^ C[3];
  D[1] = ROTL64(C[2], 1) ^ C[0];
  D[2] = ROTL64(message[3], 1) ^ C[1];
  D[3] = ROTL64(C[3], 1) ^ C[2];
  D[4] = ROTL64(C[0], 1) ^ message[3];

  mid[ 0] = message[ 0] ^ D[0];
  mid[ 1] = ROTL64( message[6] ^ D[1], 44 );
  mid[ 2] = ROTL64(D[2], 43);
  mid[ 3] = ROTL64(D[3], 21);
  mid[ 4] = ROTL64(D[4], 14);
  mid[ 5] = ROTL64( message[3] ^ D[3], 28 );
  mid[ 6] = ROTL64( message[9] ^ D[4], 20 );
  mid[ 7] = ROTL64( message[10] ^ D[0] ^ 0x100000000ull, 3 );
  mid[ 8] = ROTL64( 0x8000000000000000ull ^ D[1], 45 );
  mid[ 9] = ROTL64(D[2], 61);
  mid[10] = ROTL64( message[1] ^ D[1],  1 );
  mid[11] = ROTL64( message[7] ^ D[2],  6 );
  mid[12] = ROTL64(D[3], 25);
  mid[13] = ROTL64(D[4],  8);
  mid[14] = ROTL64(D[0], 18);
  mid[15] = ROTL64( message[4] ^ D[4], 27 );
  mid[16] = ROTL64( message[5] ^ D[0], 36 );
  mid[17] = ROTL64(D[1], 10);
  mid[18] = ROTL64(D[2], 15);
  mid[19] = ROTL64(D[3], 56);
  mid[20] = ROTL64( message[2] ^ D[2], 62 );
  mid[21] = ROTL64(D[3], 55);
  mid[22] = ROTL64(D[4], 39);
  mid[23] = ROTL64(D[0], 41);
  mid[24] = ROTL64(D[1],  2);

  cudaMemcpyToSymbol( d_mid, mid, sizeof( mid ), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol( d_target, &target, sizeof( target ), 0, cudaMemcpyHostToDevice);
}

__host__
void ConfigOptimizations (cudaDeviceProp theDevice)
{
    printf ("HashBurner (Architecture-Specific) Optimizations in use: \n");
    // TODO: do this with memcpy instead. Init all optimizations as OFF before setting them individually
    for (uint8_t i=0; i<8; ++i)
      h_HBoptimizations[i] = 0;

    // FIXME: this optimization is forced on for now, should benefit all CUDA arches
    h_HBoptimizations[NUM_HBOPT_ROTLBACK32] = 1;
    printf ("+ROTLBACK32  ");
    
    // h_HBoptimizations[NUM_HBOPT_ROTLBACK32] = 0;
    // printf ("-ROTLBACK32  ");

    if (theDevice.major >= 6)
    {
        h_HBoptimizations[NUM_HBOPT_SEL32BITXOR] = 0;
        printf ("-SEL32BITXOR  ");
    }
      else
      {
        h_HBoptimizations[NUM_HBOPT_SEL32BITXOR] = 1;
        printf ("+SEL32BITXOR  ");
      }

    // FIXME: temporarily forced on also, needs performance testing.
    h_HBoptimizations[NUM_HBOPT_NOUNCEARITH] = 1;
    printf ("+NONCEARITH  ");         // TODO: make toggle-able
    printf ("\n\n");
}

/**
 * Initializes the global variables by calling the cudaGetDeviceProperties().
 */
__host__
void gpu_init()
{
  cudaDeviceProp device_prop;
  int32_t device_count;

  char config[10];
  FILE * inf;
  inf = fopen( "0xbtc.conf", "r" );
  if( inf )
  {
    fgets( config, 10, inf );
    fclose( inf );
    intensity = atol( strtok( config, " " ) );
    cuda_device = atol( strtok( NULL, " " ) );
    printf ("\n\nRead ./0xbtc.conf - using custom intensity %d and CUDA device %d.", intensity, cuda_device);
  }
  else
  {
    intensity = INTENSITY;
    cuda_device = CUDA_DEVICE;
    printf ("\n\nNo ./0xbtc.conf found- using hardcoded intensity %d and CUDA device %d.", intensity, cuda_device);
  }

  cudaGetDeviceCount( &device_count );

  if( cudaGetDeviceProperties( &device_prop, cuda_device ) != cudaSuccess )
  {
    printf( "Problem getting properties for device. (Device unresponsive? Intensity too high?) Exiting.\n" );
    exit( EXIT_FAILURE );
  }

  cudaSetDevice( cuda_device );

  if( !gpu_initialized )
  {
    // CPU usage goes _insane_ without this.
    cudaDeviceReset();
    cudaSetDeviceFlags( cudaDeviceScheduleBlockingSync | cudaDeviceLmemResizeToMax );
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    printf ("\nInitialized CUDA device %d : %s ", cuda_device, device_prop.name);
    printf ("\nAvailable compute capability: %d.%d \n\n", device_prop.major, device_prop.minor);

    ConfigOptimizations (device_prop);                        // select optimizations, set in host array h_HBoptimizations[]

    printf ("Now mining! \n---\n");   // not really but we're about to be, and don't want this msg repeating

    cudaMalloc( (void**)&d_solution, 8);                      // solution     was sizeof(uint64_t). 8 bytes
    cudaMallocHost( (void**)&h_message, 8);                   // was sizeof(uint64_t). 8 bytes

    (uint32_t&)(init_message[52]) = 014533075101u;
    (uint32_t&)(init_message[56]) = 014132271150u;

    srand((time(NULL) & 0xFFFF) | (getpid() << 16));
    for(int8_t i_rand{ 60 }; i_rand < 84; ++i_rand){
      init_message[i_rand] = (uint8_t)rand() % 256;
    }
    memcpy( solution, &init_message[52], 32 );

    uint32_t h_threads{ 1u << intensity };
    cudaMemcpyToSymbol( threads, &h_threads, sizeof( h_threads ), 0, cudaMemcpyHostToDevice );

    //cudaMalloc ((void**)&d_HBoptimizations, sizeof(h_HBoptimizations));       // FIXME: necessary? seems not
    cudaMemcpyToSymbol ( d_HBoptimizations, &h_HBoptimizations, sizeof (h_HBoptimizations), 0, cudaMemcpyHostToDevice );

    gpu_initialized = true;
  }

  compute_version = device_prop.major * 100 + device_prop.minor * 10;

  // convert from GHz to hertz
  clock_speed = (int32_t)( device_prop.memoryClockRate * 1000 * 1000 );

  //cnt = 0;
  printable_hashrate_cnt = 0;
  print_counter = 0;

  ftime( &start );
  if( new_input ) new_input = false;
}

__host__
void update_mining_inputs( uint64_t target, uint8_t* hash_prefix )
{
  memcpy( init_message, hash_prefix, 52 );
  send_to_device( target, (uint64_t*)init_message );
}

__host__
void gpu_cleanup()
{
  cudaSetDevice( cuda_device );

  cudaThreadSynchronize();

  cudaFree( d_solution );
  cudaFreeHost( h_message );

  cudaDeviceReset();
}

__host__
bool find_message()
{
  cudaSetDevice( cuda_device );

  uint32_t threads{ 1u << intensity };

  uint32_t tpb{ compute_version > 500 ? TPB50 : TPB35 };
  dim3 grid{ (threads + tpb - 1) / tpb };
  dim3 block{ tpb };

  gpu_mine <<< grid, block >>> ( d_solution, cnt );
  // cudaError_t cudaerr = cudaDeviceSynchronize();
  // if( cudaerr != cudaSuccess )
  // {
  //  printf( "kernel launch failed with error %d: \x1b[38;5;196m%s.\x1b[0m\n", cudaerr, cudaGetErrorString( cudaerr ) );
  //  exit( EXIT_FAILURE );
  // }

  cnt += threads;
  printable_hashrate_cnt += threads;

  cudaMemcpy( h_message, d_solution, 8, cudaMemcpyDeviceToHost );       // 8 bytes, was sizeof(uint64_t)
  if( *h_message != UINT64_MAX )
    memcpy( &solution[12], h_message, 8 );                              // 8 bytes, was sizeof(uint64_t)

  ftime( &end );
  double t{ (double)((end.time * 1000 + end.millitm) - (start.time * 1000 + start.millitm)) / 1000 };

  if( t*10 > print_counter )
  {
    ++print_counter;

    // maybe breaking the control codes into macros is a good idea . . .
    printf( "\x1b[s\x1b[?25l\x1b[2;22f\x1b[38;5;221m%*.2f\x1b[0m\x1b[u\x1b[?25h"
            "\x1b[s\x1b[?25l\x1b[3;36f\x1b[38;5;208m%*" PRIu64 "\x1b[0m\x1b[u\x1b[?25h"
            "\x1b[s\x1b[?25l\x1b[2;75f\x1b[38;5;33m%02u:%02u\x1b[0m\x1b[u\x1b[?25h",
            8, ( (double)printable_hashrate_cnt / t / 1000000 ),
            25, printable_hashrate_cnt,
            ((uint32_t)t/60), ((uint32_t)t%60) );
  }

  return ( *h_message != UINT64_MAX );
  // return ( h_done[0] >= 0 );
}
