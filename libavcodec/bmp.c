/*
 * BMP image format decoder
 * Copyright (c) 2005 Mans Rullgard
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <Windows.h>
#include <inttypes.h>

#include "avcodec.h"
#include "bytestream.h"
#include "bmp.h"
#include "internal.h"
#include "msrledec.h"
#include <xmmintrin.h>
#include <emmintrin.h>

#define YUV_TBSIZE 256

typedef int8_t __attribute__((vector_size(16))) vec8;
typedef uint8_t __attribute__((vector_size(16))) uvec8;

#define NACL_R14
#define BUNDLEALIGN
#define LABELALIGN ".p2align 5\n"
#define MEMACCESS(base) "(%" #base ")"
#define MEMACCESS2(offset, base) #offset "(%" #base ")"
#define MEMLEA(offset, base) #offset "(%" #base ")"
#define MEMOPREG(opcode, offset, base, index, scale, reg) \
	#opcode " " #offset "(%" #base ",%" #index "," #scale "),%%" #reg "\n"
#define MEMOPMEM(opcode, reg, offset, base, index, scale) \
	#opcode " %%" #reg ","#offset "(%" #base ",%" #index "," #scale ")\n"

// TODO(fbarchard): Consider overlapping bits for different architectures.
// Internal flag to indicate cpuid requires initialization.
#define kCpuInit 0x1

// These flags are only valid on x86 processors.
static const int kCpuHasMMX = 0x10;
static const int kCpuHasSSE2 = 0x20;
static const int kCpuHasSSSE3 = 0x40;
static const int kCpuHasSSE41 = 0x80;
static const int kCpuHasSSE42 = 0x100;
static const int kCpuHasAVX = 0x200;
static const int kCpuHasAVX2 = 0x400;
static const int kCpuHasERMS = 0x800;
static const int kCpuHasFMA3 = 0x1000;
static const int kCpuIsAMD = 0x2000;

static int cpu_info_ = kCpuInit;  // cpu_info is not initialized yet.

// Constants for BGRA
static vec8 kBGRAToV = {  -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0,};
static vec8 kBGRAToU = {  112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0};
static vec8 kBGRAToY = {  13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0};
static uvec8 kAddUV128 = {  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u};
static uvec8 kAddY16 = {  16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u};

static unsigned short Y_R[YUV_TBSIZE],Y_G[YUV_TBSIZE],Y_B[YUV_TBSIZE],U_R[YUV_TBSIZE],U_G[YUV_TBSIZE],U_B[YUV_TBSIZE],V_R[YUV_TBSIZE],V_G[YUV_TBSIZE],V_B[YUV_TBSIZE];
static unsigned char *pic_bgra[1080], *pic_y[1080], *pic_u[1080], *pic_v[1080], **pic_yuv[3] = {NULL, };

extern void jsimd_extbgrx_ycc_convert_sse2 (uint32_t img_width, unsigned char ** input_buf, unsigned char *** output_buf
															, uint32_t output_row, int num_rows);
extern void jsimd_extbgrx_ycc_convert_mmx (uint32_t img_width, unsigned char ** input_buf, unsigned char *** output_buf
															, uint32_t output_row, int num_rows);

static void BGRAToYRow_SSSE3(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  __asm__ volatile (
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kBGRAToY),   // %3
    "m"(kAddY16)     // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

static void BGRAToUVRow_SSSE3(const uint8_t* src_argb0, int src_stride_argb,
                       uint8_t* dst_u, uint8_t* dst_v, int width) {
  __asm__ volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps    %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_argb)), // %4
    "m"(kBGRAToV),  // %5
    "m"(kBGRAToU),  // %6
    "m"(kAddUV128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}

static void MergeUVRow_SSE2(const uint8_t* src_u, const uint8_t* src_v, uint8_t* dst_uv,
                     int width) {
  __asm__ volatile (
    "sub       %0,%1                             \n"
    LABELALIGN
  "1:                                            \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0           \n"
    MEMOPREG(movdqu,0x00,0,1,1,xmm1)             //  movdqu    (%0,%1,1),%%xmm1
    "lea       " MEMLEA(0x10,0) ",%0             \n"
    "movdqa    %%xmm0,%%xmm2                     \n"
    "punpcklbw %%xmm1,%%xmm0                     \n"
    "punpckhbw %%xmm1,%%xmm2                     \n"
    "movdqu    %%xmm0," MEMACCESS(2) "           \n"
    "movdqu    %%xmm2," MEMACCESS2(0x10,2) "     \n"
    "lea       " MEMLEA(0x20,2) ",%2             \n"
    "sub       $0x10,%3                          \n"
    "jg        1b                                \n"
  : "+r"(src_u),     // %0
    "+r"(src_v),     // %1
    "+r"(dst_uv),    // %2
    "+r"(width)      // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2"
  );
}

// Convert BGRA to I420.
static int BGRAToI420(const uint8_t* src_bgra, int src_stride_bgra,
               uint8_t* dst_y, int dst_stride_y,
               uint8_t* dst_u, int dst_stride_u,
               uint8_t* dst_v, int dst_stride_v,
               int width, int height) {
  int y;
  void (*BGRAToUVRow)(const uint8_t* src_bgra0, int src_stride_bgra, uint8_t* dst_u, uint8_t* dst_v, int width);
  void (*BGRAToYRow)(const uint8_t* src_bgra, uint8_t* dst_y, int width);
  if (!src_bgra ||
      !dst_y || !dst_u || !dst_v ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_bgra = src_bgra + (height - 1) * src_stride_bgra;
    src_stride_bgra = -src_stride_bgra;
  }
  BGRAToUVRow = BGRAToUVRow_SSSE3;
  BGRAToYRow = BGRAToYRow_SSSE3;

  for (y = 0; y < height - 1; y += 2) {
    BGRAToUVRow(src_bgra, src_stride_bgra, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
    BGRAToYRow(src_bgra + src_stride_bgra, dst_y + dst_stride_y, width);
    src_bgra += src_stride_bgra * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    BGRAToUVRow(src_bgra, 0, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
  }
  return 0;
}

static int I420UVToNV12UV(const uint8_t* src_u, int src_stride_u,
               const uint8_t* src_v, int src_stride_v,
               uint8_t* dst_uv, int dst_stride_uv,
               int width, int height) {
  int y;
  void (*MergeUVRow_)(const uint8_t* src_u, const uint8_t* src_v, uint8_t* dst_uv, int width);
  // Coalesce rows.
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_u || !src_v || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    dst_uv = dst_uv + (halfheight - 1) * dst_stride_uv;
    dst_stride_uv = -dst_stride_uv;
  }

  // Coalesce rows.
  if (src_stride_u == halfwidth &&
      src_stride_v == halfwidth &&
      dst_stride_uv == halfwidth * 2) {
    halfwidth *= halfheight;
    halfheight = 1;
    src_stride_u = src_stride_v = dst_stride_uv = 0;
  }
  MergeUVRow_ = MergeUVRow_SSE2;

  for (y = 0; y < halfheight; ++y) {
    // Merge a row of U and V into a row of UV.
    MergeUVRow_(src_u, src_v, dst_uv, halfwidth);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uv += dst_stride_uv;
  }
  return 0;
}

// Low level cpuid for X86.
static void CpuId(uint32_t info_eax, uint32_t info_ecx, uint32_t* cpu_info) {
// GCC version uses inline x86 assembly.
  uint32_t info_ebx, info_edx;
  __asm__ volatile (
#if defined( __i386__) && defined(__PIC__)
    // Preserve ebx for fpic 32 bit.
    "mov %%ebx, %%edi                          \n"
    "cpuid                                     \n"
    "xchg %%edi, %%ebx                         \n"
    : "=D" (info_ebx),
#else
    "cpuid                                     \n"
    : "=b" (info_ebx),
#endif  //  defined( __i386__) && defined(__PIC__)
      "+a" (info_eax), "+c" (info_ecx), "=d" (info_edx));
  cpu_info[0] = info_eax;
  cpu_info[1] = info_ebx;
  cpu_info[2] = info_ecx;
  cpu_info[3] = info_edx;
}

// CPU detect function for SIMD instruction sets.
static int InitCpuFlags(void) {
  uint32_t cpu_info0[4] = { 0, 0, 0, 0 };
  uint32_t cpu_info1[4] = { 0, 0, 0, 0 };
  uint32_t cpu_info7[4] = { 0, 0, 0, 0 };
  CpuId(0, 0, cpu_info0);
  CpuId(1, 0, cpu_info1);
  if (cpu_info0[0] >= 7) {
    CpuId(7, 0, cpu_info7);
  }
  cpu_info_ = ((cpu_info1[3] & 0x04000000) ? kCpuHasSSE2 : 0) |
              ((cpu_info1[2] & 0x00000200) ? kCpuHasSSSE3 : 0) |
              ((cpu_info1[2] & 0x00080000) ? kCpuHasSSE41 : 0) |
              ((cpu_info1[2] & 0x00100000) ? kCpuHasSSE42 : 0) |
              ((cpu_info7[1] & 0x00000200) ? kCpuHasERMS : 0) |
              ((cpu_info1[3] & 0x00800000) ? kCpuHasMMX : 0) |
              ((cpu_info1[2] & 0x00001000) ? kCpuHasFMA3 : 0);
  if(!memcmp(&cpu_info0[2], "cAMD", 4))
  	cpu_info_ |= kCpuIsAMD;
  return cpu_info_;
}

static void bgra2nv12_init(void)
{
	static int init_done = 0;
    int i;

	if(init_done)
		return;
	
    for(i = 0; i < YUV_TBSIZE; i++)
    {
        Y_R[i] = (i * 1052) >> 12; //Y
        Y_G[i] = (i * 2065) >> 12; 
        Y_B[i] = (i * 401)  >> 12;
        U_R[i] = (i * 607)  >> 12; //U
        U_G[i] = (i * 1192) >> 12; 
        U_B[i] = (i * 1799) >> 12;
        V_R[i] = (i * 1799) >> 12; //V
        V_G[i] = (i * 1506) >> 12; 
        V_B[i] = (i * 293)  >> 12;
    }
	InitCpuFlags();

	init_done = 1;
}

static int bmp_decode_frame(AVCodecContext *avctx,
                            void *data, int *got_frame,
                            AVPacket *avpkt)
{
	static AVFrame yuv444p_frame;
    const uint8_t *buf = avpkt->data;
    int buf_size       = avpkt->size;
    AVFrame *p         = data, *p1 = &yuv444p_frame;
    unsigned int fsize, hsize;
    int width, height;
    unsigned int depth;
    BiCompression comp;
    unsigned int ihsize;
    int i, j, n, linesize, linesize1, linesize2, linesize3, linesize4, ret;
    uint32_t rgb[3] = {0};
    uint32_t alpha = 0;
    uint8_t *ptr, *ptr1, *ptr2, *ptr3, *ptr4;
    int dsize;
    const uint8_t *buf0 = buf;
    GetByteContext gb;  

	bgra2nv12_init();

    if (buf_size < 14) {
        av_log(avctx, AV_LOG_ERROR, "buf size too small (%d)\n", buf_size);
        return AVERROR_INVALIDDATA;
    }

    if (bytestream_get_byte(&buf) != 'B' ||
        bytestream_get_byte(&buf) != 'M') {
        av_log(avctx, AV_LOG_ERROR, "bad magic number\n");
        return AVERROR_INVALIDDATA;
    }

    fsize = bytestream_get_le32(&buf);
    if (buf_size < fsize) {
        av_log(avctx, AV_LOG_ERROR, "not enough data (%d < %u), trying to decode anyway\n",
               buf_size, fsize);
        fsize = buf_size;
    }

    buf += 2; /* reserved1 */
    buf += 2; /* reserved2 */

    hsize  = bytestream_get_le32(&buf); /* header size */
    ihsize = bytestream_get_le32(&buf); /* more header size */
    if (ihsize + 14LL > hsize) {
        av_log(avctx, AV_LOG_ERROR, "invalid header size %u\n", hsize);
        return AVERROR_INVALIDDATA;
    }

    /* sometimes file size is set to some headers size, set a real size in that case */
    if (fsize == 14 || fsize == ihsize + 14)
        fsize = buf_size - 2;

    if (fsize <= hsize) {
        av_log(avctx, AV_LOG_ERROR,
               "Declared file size is less than header size (%u < %u)\n",
               fsize, hsize);
        return AVERROR_INVALIDDATA;
    }

    switch (ihsize) {
    case  40: // windib
    case  56: // windib v3
    case  64: // OS/2 v2
    case 108: // windib v4
    case 124: // windib v5
        width  = bytestream_get_le32(&buf);
        height = bytestream_get_le32(&buf);
        break;
    case  12: // OS/2 v1
        width  = bytestream_get_le16(&buf);
        height = bytestream_get_le16(&buf);
        break;
    default:
        avpriv_report_missing_feature(avctx, "Information header size %u",
                                      ihsize);
        return AVERROR_PATCHWELCOME;
    }

    /* planes */
    if (bytestream_get_le16(&buf) != 1) {
        av_log(avctx, AV_LOG_ERROR, "invalid BMP header\n");
        return AVERROR_INVALIDDATA;
    }

    depth = bytestream_get_le16(&buf);

    if (ihsize >= 40)
        comp = bytestream_get_le32(&buf);
    else
        comp = BMP_RGB;

    if (comp != BMP_RGB && comp != BMP_BITFIELDS && comp != BMP_RLE4 &&
        comp != BMP_RLE8) {
        av_log(avctx, AV_LOG_ERROR, "BMP coding %d not supported\n", comp);
        return AVERROR_INVALIDDATA;
    }

    if (comp == BMP_BITFIELDS) {
        buf += 20;
        rgb[0] = bytestream_get_le32(&buf);
        rgb[1] = bytestream_get_le32(&buf);
        rgb[2] = bytestream_get_le32(&buf);
        if (ihsize > 40)
        alpha = bytestream_get_le32(&buf);
    }

    ret = ff_set_dimensions(avctx, width, height > 0 ? height : -(unsigned)height);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set dimensions %d %d\n", width, height);
        return AVERROR_INVALIDDATA;
    }

    avctx->pix_fmt = AV_PIX_FMT_NONE;

    switch (depth) {
    case 32:
        if (comp == BMP_BITFIELDS) {
            if (rgb[0] == 0xFF000000 && rgb[1] == 0x00FF0000 && rgb[2] == 0x0000FF00)
                avctx->pix_fmt = alpha ? AV_PIX_FMT_ABGR : AV_PIX_FMT_0BGR;
            else if (rgb[0] == 0x00FF0000 && rgb[1] == 0x0000FF00 && rgb[2] == 0x000000FF)
                avctx->pix_fmt = alpha ? AV_PIX_FMT_BGRA : AV_PIX_FMT_BGR0;
            else if (rgb[0] == 0x0000FF00 && rgb[1] == 0x00FF0000 && rgb[2] == 0xFF000000)
                avctx->pix_fmt = alpha ? AV_PIX_FMT_ARGB : AV_PIX_FMT_0RGB;
            else if (rgb[0] == 0x000000FF && rgb[1] == 0x0000FF00 && rgb[2] == 0x00FF0000)
                avctx->pix_fmt = alpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB0;
            else {
                av_log(avctx, AV_LOG_ERROR, "Unknown bitfields "
                       "%0"PRIX32" %0"PRIX32" %0"PRIX32"\n", rgb[0], rgb[1], rgb[2]);
                return AVERROR(EINVAL);
            }
        } else {
            avctx->pix_fmt = AV_PIX_FMT_BGRA;
        }
        break;
    case 24:
        avctx->pix_fmt = AV_PIX_FMT_BGR24;
        break;
    case 16:
        if (comp == BMP_RGB)
            avctx->pix_fmt = AV_PIX_FMT_RGB555;
        else if (comp == BMP_BITFIELDS) {
            if (rgb[0] == 0xF800 && rgb[1] == 0x07E0 && rgb[2] == 0x001F)
               avctx->pix_fmt = AV_PIX_FMT_RGB565;
            else if (rgb[0] == 0x7C00 && rgb[1] == 0x03E0 && rgb[2] == 0x001F)
               avctx->pix_fmt = AV_PIX_FMT_RGB555;
            else if (rgb[0] == 0x0F00 && rgb[1] == 0x00F0 && rgb[2] == 0x000F)
               avctx->pix_fmt = AV_PIX_FMT_RGB444;
            else {
               av_log(avctx, AV_LOG_ERROR,
                      "Unknown bitfields %0"PRIX32" %0"PRIX32" %0"PRIX32"\n",
                      rgb[0], rgb[1], rgb[2]);
               return AVERROR(EINVAL);
            }
        }
        break;
    case 8:
        if (hsize - ihsize - 14 > 0)
            avctx->pix_fmt = AV_PIX_FMT_PAL8;
        else
            avctx->pix_fmt = AV_PIX_FMT_GRAY8;
        break;
    case 1:
    case 4:
        if (hsize - ihsize - 14 > 0) {
            avctx->pix_fmt = AV_PIX_FMT_PAL8;
        } else {
            av_log(avctx, AV_LOG_ERROR, "Unknown palette for %u-colour BMP\n",
                   1 << depth);
            return AVERROR_INVALIDDATA;
        }
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "depth %u not supported\n", depth);
        return AVERROR_INVALIDDATA;
    }

    if (avctx->pix_fmt == AV_PIX_FMT_NONE) {
        av_log(avctx, AV_LOG_ERROR, "unsupported pixel format\n");
        return AVERROR_INVALIDDATA;
    }
	
#if 1
	if(depth == 32)
	{
		if (yuv444p_frame.data[0] == NULL)
		{
			avctx->pix_fmt = AV_PIX_FMT_YUV444P;
			if((ret = ff_get_buffer(avctx, &yuv444p_frame, 0)) < 0)
			{
				yuv444p_frame.data[0] = NULL;
				av_log(avctx, AV_LOG_ERROR, "ff_get_buffer yuv444p_frame error!\n");
	        	return ret;
			}
		}
		avctx->pix_fmt = AV_PIX_FMT_YUV420P;
		//avctx->pix_fmt = AV_PIX_FMT_NV12;	
	}
#endif
    if ((ret = ff_get_buffer(avctx, p, 0)) < 0)
    {
    	av_log(avctx, AV_LOG_ERROR, "ff_get_buffer error!\n");
        return ret;
    }
    p->pict_type = AV_PICTURE_TYPE_I;
    p->key_frame = 1;
#if 1
	if(depth == 32)
	{
		p->width = avctx->width;
		p->height = avctx->height;
		p->format = avctx->pix_fmt;
	}
#endif

    buf   = buf0 + hsize;
    dsize = buf_size - hsize;

    /* Line size in file multiple of 4 */
    n = ((avctx->width * depth + 31) / 8) & ~3;

    if (n * avctx->height > dsize && comp != BMP_RLE4 && comp != BMP_RLE8) {
        n = (avctx->width * depth + 7) / 8;
        if (n * avctx->height > dsize) {
            av_log(avctx, AV_LOG_ERROR, "not enough data (%d < %d)\n",
                   dsize, n * avctx->height);
            return AVERROR_INVALIDDATA;
        }
        av_log(avctx, AV_LOG_ERROR, "data size too small, assuming missing line alignment\n");
    }

    // RLE may skip decoding some picture areas, so blank picture before decoding
    if (comp == BMP_RLE4 || comp == BMP_RLE8)
        memset(p->data[0], 0, avctx->height * p->linesize[0]);

    if (height > 0) {
        ptr      = p->data[0] + (avctx->height - 1) * p->linesize[0];
		ptr1     = p1->data[1] + (avctx->height - 1) * p1->linesize[1];
		ptr2     = p1->data[2] + (avctx->height - 1) * p1->linesize[2];
		ptr3     = p->data[1] + ((avctx->height + 1) / 2 - 1) * p->linesize[1];
		ptr4     = p->data[2] + ((avctx->height + 1) / 2 - 1) * p->linesize[2];
        linesize = -p->linesize[0];
		linesize1 = -p1->linesize[1];
		linesize2 = -p1->linesize[2];
		linesize3 = -p->linesize[1];
		linesize4 = -p->linesize[2];
    } else {
        ptr      = p->data[0];
		ptr1     = p1->data[1];
		ptr2     = p1->data[2];
		ptr3     = p->data[1];
		ptr4     = p->data[2];
        linesize = p->linesize[0];
		linesize1 = p1->linesize[1];
		linesize2 = p1->linesize[2];
		linesize3 = p->linesize[1];
		linesize4 = p->linesize[2];
    }

    if (avctx->pix_fmt == AV_PIX_FMT_PAL8) {
        int colors = 1 << depth;

        memset(p->data[1], 0, 1024);

        if (ihsize >= 36) {
            int t;
            buf = buf0 + 46;
            t   = bytestream_get_le32(&buf);
            if (t < 0 || t > (1 << depth)) {
                av_log(avctx, AV_LOG_ERROR,
                       "Incorrect number of colors - %X for bitdepth %u\n",
                       t, depth);
            } else if (t) {
                colors = t;
            }
        } else {
            colors = FFMIN(256, (hsize-ihsize-14) / 3);
        }
        buf = buf0 + 14 + ihsize; //palette location
        // OS/2 bitmap, 3 bytes per palette entry
        if ((hsize-ihsize-14) < (colors << 2)) {
            if ((hsize-ihsize-14) < colors * 3) {
                av_log(avctx, AV_LOG_ERROR, "palette doesn't fit in packet\n");
                return AVERROR_INVALIDDATA;
            }
            for (i = 0; i < colors; i++)
                ((uint32_t*)p->data[1])[i] = (0xFFU<<24) | bytestream_get_le24(&buf);
        } else {
            for (i = 0; i < colors; i++)
                ((uint32_t*)p->data[1])[i] = 0xFFU << 24 | bytestream_get_le32(&buf);
        }
        buf = buf0 + hsize;
    }
    if (comp == BMP_RLE4 || comp == BMP_RLE8) {
        if (comp == BMP_RLE8 && height < 0) {
            p->data[0]    +=  p->linesize[0] * (avctx->height - 1);
            p->linesize[0] = -p->linesize[0];
        }
        bytestream2_init(&gb, buf, dsize);
        ff_msrle_decode(avctx, p, depth, &gb);
        if (height < 0) {
            p->data[0]    +=  p->linesize[0] * (avctx->height - 1);
            p->linesize[0] = -p->linesize[0];
        }
    } else {
        switch (depth) {
        case 1:
            for (i = 0; i < avctx->height; i++) {
                int j;
                for (j = 0; j < n; j++) {
                    ptr[j*8+0] =  buf[j] >> 7;
                    ptr[j*8+1] = (buf[j] >> 6) & 1;
                    ptr[j*8+2] = (buf[j] >> 5) & 1;
                    ptr[j*8+3] = (buf[j] >> 4) & 1;
                    ptr[j*8+4] = (buf[j] >> 3) & 1;
                    ptr[j*8+5] = (buf[j] >> 2) & 1;
                    ptr[j*8+6] = (buf[j] >> 1) & 1;
                    ptr[j*8+7] =  buf[j]       & 1;
                }
                buf += n;
                ptr += linesize;
            }
            break;
        case 8:
        case 24:
			for (i = 0; i < avctx->height; i++) 
			{
                memcpy(ptr, buf, n);  
                buf += n;
				ptr += linesize;
			}
			break;
        case 32:
#if 1
			for(i = 0; i < 30; i++)
				if(buf[i] == '\0')
					break;
			if(i < 30 && !memcmp(buf, "DXGI", 4) 
				&& sscanf(buf, "DXGI %x %x DXGI", &i, &j) == 2)
			{
				buf = (const uint8_t *)i;
				n = j;
			}
			if(cpu_info_ & kCpuHasSSSE3)
			{
				if(avctx->pix_fmt == AV_PIX_FMT_YUV420P)
				{
					BGRAToI420(buf, n, ptr, linesize, ptr3, linesize3, ptr4, linesize4, (avctx->width + 15) / 16 * 16, avctx->height);
					break;
				}
				BGRAToI420(buf, n, ptr, linesize, ptr1, linesize1, ptr2, linesize2, (avctx->width + 15) / 16 * 16, avctx->height);
				if(cpu_info_ & kCpuHasSSE2)
				{
					I420UVToNV12UV(ptr1, linesize1, ptr2, linesize2, ptr3, linesize3, (avctx->width + 15) / 16 * 16, avctx->height);
				}
				else
				{
					int x, y;
					
					for(i = 0; i < (avctx->height + 1) / 2; i++)
					{
						for(x = 0, y = 0; x < avctx->width; x += 2, y++)
						{
							ptr3[x] = ptr1[y];
							ptr3[x + 1] = ptr2[y];
						}
						ptr3 += linesize3;
						ptr1 += linesize1;
						ptr2 += linesize2;
					}
				}
			}
			else if((cpu_info_ & kCpuHasSSE2) && 
						(!(cpu_info_ & kCpuIsAMD) || (cpu_info_ & kCpuHasMMX)))
			{
				__m128i uvMask = _mm_set1_epi16(0x00FF);
				__m128i uvMask1 = _mm_set1_epi16(0xFF00);  
				
				for(i = 0; i < avctx->height; i++)
					pic_bgra[i] = (unsigned char *)buf + i * n;
				for(i = 0; i < avctx->height; i++)
					pic_y[i] = ptr + i * linesize;
				if(pic_yuv[0] == NULL)
				{
					for(i = 0; i < avctx->height; i++)
						pic_u[i] = ptr1 + i * linesize1;
					for(i = 0; i < avctx->height; i++)
						pic_v[i] = ptr2 + i * linesize2;
					pic_yuv[0] = pic_y;
					pic_yuv[1] = pic_u;
					pic_yuv[2] = pic_v;
				}
				if(cpu_info_ & kCpuIsAMD)
					jsimd_extbgrx_ycc_convert_mmx(avctx->width, pic_bgra, pic_yuv, 0, avctx->height);
				else
					jsimd_extbgrx_ycc_convert_sse2(avctx->width, pic_bgra, pic_yuv, 0, avctx->height);

				if(avctx->pix_fmt == AV_PIX_FMT_YUV420P)
				{			
					for (i = 0; i < avctx->height; i += 2) 
					{ 
						int uYPos = (i >> 1) * linesize3;
						int vYPos = (i >> 1) * linesize4;
						
						for(int x = 0; x < avctx->width; x += 32)
						{                     
							int uPos  = uYPos + x / 2;
							int vPos  = vYPos + x / 2;
							uint8_t *ptr5 = ptr1 + x, *ptr6 = ptr2 + x;

							__m128i line1 = _mm_loadu_si128((__m128i*)ptr5);            
							__m128i line2 = _mm_loadu_si128((__m128i*)(ptr5+linesize1));   
							__m128i line3 = _mm_loadu_si128((__m128i*)ptr5 + 1);            
							__m128i line4 = _mm_loadu_si128((__m128i*)(ptr5+linesize1 + 1));                     
							__m128i addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
							__m128i addVal1 = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
							__m128i avgVal = _mm_and_si128(_mm_srai_epi16(addVal1, 2), uvMask);

							ptr5 += 16;
							line1 = _mm_loadu_si128((__m128i*)ptr5);            
							line2 = _mm_loadu_si128((__m128i*)(ptr5+linesize1));   
							line3 = _mm_loadu_si128((__m128i*)ptr5 + 1);            
							line4 = _mm_loadu_si128((__m128i*)(ptr5+linesize1 + 1));                      
							addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
							addVal = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
							avgVal = _mm_packus_epi16(avgVal, _mm_and_si128(_mm_srai_epi16(addVal, 2), uvMask));
							_mm_storeu_si128((__m128i *)(ptr3+uPos), avgVal);
							
							line1 = _mm_loadu_si128((__m128i*)ptr6);            
							line2 = _mm_loadu_si128((__m128i*)(ptr6+linesize2));   
							line3 = _mm_loadu_si128((__m128i*)ptr6 + 1);            
							line4 = _mm_loadu_si128((__m128i*)(ptr6+linesize2 + 1));                      
							addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
							addVal = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
							avgVal = _mm_and_si128(_mm_srai_epi16(addVal, 2), uvMask);
							
							ptr6 += 16;
							line1 = _mm_loadu_si128((__m128i*)ptr6);            
							line2 = _mm_loadu_si128((__m128i*)(ptr6+linesize2));   
							line3 = _mm_loadu_si128((__m128i*)ptr6 + 1);            
							line4 = _mm_loadu_si128((__m128i*)(ptr6+linesize2 + 1));                      
							addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
							addVal = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
							avgVal = _mm_packus_epi16(avgVal, _mm_and_si128(_mm_srai_epi16(addVal, 2), uvMask));
							_mm_storeu_si128((__m128i *)(ptr4+vPos), avgVal);
						}

						ptr1 += linesize1 * 2;
						ptr2 += linesize2 * 2;
					}
					break;
				}
				for (i = 0; i < avctx->height; i += 2) 
				{ 
					int uvYPos = (i >> 1) * linesize3;         
					
					for(int x = 0; x < avctx->width; x += 16)
					{                     
						int uvPos  = uvYPos + x;
						uint8_t *ptr5 = ptr1 + x, *ptr6 = ptr2 + x;

						__m128i line1 = _mm_loadu_si128((__m128i*)ptr5);            
						__m128i line2 = _mm_loadu_si128((__m128i*)(ptr5+linesize1));   
						__m128i line3 = _mm_loadu_si128((__m128i*)ptr5 + 1);            
						__m128i line4 = _mm_loadu_si128((__m128i*)(ptr5+linesize1 + 1));                     
						__m128i addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
						__m128i addVal1 = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
						__m128i avgVal = _mm_srai_epi16(addVal1, 2);
						
						line1 = _mm_loadu_si128((__m128i*)ptr6);            
						line2 = _mm_loadu_si128((__m128i*)(ptr6+linesize2));   
						line3 = _mm_loadu_si128((__m128i*)ptr6 + 1);            
						line4 = _mm_loadu_si128((__m128i*)(ptr6+linesize2 + 1));                      
						addVal = _mm_add_epi64(_mm_and_si128(line1, uvMask), _mm_and_si128(line2, uvMask));   
						addVal = _mm_add_epi64(addVal, _mm_add_epi64(_mm_and_si128(line3, uvMask), _mm_and_si128(line4, uvMask)));
						avgVal = _mm_add_epi64(_mm_and_si128(_mm_slli_epi16(addVal, 6), uvMask1), _mm_and_si128(avgVal, uvMask));

						_mm_storeu_si128((__m128i *)(ptr3+uvPos), avgVal);
					}

					ptr1 += linesize1 * 2;
					ptr2 += linesize2 * 2;
				}
			}
			else
			{
				if(avctx->pix_fmt == AV_PIX_FMT_YUV420P)
				{
					for (i = 0; i < avctx->height; i += 2) 
					{
						int uYPos = (i >> 1) * linesize3;  
						int vYPos = (i >> 1) * linesize4; 
						int lumYPos = i * linesize;  
						
						for(int x = 0; x < avctx->width; x += 8)
						{            
							LPBYTE lpImagePos = (LPBYTE)buf + (x * 4);            
							int uPos  = uYPos + x / 2;   
							int vPos  = vYPos + x / 2;   
							int lumPos0 = lumYPos + x;            
							int lumPos1 = lumPos0 + linesize;  
							int y1, y2, u, v;

							y1 = Y_R[lpImagePos[12 + 2]] + Y_G[lpImagePos[12 + 1]] + Y_B[lpImagePos[12 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[8 + 2]] + Y_G[lpImagePos[8 + 1]] + Y_B[lpImagePos[8 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[4 + 2]] + Y_G[lpImagePos[4 + 1]] + Y_B[lpImagePos[4 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[0 + 2]] + Y_G[lpImagePos[0 + 1]] + Y_B[lpImagePos[0 + 0]]; //Y
							*(LPUINT)(ptr+lumPos0) = y1;
							y1 = Y_R[lpImagePos[16 + 12 + 2]] + Y_G[lpImagePos[16 + 12 + 1]] + Y_B[lpImagePos[16 + 12 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[16 + 8 + 2]] + Y_G[lpImagePos[16 + 8 + 1]] + Y_B[lpImagePos[16 + 8 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[16 + 4 + 2]] + Y_G[lpImagePos[16 + 4 + 1]] + Y_B[lpImagePos[16 + 4 + 0]]; //Y
							y1 <<= 8;
							y1 |= Y_R[lpImagePos[16 + 0 + 2]] + Y_G[lpImagePos[16 + 0 + 1]] + Y_B[lpImagePos[16 + 0 + 0]]; //Y
							*(LPUINT)(ptr+lumPos0+4) = y1;

							v = V_R[lpImagePos[16 + 8 + 2]] - V_G[lpImagePos[16 + 8 + 1]] - V_B[lpImagePos[16 + 8 + 0]] + 128; //V
							v <<= 8;
							u = U_B[lpImagePos[16 + 8 + 0]] - U_R[lpImagePos[16 + 8 + 2]] - U_G[lpImagePos[16 + 8 + 1]] + 128; //U
							u <<= 8;
							v |= V_R[lpImagePos[16 + 0 + 2]] - V_G[lpImagePos[16 + 0 + 1]] - V_B[lpImagePos[16 + 0 + 0]] + 128; //V
							v <<= 8;
							u |= U_B[lpImagePos[16 + 0 + 0]] - U_R[lpImagePos[16 + 0 + 2]] - U_G[lpImagePos[16 + 0 + 1]] + 128; //U
							u <<= 8;
							v |= V_R[lpImagePos[8 + 2]] - V_G[lpImagePos[8 + 1]] - V_B[lpImagePos[8 + 0]] + 128; //V
							v <<= 8;
							u |= U_B[lpImagePos[8 + 0]] - U_R[lpImagePos[8 + 2]] - U_G[lpImagePos[8 + 1]] + 128; //U
							u <<= 8;
							v |= V_R[lpImagePos[0 + 2]] - V_G[lpImagePos[0 + 1]] - V_B[lpImagePos[0 + 0]] + 128; //V
							u |= U_B[lpImagePos[0 + 0]] - U_R[lpImagePos[0 + 2]] - U_G[lpImagePos[0 + 1]] + 128; //U
							*(LPUINT)(ptr3+uPos) = u;
							*(LPUINT)(ptr4+vPos) = v;

							lpImagePos += n;
							y2 = Y_R[lpImagePos[12 + 2]] + Y_G[lpImagePos[12 + 1]] + Y_B[lpImagePos[12 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[8 + 2]] + Y_G[lpImagePos[8 + 1]] + Y_B[lpImagePos[8 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[4 + 2]] + Y_G[lpImagePos[4 + 1]] + Y_B[lpImagePos[4 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[0 + 2]] + Y_G[lpImagePos[0 + 1]] + Y_B[lpImagePos[0 + 0]]; //Y         
							*(LPUINT)(ptr+lumPos1) = y2;
							y2 = Y_R[lpImagePos[16 + 12 + 2]] + Y_G[lpImagePos[16 + 12 + 1]] + Y_B[lpImagePos[16 + 12 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[16 + 8 + 2]] + Y_G[lpImagePos[16 + 8 + 1]] + Y_B[lpImagePos[16 + 8 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[16 + 4 + 2]] + Y_G[lpImagePos[16 + 4 + 1]] + Y_B[lpImagePos[16 + 4 + 0]]; //Y
							y2 <<= 8;
							y2 |= Y_R[lpImagePos[16 + 0 + 2]] + Y_G[lpImagePos[16 + 0 + 1]] + Y_B[lpImagePos[16 + 0 + 0]]; //Y
							*(LPUINT)(ptr+lumPos1+4) = y2;
						}
		                buf += 2 * n;
		            }
					break;
				}
	            for (i = 0; i < avctx->height; i += 2) 
				{
					int uvYPos = (i >> 1) * linesize3;        
					int lumYPos = i * linesize;  
					
					for(int x = 0; x < avctx->width; x += 4)
					{            
						LPBYTE lpImagePos = (LPBYTE)buf + (x * 4);            
						int uvPos  = uvYPos + x;            
						int lumPos0 = lumYPos + x;            
						int lumPos1 = lumPos0 + linesize;  
						int y1, y2, uv;

						y1 = Y_R[lpImagePos[12 + 2]] + Y_G[lpImagePos[12 + 1]] + Y_B[lpImagePos[12 + 0]]; //Y
						y1 <<= 8;
						y1 |= Y_R[lpImagePos[8 + 2]] + Y_G[lpImagePos[8 + 1]] + Y_B[lpImagePos[8 + 0]]; //Y
						y1 <<= 8;
						y1 |= Y_R[lpImagePos[4 + 2]] + Y_G[lpImagePos[4 + 1]] + Y_B[lpImagePos[4 + 0]]; //Y
						y1 <<= 8;
						y1 |= Y_R[lpImagePos[0 + 2]] + Y_G[lpImagePos[0 + 1]] + Y_B[lpImagePos[0 + 0]]; //Y

						uv = V_R[lpImagePos[8 + 2]] - V_G[lpImagePos[8 + 1]] - V_B[lpImagePos[8 + 0]] + 128; //V
						uv <<= 8;
						uv |= U_B[lpImagePos[8 + 0]] - U_R[lpImagePos[8 + 2]] - U_G[lpImagePos[8 + 1]] + 128; //U
						uv <<= 8;
						uv |= V_R[lpImagePos[0 + 2]] - V_G[lpImagePos[0 + 1]] - V_B[lpImagePos[0 + 0]] + 128; //V
						uv <<= 8;
						uv |= U_B[lpImagePos[0 + 0]] - U_R[lpImagePos[0 + 2]] - U_G[lpImagePos[0 + 1]] + 128; //U

						lpImagePos += n;
						y2 = Y_R[lpImagePos[12 + 2]] + Y_G[lpImagePos[12 + 1]] + Y_B[lpImagePos[12 + 0]]; //Y
						y2 <<= 8;
						y2 |= Y_R[lpImagePos[8 + 2]] + Y_G[lpImagePos[8 + 1]] + Y_B[lpImagePos[8 + 0]]; //Y
						y2 <<= 8;
						y2 |= Y_R[lpImagePos[4 + 2]] + Y_G[lpImagePos[4 + 1]] + Y_B[lpImagePos[4 + 0]]; //Y
						y2 <<= 8;
						y2 |= Y_R[lpImagePos[0 + 2]] + Y_G[lpImagePos[0 + 1]] + Y_B[lpImagePos[0 + 0]]; //Y
						*(LPUINT)(ptr+lumPos0) = y1;           
						*(LPUINT)(ptr+lumPos1) = y2;
						*(LPUINT)(ptr3+uvPos) = uv;
					}
	                buf += 2 * n;
	            }
			}
#else
			
			for (i = 0; i < avctx->height; i += 1) 
			{
                memcpy(ptr, buf, n);  
                buf += n;
				ptr += linesize;
			}
#endif
            break;
        case 4:
            for (i = 0; i < avctx->height; i++) {
                int j;
                for (j = 0; j < n; j++) {
                    ptr[j*2+0] = (buf[j] >> 4) & 0xF;
                    ptr[j*2+1] = buf[j] & 0xF;
                }
                buf += n;
                ptr += linesize;
            }
            break;
        case 16:
            for (i = 0; i < avctx->height; i++) {
                const uint16_t *src = (const uint16_t *) buf;
                uint16_t *dst       = (uint16_t *) ptr;

                for (j = 0; j < avctx->width; j++)
                    *dst++ = av_le2ne16(*src++);

                buf += n;
                ptr += linesize;
            }
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "BMP decoder is broken\n");
            return AVERROR_INVALIDDATA;
        }
    }
    if (avctx->pix_fmt == AV_PIX_FMT_BGRA) {
        for (i = 0; i < avctx->height; i++) {
            int j;
            uint8_t *ptr = p->data[0] + p->linesize[0]*i + 3;
            for (j = 0; j < avctx->width; j++) {
                if (ptr[4*j])
                    break;
            }
            if (j < avctx->width)
                break;
        }
        if (i == avctx->height)
            avctx->pix_fmt = p->format = AV_PIX_FMT_BGR0;
    }

    *got_frame = 1;

    return buf_size;
}

AVCodec ff_bmp_decoder = {
    .name           = "bmp",
    .long_name      = NULL_IF_CONFIG_SMALL("BMP (Windows and OS/2 bitmap)"),
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_BMP,
    .decode         = bmp_decode_frame,
    .capabilities   = AV_CODEC_CAP_DR1,
};
