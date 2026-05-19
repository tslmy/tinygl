/*
 * NEON-optimized scanline rasterization for ARM Cortex-A7.
 *
 * Provides fast-path scanline functions for the most common blend modes
 * used by the dice roller app:
 *   - GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA (translucent dice, shadows)
 *   - GL_ONE / GL_ONE (additive bloom halos)
 *
 * Processes 8 pixels per NEON iteration (128-bit registers).
 * Falls back to scalar for unsupported blend modes and remainders.
 */

#ifndef ZTRIANGLE_NEON_H
#define ZTRIANGLE_NEON_H

#ifdef __ARM_NEON
#include <arm_neon.h>

/*
 * Flat-shaded scanline with SRC_ALPHA / ONE_MINUS_SRC_ALPHA blend.
 * Source color (with alpha) is constant for the entire triangle.
 *
 * Formula per channel: result = (src * alpha + dst * (255 - alpha) + 128) >> 8
 *
 * Processes 8 pixels per NEON iteration using vld4/vst4 for channel
 * deinterleaving (pixel format: B G R A in memory, little-endian 0xAARRGGBB).
 */
static inline void neon_flat_blend_srcalpha(PIXEL *pp, GLint count,
                                            GLuint color) {
    uint8_t sa = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;
    uint8_t inv_sa = 255 - sa;

    /* Pre-multiply source by alpha (16-bit, +128 for rounding) */
    uint16x8_t src_r_premul = vdupq_n_u16((uint16_t)sr * sa + 128);
    uint16x8_t src_g_premul = vdupq_n_u16((uint16_t)sg * sa + 128);
    uint16x8_t src_b_premul = vdupq_n_u16((uint16_t)sb * sa + 128);
    uint8x8_t inv_alpha_vec = vdup_n_u8(inv_sa);

    while (count >= 8) {
        /* Load 8 pixels, deinterleave into B, G, R, A channels */
        uint8x8x4_t dst = vld4_u8((const uint8_t *)pp);

        /* dst_channel * (255 - alpha) → 16-bit */
        uint16x8_t dr = vmull_u8(dst.val[2], inv_alpha_vec);
        uint16x8_t dg = vmull_u8(dst.val[1], inv_alpha_vec);
        uint16x8_t db = vmull_u8(dst.val[0], inv_alpha_vec);

        /* Add pre-multiplied source */
        dr = vaddq_u16(dr, src_r_premul);
        dg = vaddq_u16(dg, src_g_premul);
        db = vaddq_u16(db, src_b_premul);

        /* Narrow: >> 8 to get final 8-bit channels */
        uint8x8x4_t result;
        result.val[2] = vshrn_n_u16(dr, 8);
        result.val[1] = vshrn_n_u16(dg, 8);
        result.val[0] = vshrn_n_u16(db, 8);
        result.val[3] = vdup_n_u8(0); /* alpha output = 0 (not used) */

        vst4_u8((uint8_t *)pp, result);
        pp += 8;
        count -= 8;
    }

    /* Scalar remainder */
    while (count > 0) {
        GLuint dest = *pp;
        GLuint dr = (dest >> 16) & 0xFF;
        GLuint dg = (dest >> 8) & 0xFF;
        GLuint db = dest & 0xFF;
        GLuint rr = ((GLuint)sr * sa + dr * inv_sa + 128) >> 8;
        GLuint rg = ((GLuint)sg * sa + dg * inv_sa + 128) >> 8;
        GLuint rb = ((GLuint)sb * sa + db * inv_sa + 128) >> 8;
        *pp = (rr << 16) | (rg << 8) | rb;
        pp++;
        count--;
    }
}

/*
 * Flat-shaded scanline with GL_ONE / GL_ONE (additive blend).
 * result = saturate(src + dst)
 *
 * Extremely fast with NEON saturating add (vqadd_u8).
 */
static inline void neon_flat_blend_additive(PIXEL *pp, GLint count,
                                            GLuint color) {
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;

    /* Broadcast source color as interleaved B,G,R,A bytes */
    uint8x8_t src_b = vdup_n_u8(sb);
    uint8x8_t src_g = vdup_n_u8(sg);
    uint8x8_t src_r = vdup_n_u8(sr);

    while (count >= 8) {
        uint8x8x4_t dst = vld4_u8((const uint8_t *)pp);

        uint8x8x4_t result;
        result.val[0] = vqadd_u8(dst.val[0], src_b);
        result.val[1] = vqadd_u8(dst.val[1], src_g);
        result.val[2] = vqadd_u8(dst.val[2], src_r);
        result.val[3] = dst.val[3];

        vst4_u8((uint8_t *)pp, result);
        pp += 8;
        count -= 8;
    }

    /* Scalar remainder */
    while (count > 0) {
        GLuint dest = *pp;
        GLuint dr = (dest >> 16) & 0xFF;
        GLuint dg = (dest >> 8) & 0xFF;
        GLuint db = dest & 0xFF;
        GLuint rr = dr + sr; if (rr > 255) rr = 255;
        GLuint rg = dg + sg; if (rg > 255) rg = 255;
        GLuint rb = db + sb; if (rb > 255) rb = 255;
        *pp = (rr << 16) | (rg << 8) | rb;
        pp++;
        count--;
    }
}

/*
 * Flat-shaded scanline dispatcher.
 * Checks blend mode at runtime and dispatches to the appropriate NEON
 * fast-path, or falls back to scalar TGL_BLEND_FUNC for unsupported modes.
 */
static inline void neon_flat_blend_scanline(PIXEL *pp, GLint count,
                                            GLuint color,
                                            GLuint sfactor, GLuint dfactor,
                                            GLuint blendeq) {
    if (blendeq == GL_FUNC_ADD) {
        if (sfactor == GL_SRC_ALPHA && dfactor == GL_ONE_MINUS_SRC_ALPHA) {
            neon_flat_blend_srcalpha(pp, count, color);
            return;
        }
        if (sfactor == GL_ONE && dfactor == GL_ONE) {
            neon_flat_blend_additive(pp, count, color);
            return;
        }
    }
    /* Fallback: generic scalar blend */
    {
        GLuint zbblendeq = blendeq; /* alias for TGL_BLEND_FUNC macro */
        while (count > 0) {
            TGL_BLEND_FUNC(color, (*pp));
            pp++;
            count--;
        }
    }
}

/*
 * Smooth-shaded scanline with SRC_ALPHA / ONE_MINUS_SRC_ALPHA blend.
 *
 * In TinyGL's smooth blend, alpha is hardcoded to 0xFF, which means:
 *   sfactor=GL_SRC_ALPHA  → multiply source by 255 (≈identity)
 *   dfactor=GL_ONE_MINUS_SRC_ALPHA → multiply dest by 0 (zeroed)
 *   result ≈ source
 *
 * So smooth+blend with alpha=0xFF degenerates to just writing the
 * interpolated color. We exploit this by packing R,G,B directly to pixels
 * using NEON shift+mask operations on 4 pixels at a time.
 *
 * r1/g1/b1 are 24-bit fixed-point (channel value in bits 16-23).
 * RGB_TO_PIXEL(r,g,b) = (r & 0xff0000) | ((g >> 8) & 0xff00) | ((b >> 16) & 0xff)
 */
static inline void neon_smooth_write_scanline(PIXEL *pp, GLint count,
                                              GLint r_start, GLint g_start,
                                              GLint b_start,
                                              GLint drdx, GLint dgdx,
                                              GLint dbdx) {
    /* NEON: process 4 pixels at a time using int32x4 */

    /* Initial values: r_start + {0,1,2,3} * drdx */
    GLint offsets[4] = {0, 1, 2, 3};
    int32x4_t offset = vld1q_s32(offsets);
    int32x4_t r_vec = vmlaq_s32(vdupq_n_s32(r_start), vdupq_n_s32(drdx), offset);
    int32x4_t g_vec = vmlaq_s32(vdupq_n_s32(g_start), vdupq_n_s32(dgdx), offset);
    int32x4_t b_vec = vmlaq_s32(vdupq_n_s32(b_start), vdupq_n_s32(dbdx), offset);

    /* Step by 4 pixels per iteration: stride = 4 * dx */
    int32x4_t stride_r = vdupq_n_s32(drdx * 4);
    int32x4_t stride_g = vdupq_n_s32(dgdx * 4);
    int32x4_t stride_b = vdupq_n_s32(dbdx * 4);

    uint32x4_t mask_r = vdupq_n_u32(0x00FF0000);
    uint32x4_t mask_g = vdupq_n_u32(0x0000FF00);
    uint32x4_t mask_b = vdupq_n_u32(0x000000FF);

    while (count >= 4) {
        /* Pack: pixel = (r & 0xff0000) | ((g >> 8) & 0xff00) | ((b >> 16) & 0xff) */
        uint32x4_t r_masked = vandq_u32(vreinterpretq_u32_s32(r_vec), mask_r);
        uint32x4_t g_shifted = vshrq_n_u32(vreinterpretq_u32_s32(g_vec), 8);
        uint32x4_t g_masked = vandq_u32(g_shifted, mask_g);
        uint32x4_t b_shifted = vshrq_n_u32(vreinterpretq_u32_s32(b_vec), 16);
        uint32x4_t b_masked = vandq_u32(b_shifted, mask_b);

        uint32x4_t pixels = vorrq_u32(vorrq_u32(r_masked, g_masked), b_masked);
        vst1q_u32((uint32_t *)pp, pixels);

        /* Advance interpolation by 4 pixels */
        r_vec = vaddq_s32(r_vec, stride_r);
        g_vec = vaddq_s32(g_vec, stride_g);
        b_vec = vaddq_s32(b_vec, stride_b);

        pp += 4;
        count -= 4;
    }

    /* Scalar remainder: extract current r/g/b from vector lane 0 */
    GLint or1 = vgetq_lane_s32(r_vec, 0);
    GLint og1 = vgetq_lane_s32(g_vec, 0);
    GLint ob1 = vgetq_lane_s32(b_vec, 0);
    while (count > 0) {
        *pp = (or1 & 0xff0000) | ((og1 >> 8) & 0xff00) | ((ob1 >> 16) & 0xff);
        or1 += drdx;
        og1 += dgdx;
        ob1 += dbdx;
        pp++;
        count--;
    }
}

/*
 * Smooth-shaded scanline with actual alpha blending.
 * For the general case where we need full SRC_ALPHA/ONE_MINUS_SRC_ALPHA
 * blend with varying source color per pixel.
 *
 * Interpolated channels are in 24-bit fixed-point (value in bits 16-23).
 * We extract the 8-bit channel, blend with dest, and write.
 */
static inline void neon_smooth_blend_srcalpha_scanline(
    PIXEL *pp, GLint count,
    GLint r_start, GLint g_start, GLint b_start,
    GLint drdx, GLint dgdx, GLint dbdx) {
    /*
     * For smooth shading, TGL_BLEND_FUNC_RGB hardcodes sa=0xFF.
     * With sfactor=GL_SRC_ALPHA: source is multiplied by 255 (≈identity)
     * With dfactor=GL_ONE_MINUS_SRC_ALPHA: dest is multiplied by 0
     * Result: just write source color directly.
     */
    neon_smooth_write_scanline(pp, count, r_start, g_start, b_start,
                               drdx, dgdx, dbdx);
}

/*
 * Smooth-shaded scanline with additive blend (GL_ONE/GL_ONE).
 * result = saturate(interpolated_color + dest)
 */
static inline void neon_smooth_blend_additive_scanline(
    PIXEL *pp, GLint count,
    GLint r_start, GLint g_start, GLint b_start,
    GLint drdx, GLint dgdx, GLint dbdx) {
    GLint or1 = r_start, og1 = g_start, ob1 = b_start;

    while (count >= 8) {
        /* Compute 8 source pixels by interpolation */
        uint8_t src_pixels[32]; /* 8 pixels × 4 bytes (BGRA) */
        for (int i = 0; i < 8; i++) {
            src_pixels[i * 4 + 0] = (ob1 >> 16) & 0xFF; /* B */
            src_pixels[i * 4 + 1] = (og1 >> 16) & 0xFF; /* G */
            src_pixels[i * 4 + 2] = (or1 >> 16) & 0xFF; /* R */
            src_pixels[i * 4 + 3] = 0;                   /* A */
            or1 += drdx;
            og1 += dgdx;
            ob1 += dbdx;
        }

        uint8x8x4_t src = vld4_u8(src_pixels);
        uint8x8x4_t dst = vld4_u8((const uint8_t *)pp);

        uint8x8x4_t result;
        result.val[0] = vqadd_u8(dst.val[0], src.val[0]);
        result.val[1] = vqadd_u8(dst.val[1], src.val[1]);
        result.val[2] = vqadd_u8(dst.val[2], src.val[2]);
        result.val[3] = dst.val[3];

        vst4_u8((uint8_t *)pp, result);
        pp += 8;
        count -= 8;
    }

    /* Scalar remainder */
    while (count > 0) {
        GLuint dest = *pp;
        GLuint dr = (dest >> 16) & 0xFF;
        GLuint dg = (dest >> 8) & 0xFF;
        GLuint db = dest & 0xFF;
        GLuint rr = dr + ((or1 >> 16) & 0xFF); if (rr > 255) rr = 255;
        GLuint rg = dg + ((og1 >> 16) & 0xFF); if (rg > 255) rg = 255;
        GLuint rb = db + ((ob1 >> 16) & 0xFF); if (rb > 255) rb = 255;
        *pp = (rr << 16) | (rg << 8) | rb;
        or1 += drdx;
        og1 += dgdx;
        ob1 += dbdx;
        pp++;
        count--;
    }
}

/*
 * Smooth-shaded scanline dispatcher.
 */
static inline void neon_smooth_blend_scanline(PIXEL *pp, GLint count,
                                              GLint r_start, GLint g_start,
                                              GLint b_start,
                                              GLint drdx, GLint dgdx,
                                              GLint dbdx,
                                              GLuint sfactor, GLuint dfactor,
                                              GLuint blendeq) {
    if (blendeq == GL_FUNC_ADD) {
        if (sfactor == GL_SRC_ALPHA && dfactor == GL_ONE_MINUS_SRC_ALPHA) {
            /* sa=0xFF in smooth mode → blend degenerates to direct write */
            neon_smooth_write_scanline(pp, count, r_start, g_start, b_start,
                                       drdx, dgdx, dbdx);
            return;
        }
        if (sfactor == GL_ONE && dfactor == GL_ONE) {
            neon_smooth_blend_additive_scanline(pp, count, r_start, g_start,
                                                b_start, drdx, dgdx, dbdx);
            return;
        }
    }
    /* Fallback: use scalar TGL_BLEND_FUNC_RGB per pixel */
    {
        GLuint zbblendeq = blendeq; /* alias for TGL_BLEND_FUNC_RGB macro */
        GLint or1 = r_start, og1 = g_start, ob1 = b_start;
        while (count > 0) {
            TGL_BLEND_FUNC_RGB(or1, og1, ob1, (*pp));
            pp++;
            or1 += drdx;
            og1 += dgdx;
            ob1 += dbdx;
            count--;
        }
    }
}

/*
 * Smooth-shaded scanline with depth test (DT1), no depth write (DW0).
 * This is the HOT PATH for translucent dice faces.
 *
 * Since TGL_BLEND_FUNC_RGB hardcodes sa=0xFF:
 *   sfactor=GL_SRC_ALPHA → source * 255 / 255 ≈ source (identity)
 *   dfactor=GL_ONE_MINUS_SRC_ALPHA → dest * 0 = 0
 *   result = RGB_TO_PIXEL(r, g, b) — direct write, no blend math needed!
 *
 * NEON strategy: process 4 pixels per iteration.
 *   - Compute 4 z values, compare with 4 zbuffer entries (vectorized)
 *   - Pack 4 RGB pixels using shift+mask (identical to neon_smooth_write)
 *   - Use vbsl (bitwise select) to conditionally write only passing pixels
 *   - No branches in inner loop!
 */
static inline void neon_smooth_dt1_scanline(PIXEL *pp, GLushort *pz,
                                            GLint count,
                                            GLint r_start, GLint g_start,
                                            GLint b_start,
                                            GLint drdx, GLint dgdx,
                                            GLint dbdx,
                                            GLuint z_start, GLint dzdx_val) {
    /* Set up NEON vectors for 4-pixel processing */
    int32_t init_offsets[4] = {0, 1, 2, 3};
    int32x4_t offset = vld1q_s32(init_offsets);

    /* Color interpolation vectors (same as neon_smooth_write_scanline) */
    int32x4_t r_vec = vmlaq_s32(vdupq_n_s32(r_start), vdupq_n_s32(drdx), offset);
    int32x4_t g_vec = vmlaq_s32(vdupq_n_s32(g_start), vdupq_n_s32(dgdx), offset);
    int32x4_t b_vec = vmlaq_s32(vdupq_n_s32(b_start), vdupq_n_s32(dbdx), offset);

    int32x4_t stride_r = vdupq_n_s32(drdx * 4);
    int32x4_t stride_g = vdupq_n_s32(dgdx * 4);
    int32x4_t stride_b = vdupq_n_s32(dbdx * 4);

    /* Z interpolation vector */
    uint32x4_t z_vec = vmlaq_u32(vdupq_n_u32(z_start),
                                  vreinterpretq_u32_s32(vdupq_n_s32(dzdx_val)),
                                  vreinterpretq_u32_s32(offset));
    uint32x4_t z_stride = vdupq_n_u32((uint32_t)(dzdx_val * 4));

    /* RGB packing masks */
    uint32x4_t mask_r = vdupq_n_u32(0x00FF0000);
    uint32x4_t mask_g = vdupq_n_u32(0x0000FF00);
    uint32x4_t mask_b = vdupq_n_u32(0x000000FF);

    while (count >= 4) {
        /* Load 4 z-buffer values (16-bit) and zero-extend to 32-bit */
        uint16x4_t zbuf4 = vld1_u16(pz);
        uint32x4_t zbuf32 = vmovl_u16(zbuf4);

        /* Shift z >> ZB_POINT_Z_FRAC_BITS (14) */
        uint32x4_t zz4 = vshrq_n_u32(z_vec, ZB_POINT_Z_FRAC_BITS);

        /* Compare: zz >= zbuf (unsigned) → mask is all-ones where passes */
        uint32x4_t zmask = vcgeq_u32(zz4, zbuf32);

        /* Pack RGB pixels: (r & 0xff0000) | ((g >> 8) & 0xff00) | ((b >> 16) & 0xff) */
        uint32x4_t r_masked = vandq_u32(vreinterpretq_u32_s32(r_vec), mask_r);
        uint32x4_t g_shifted = vshrq_n_u32(vreinterpretq_u32_s32(g_vec), 8);
        uint32x4_t g_masked = vandq_u32(g_shifted, mask_g);
        uint32x4_t b_shifted = vshrq_n_u32(vreinterpretq_u32_s32(b_vec), 16);
        uint32x4_t b_masked = vandq_u32(b_shifted, mask_b);
        uint32x4_t new_pixels = vorrq_u32(vorrq_u32(r_masked, g_masked), b_masked);

        /* Load existing pixels and blend: write new where z passes, keep old otherwise */
        uint32x4_t old_pixels = vld1q_u32((const uint32_t *)pp);
        uint32x4_t result = vbslq_u32(zmask, new_pixels, old_pixels);
        vst1q_u32((uint32_t *)pp, result);

        /* Advance all interpolators by 4 pixels */
        r_vec = vaddq_s32(r_vec, stride_r);
        g_vec = vaddq_s32(g_vec, stride_g);
        b_vec = vaddq_s32(b_vec, stride_b);
        z_vec = vaddq_u32(z_vec, z_stride);

        pp += 4;
        pz += 4;
        count -= 4;
    }

    /* Scalar remainder */
    GLint or1 = vgetq_lane_s32(r_vec, 0);
    GLint og1 = vgetq_lane_s32(g_vec, 0);
    GLint ob1 = vgetq_lane_s32(b_vec, 0);
    GLuint z = vgetq_lane_u32(z_vec, 0);
    while (count > 0) {
        GLuint zz = z >> ZB_POINT_Z_FRAC_BITS;
        if (zz >= *pz) {
            *pp = (or1 & 0xff0000) | ((og1 >> 8) & 0xff00) | ((ob1 >> 16) & 0xff);
        }
        z += dzdx_val;
        or1 += drdx;
        og1 += dgdx;
        ob1 += dbdx;
        pp++;
        pz++;
        count--;
    }
}

/*
 * Smooth-shaded scanline dispatcher with depth test (DT1), no depth write.
 * Selects fast-path for SRC_ALPHA/ONE_MINUS_SRC_ALPHA (common dice blend).
 */
static inline void neon_smooth_blend_dt1_dispatch(PIXEL *pp, GLushort *pz,
                                                  GLint count,
                                                  GLint r_start, GLint g_start,
                                                  GLint b_start,
                                                  GLint drdx, GLint dgdx,
                                                  GLint dbdx,
                                                  GLuint z_start, GLint dzdx_val,
                                                  GLuint sfactor, GLuint dfactor,
                                                  GLuint blendeq) {
    if (blendeq == GL_FUNC_ADD &&
        sfactor == GL_SRC_ALPHA &&
        dfactor == GL_ONE_MINUS_SRC_ALPHA) {
        neon_smooth_dt1_scanline(pp, pz, count, r_start, g_start, b_start,
                                 drdx, dgdx, dbdx, z_start, dzdx_val);
        return;
    }
    /* Scalar fallback with depth test */
    {
        GLuint zbblendeq = blendeq;
        GLint or1 = r_start, og1 = g_start, ob1 = b_start;
        GLuint z = z_start;
        while (count > 0) {
            GLuint zz = z >> ZB_POINT_Z_FRAC_BITS;
            if (zz >= *pz) {
                TGL_BLEND_FUNC_RGB(or1, og1, ob1, (*pp));
            }
            z += dzdx_val;
            or1 += drdx;
            og1 += dgdx;
            ob1 += dbdx;
            pp++;
            pz++;
            count--;
        }
    }
}

/*
 * Flat-shaded scanline with depth test (DT1) and SRC_ALPHA blend.
 * Tests z per pixel; only blends pixels that pass.
 * No depth write (DW0).
 */
static inline void neon_flat_blend_srcalpha_dt1(PIXEL *pp, GLushort *pz,
                                                GLint count, GLuint color,
                                                GLuint z_start, GLint dzdx) {
    uint8_t sa = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;
    uint8_t inv_sa = 255 - sa;

    GLuint z = z_start;

    /* For depth-tested scanlines, pixels may be non-contiguous, so process
     * one at a time but with reduced overhead (no switch statements). */
    while (count > 0) {
        GLuint zz = z >> ZB_POINT_Z_FRAC_BITS;
        if (zz >= *pz) {
            GLuint dest = *pp;
            GLuint dr = (dest >> 16) & 0xFF;
            GLuint dg = (dest >> 8) & 0xFF;
            GLuint db = dest & 0xFF;
            GLuint rr = ((GLuint)sr * sa + dr * inv_sa + 128) >> 8;
            GLuint rg = ((GLuint)sg * sa + dg * inv_sa + 128) >> 8;
            GLuint rb = ((GLuint)sb * sa + db * inv_sa + 128) >> 8;
            *pp = (rr << 16) | (rg << 8) | rb;
        }
        z += dzdx;
        pp++;
        pz++;
        count--;
    }
}

#endif /* __ARM_NEON */
#endif /* ZTRIANGLE_NEON_H */
