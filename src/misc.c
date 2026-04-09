#include "msghandling.h"
#include "zgl.h"

GLContext *gl_get_context(void)
{
    return &gl_ctx;
}

void glPolygonStipple(void *a)
{
#if TGL_HAS(POLYGON_STIPPLE)
    GLContext *c = gl_get_context();
#include "error_check.h"
    ZBuffer *zb = c->zb;

    memcpy(zb->stipplepattern, a, TGL_POLYGON_STIPPLE_BYTES);
    for (GLint i = 0; i < TGL_POLYGON_STIPPLE_BYTES; i++) {
        zb->stipplepattern[i] = ((GLubyte *) a)[i];
    }
#endif
}

void glopViewport(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint xsize, ysize, xmin, ymin, xsize_req, ysize_req;

    xmin = p[1].i;
    ymin = p[2].i;
    xsize = p[3].i;
    ysize = p[4].i;

    /* we may need to resize the zbuffer */

    if (c->viewport.xmin != xmin || c->viewport.ymin != ymin ||
        c->viewport.xsize != xsize || c->viewport.ysize != ysize) {
        xsize_req = xmin + xsize;
        ysize_req = ymin + ysize;

        if (c->gl_resize_viewport &&
            c->gl_resize_viewport(&xsize_req, &ysize_req) != 0) {
            gl_fatal_error("glViewport: error while resizing display");
        }
        if (xsize <= 0 || ysize <= 0) {
            gl_fatal_error("glViewport: size too small");
        }

        c->viewport.xmin = xmin;
        c->viewport.ymin = ymin;
        c->viewport.xsize = xsize;
        c->viewport.ysize = ysize;

        gl_eval_viewport();
    }
}
void glBlendFunc(GLenum sfactor, GLenum dfactor)
{
    GLParam p[3];
#include "error_check_no_context.h"
    p[0].op = OP_BlendFunc;
    p[1].i = sfactor;
    p[2].i = dfactor;
    gl_add_op(p);
    return;
}
void glopBlendFunc(GLParam *p)
{
    GLContext *c = gl_get_context();
    c->zb->sfactor = p[1].i;
    c->zb->dfactor = p[2].i;
}

void glBlendEquation(GLenum mode)
{
    GLParam p[2];
#include "error_check_no_context.h"
    p[0].op = OP_BlendEquation;
    p[1].i = mode;
    gl_add_op(p);
}
void glopBlendEquation(GLParam *p)
{
    GLContext *c = gl_get_context();
    c->zb->blendeq = p[1].i;
}

void glopPointSize(GLParam *p)
{
    GLContext *c = gl_get_context();
    c->zb->pointsize = p[1].f;
}
void glPointSize(GLfloat f)
{
    GLParam p[2];
    p[0].op = OP_PointSize;
#include "error_check_no_context.h"
    p[1].f = f;
    gl_add_op(p);
}

void glopEnableDisable(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint code = p[1].i;
    GLint v = p[2].i;

    switch (code) {
    case GL_CULL_FACE:
        c->cull_face_enabled = v;
        break;
    case GL_LIGHTING:
        c->lighting_enabled = v;
        break;
    case GL_COLOR_MATERIAL:
        c->color_material_enabled = v;
        break;
    case GL_TEXTURE_2D:
        c->texture_2d_enabled = v;
        break;
    case GL_BLEND:
        c->zb->enable_blend = v;
        break;
    case GL_NORMALIZE:
        c->normalize_enabled = v;
        break;
    case GL_DEPTH_TEST:
        c->zb->depth_test = v;
        break;
    case GL_POLYGON_OFFSET_FILL:
        if (v)
            c->offset_states |= TGL_OFFSET_FILL;
        else
            c->offset_states &= ~TGL_OFFSET_FILL;
        break;
    case GL_POLYGON_STIPPLE:
#if TGL_HAS(POLYGON_STIPPLE)
        c->zb->dostipple = v;
#endif
        break;
    case GL_POLYGON_OFFSET_POINT:
        if (v)
            c->offset_states |= TGL_OFFSET_POINT;
        else
            c->offset_states &= ~TGL_OFFSET_POINT;
        break;
    case GL_POLYGON_OFFSET_LINE:
        if (v)
            c->offset_states |= TGL_OFFSET_LINE;
        else
            c->offset_states &= ~TGL_OFFSET_LINE;
        break;
    default:
        if (code >= GL_LIGHT0 && code < GL_LIGHT0 + MAX_LIGHTS) {
            gl_enable_disable_light(code - GL_LIGHT0, v);
        } else {
            tgl_warning("glEnableDisable: 0x%X not supported.\n", code);
        }
        break;
    }
}

void glopShadeModel(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint code = p[1].i;
    c->current_shade_model = code;
}

void glopCullFace(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint code = p[1].i;
    c->current_cull_face = code;
}

void glopFrontFace(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint code = p[1].i;
    c->current_front_face = code;
}

void glopPolygonMode(GLParam *p)
{
    GLContext *c = gl_get_context();
    GLint face = p[1].i;
    GLint mode = p[2].i;

    switch (face) {
    case GL_BACK:
        c->polygon_mode_back = mode;
        break;
    case GL_FRONT:
        c->polygon_mode_front = mode;
        break;
    case GL_FRONT_AND_BACK:
        c->polygon_mode_front = mode;
        c->polygon_mode_back = mode;
        break;
    default:
        break;
    }
}

void glopPolygonOffset(GLParam *p)
{
    GLContext *c = gl_get_context();
    c->offset_factor = p[1].f;
    c->offset_units = p[2].f;
}

GLenum glGetError()
{
#if TGL_HAS(ERROR_CHECK)
    GLContext *c = gl_get_context();
    GLenum eflag = c->error_flag;
    if (eflag != GL_OUT_OF_MEMORY)
        c->error_flag = GL_NO_ERROR;
    return eflag;
#else
    return GL_NO_ERROR;
#endif
}

void glDrawBuffer(GLenum mode)
{
    GLContext *c = gl_get_context();
#include "error_check.h"
    if ((mode != GL_FRONT && mode != GL_NONE) || c->in_begin) {
#if TGL_HAS(ERROR_CHECK)
#define ERROR_FLAG GL_INVALID_OPERATION
#include "error_check.h"
#else
        return;
#endif
    }
    c->drawbuffer = mode;
}

void glReadBuffer(GLenum mode)
{
    GLContext *c = gl_get_context();
#include "error_check.h"
    if ((mode != GL_FRONT && mode != GL_NONE) || c->in_begin) {
#if TGL_HAS(ERROR_CHECK)
#define ERROR_FLAG GL_INVALID_OPERATION
#include "error_check.h"
#else
        return;
#endif
    }
    c->readbuffer = mode;
}

void glReadPixels(GLint x,
                  GLint y,
                  GLsizei width,
                  GLsizei height,
                  GLenum format,
                  GLenum type,
                  void *data)
{
    GLContext *c = gl_get_context();
#include "error_check.h"

    /* Accept GL_UNSIGNED_BYTE (used by raylib's rlReadScreenPixels) in addition
     * to the original GL_UNSIGNED_INT types. */
    GLint type_ok = 0;
#if TGL_FEATURE_RENDER_BITS == 32
    type_ok = (type == GL_UNSIGNED_INT || type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_BYTE);
#elif TGL_FEATURE_RENDER_BITS == 16
    type_ok = (type == GL_UNSIGNED_SHORT || type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_BYTE);
#endif

    if (c->readbuffer != GL_FRONT ||
        (format != GL_RGBA && format != GL_RGB &&
         format != GL_DEPTH_COMPONENT) ||
        !type_ok) {
#if TGL_HAS(ERROR_CHECK)
#define ERROR_FLAG GL_INVALID_OPERATION
#include "error_check.h"
#else
        return;
#endif
    }

#if TGL_FEATURE_RENDER_BITS == 32
    if (data == NULL) return;
    ZBuffer *zb = c->zb;
    GLubyte *dst = (GLubyte *)data;

    for (GLint j = 0; j < height; j++) {
        /* OpenGL y=0 is bottom; TinyGL pbuf y=0 is top */
        GLint src_y = (zb->ysize - 1) - (y + j);
        if (src_y < 0 || src_y >= zb->ysize) continue;
        PIXEL *row = zb->pbuf + src_y * zb->xsize;

        for (GLint i = 0; i < width; i++) {
            GLint src_x = x + i;
            if (src_x < 0 || src_x >= zb->xsize) continue;
            PIXEL p = row[src_x];
            /* TinyGL 32-bit pixel layout: 0xAARRGGBB */
            GLubyte r = (p >> 16) & 0xFF;
            GLubyte g = (p >>  8) & 0xFF;
            GLubyte b =  p        & 0xFF;
            GLubyte a = (p >> 24) & 0xFF;
            if (a == 0) a = 255;  /* default opaque if alpha was never set */

            if (type == GL_UNSIGNED_BYTE) {
                if (format == GL_RGBA) {
                    *dst++ = r; *dst++ = g; *dst++ = b; *dst++ = a;
                } else if (format == GL_RGB) {
                    *dst++ = r; *dst++ = g; *dst++ = b;
                }
            } else {
                /* GL_UNSIGNED_INT: write as packed 32-bit */
                ((GLuint *)dst)[0] = p;
                dst += 4;
            }
        }
    }
#endif
}

void glFinish()
{
    return;
}

void gl_eval_viewport()
{
    GLContext *c = gl_get_context();
    GLViewport *v;
    GLfloat zsize = (1 << (ZB_Z_BITS + ZB_POINT_Z_FRAC_BITS));

    v = &c->viewport;

    v->trans.X = ((v->xsize - 0.5) / 2.0) + v->xmin;
    v->trans.Y = ((v->ysize - 0.5) / 2.0) + v->ymin;
    v->trans.Z = ((zsize - 0.5) / 2.0) + ((1 << ZB_POINT_Z_FRAC_BITS)) / 2;

    v->scale.X = (v->xsize - 0.5) / 2.0;
    v->scale.Y = -(v->ysize - 0.5) / 2.0;
    v->scale.Z = -((zsize - 0.5) / 2.0);
}

GLint gl_clipcode(GLfloat x, GLfloat y, GLfloat z, GLfloat w1)
{
    GLfloat w;

    w = w1 * (1.0 + CLIP_EPSILON);
    return (x < -w) | ((x > w) << 1) | ((y < -w) << 2) | ((y > w) << 3) |
           ((z < -w) << 4) | ((z > w) << 5);
}

GLfloat clampf(GLfloat a, GLfloat min, GLfloat max)
{
    if (a < min)
        return min;
    else if (a > max)
        return max;
    else
        return a;
}
