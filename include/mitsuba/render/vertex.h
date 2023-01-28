#pragma once

#include <drjit/struct.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

template<typename Float_, typename Spectrum_>
struct Vertex {
    // =============================================================
    //! @{ \name Type declarations
    // =============================================================

    using Float = Float_;
    using Spectrum = Spectrum_;
    MI_IMPORT_RENDER_BASIC_TYPES()
    MI_IMPORT_OBJECT_TYPES()
    using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;
    using PositionSample3f = typename RenderAliases::PositionSample3f;
    using DirectionSample3f = typename RenderAliases::DirectionSample3f;


    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Fields
    // =============================================================

    Point3f p;

    Normal3f n;

    Frame3f sh_frame;

    Point2f uv;

    Float time;

    Wavelength wavelengths;

    Float pdf_fwd = 0.f;

    Float pdf_rev = 0.f;

//    Mask delta;

    Float J = 1.f;

    /// Direction from this vertex to next vertex
    Vector3f d;

    /// Distance from previous vertex to this vertex
    Float dist;

    EmitterPtr emitter = nullptr;

    BSDFPtr bsdf = nullptr;

    Spectrum throughput = 0.f;

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Methods
    // =============================================================

    /// Constructor
    Vertex(const SurfaceInteraction3f &si,
           Float pdf)
        : p(si.p), n(si.n), sh_frame(si.sh_frame), uv(si.uv),
          time(si.time), wavelengths(si.wavelengths), J(si.J),
          dist(si.t), bsdf(si.bsdf()) {
        pdf_fwd = dr::select(si.is_valid(),
                        pdf * dr::rcp(dr::sqr(dist)) * dr::abs_dot(n, si.wi),
                        pdf);
    }

    Vertex(const Ray3f &ray,
           Float pdf)
        : p(ray.o), time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          J(1.f), throughput(1.f) {}

    Vertex(const Ray3f &ray,
           const PositionSample3f &ps,
           EmitterPtr emitter,
           Float pdf,
           Spectrum throughput)
        : p(ray.o), n(ps.n), sh_frame(ps.n), uv(ps.uv),
          time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          J(ps.J), throughput(throughput) {}

    void zero_(size_t size = 1) {
        dist = dr::full<Float>(dr::Infinity<Float>, size);
        J = dr::full<Float>(1.f, size);
    }

//    void set_emitter(const Scene *scene,
//                     const SurfaceInteraction3f &si,
//                     Mask active) {
//        emitter = si.emitter(scene, active);
//    }
//
//    void sample(const BSDFContext &ctx,
//                Vertex &prev,
//                Float sample1,
//                Point2f sample2) {
//        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
//        si.
//        bsdf->eval_pdf_sample(ctx, )
//    }

    //! @}
    // =============================================================

    DRJIT_STRUCT(Vertex, p, n, sh_frame, uv, time, wavelengths,
                 pdf_fwd, pdf_rev, J, d, dist, emitter,
                 bsdf, throughput)
};

NAMESPACE_END(mitsuba)
