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

    Mask delta;

    Float J = 1.f;

    /// Direction from this vertex to next vertex
    Vector3f d;

    /// Distance from this vertex to next vertex
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
    Vertex(const SurfaceInteraction3f& si,
           const Scene *scene)
        : p(si.p), n(si.n), sh_frame(si.sh_frame), uv(si.uv),
          time(si.time), wavelengths(si.wavelengths), J(si.J),
          dist(si.t), emitter(si.emitter(scene)), bsdf(si.bsdf()) {}

    void zero_(size_t size = 1) {
        dist = dr::full<Float>(dr::Infinity<Float>, size);
        J = dr::full<Float>(1.f, size);
    }


    DRJIT_STRUCT(Vertex, p, n, sh_frame, uv, time, wavelengths,
                 pdf_fwd, pdf_rev, delta, J, d, dist, emitter,
                 bsdf, throughput)
};

NAMESPACE_END(mitsuba)
