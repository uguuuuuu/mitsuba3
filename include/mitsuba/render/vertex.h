#pragma once

#include <mitsuba/core/frame.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
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

    /// Direction from this vertex to previous vertex
    Vector3f d;

    /// Distance from this vertex to previous vertex
    Float dist = dr::Infinity<Float>;

    EmitterPtr emitter = nullptr;

    BSDFPtr bsdf = nullptr;

    Spectrum throughput = 0.f;

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Methods
    // =============================================================

    /**
     * \brief Create a vertex from a surface interaction
     *
     * Used to create intermediate vertices
     */
    Vertex(const Vertex &prev,
           const SurfaceInteraction3f &si,
           Float pdf,
           Spectrum throughput)
        : p(si.p), n(si.n), sh_frame(si.sh_frame), uv(si.uv),
          time(si.time), wavelengths(si.wavelengths), J(si.J),
          d(si.wi), dist(si.t), bsdf(si.bsdf()), throughput(throughput) {

        Mask is_inf = has_flag(prev.emitter->flags(), EmitterFlags::Infinite);
        // If previous vertex is infinite light, `pdf` is area probability density.
        // Otherwise, `pdf` is directional
        pdf_fwd = pdf * dr::abs_dot(n, si.wi) *
                  dr::select(is_inf, 1.f, dr::rcp(dr::sqr(dist)));
        // If next vertex is infinite light, `pdf_fwd` stores directional probability density,
        // Otherwise, `pdf_fwd` stores surface area probability density
        pdf_fwd = dr::select(si.is_valid(),
                        pdf_fwd,
                        pdf);
    }

    /// Create a vertex from a sensor ray
    Vertex(const Ray3f &ray,
           Float pdf)
        : p(ray.o), time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          J(1.f), throughput(1.f) {}

    /// Create a vertex from an emitter ray
    Vertex(const Ray3f &ray,
           const PositionSample3f &ps,
           EmitterPtr emitter,
           Float pdf,
           Spectrum throughput)
        : p(ray.o), n(ps.n), sh_frame(ps.n), uv(ps.uv),
          time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          J(ps.J), emitter(emitter), throughput(throughput) {}

    void zero_(size_t size = 1) {
        dist = dr::full<Float>(dr::Infinity<Float>, size);
        J = dr::full<Float>(1.f, size);
    }

    //! @}
    // =============================================================

    DRJIT_STRUCT(Vertex, p, n, sh_frame, uv, time, wavelengths,
                 pdf_fwd, pdf_rev, J, d, dist, emitter,
                 bsdf, throughput);
};

NAMESPACE_END(mitsuba)
