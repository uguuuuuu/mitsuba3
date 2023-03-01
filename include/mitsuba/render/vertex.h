#pragma once

#include <mitsuba/core/frame.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/bsdf.h>

NAMESPACE_BEGIN(mitsuba)

// TODO: UV partials

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

    Float J = 1.f;

    /// Direction from this vertex to previous vertex in *local frame*
    Vector3f wi;

    /// Distance from this vertex to previous vertex
    Float dist = dr::Infinity<Float>;

    EmitterPtr emitter = nullptr;

    ShapePtr shape = nullptr;

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
          time(si.time), wavelengths(si.wavelengths), pdf_fwd(0.f),
          pdf_rev(0.f), J(si.J), wi(si.wi), dist(si.t), emitter(nullptr),
          shape(si.shape), throughput(throughput) {

        Mask is_inf = has_flag(prev.emitter->flags(), EmitterFlags::Infinite);
        Vector3f wi_world = si.to_world(si.wi);
        // If previous vertex is infinite light, `pdf` is area probability density.
        // Otherwise, `pdf` is directional
        pdf_fwd = pdf * dr::abs_dot(n, wi_world) *
                  dr::select(is_inf, 1.f, dr::rcp(dr::sqr(dist)));
        // If current vertex is infinite light, `pdf_fwd` stores directional probability density,
        // Otherwise, `pdf_fwd` stores surface area probability density
        pdf_fwd = dr::select(si.is_valid(),
                        pdf_fwd,
                        pdf);
    }

    /// Create a vertex from a sensor ray
    Vertex(const Ray3f &ray,
           Float pdf)
        : p(ray.o), n(0.f), sh_frame(n), uv(0.f),
          time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          pdf_rev(0.f), J(1.f), wi(0.f), dist(0.f),
          emitter(nullptr), shape(nullptr), throughput(1.f) {}

    /// Create a vertex from an emitter ray
    Vertex(const Ray3f &ray,
           const PositionSample3f &ps,
           EmitterPtr emitter,
           Float pdf,
           Spectrum throughput)
        : p(ray.o), n(ps.n), sh_frame(ps.n), uv(ps.uv),
          time(ray.time), wavelengths(ray.wavelengths), pdf_fwd(pdf),
          pdf_rev(0.f), J(ps.J), wi(0.f), dist(0.f),
          emitter(emitter), shape(emitter->shape()), throughput(throughput) {}

    void zero_(size_t size = 1) {
        dist = dr::full<Float>(dr::Infinity<Float>, size);
        J = dr::full<Float>(1.f, size);
    }

    BSDFPtr bsdf() const { return shape->bsdf(); }

    Mask is_delta() const {
        Mask bsdf_delta = has_flag(bsdf()->flags(), BSDFFlags::Delta);

        return bsdf_delta;
    };

    Mask is_delta_light() const {
        return has_flag(emitter->flags(), EmitterFlags::Delta);
    }

    Mask is_connectible() const {
        return !(dr::eq(dist, dr::Infinity<Float>) || is_delta());
    }

    //! @}
    // =============================================================

    DRJIT_STRUCT(Vertex, p, n, sh_frame, uv, time, wavelengths,
                 pdf_fwd, pdf_rev, J, wi, dist, emitter,
                 shape, throughput);
};

NAMESPACE_END(mitsuba)
