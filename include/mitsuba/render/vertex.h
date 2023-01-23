#pragma once

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

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Fields
    // =============================================================

    Spectrum throughput;

    SurfaceInteraction3f ds;

    Float pdf_rev;

};

NAMESPACE_END(mitsuba)
