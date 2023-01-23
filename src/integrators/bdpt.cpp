#include <mitsuba/render/integrator.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BDPTIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_hide_emitters, m_max_depth, m_rr_depth)
    MI_IMPORT_TYPES(Scene, Film, Sampler, ImageBlock, Emitter, EmitterPtr,
                    Sensor, SensorPtr, BSDF, BSDFPtr)

    BDPTIntegrator(const Properties &props) : Base(props) {
    }

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       Sampler *sampler,
                       ImageBlock *block,
                       Float *aovs,
                       const Vector2f &pos,
                       ScalarFloat diff_scale_factor,
                       Mask active) const override {
        const Film *film = sensor->film();
        const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);
        const bool box_filter = film->rfilter()->is_box_filter();

        ScalarVector2f scale = 1.f / ScalarVector2f(film->crop_size()),
                       offset = -ScalarVector2f(film->crop_offset()) * scale;

        Vector2f sample_pos   = pos + sampler->next_2d(active),
                 adjusted_pos = dr::fmadd(sample_pos, scale, offset);

        Point2f aperture_sample(.5f);
        if (sensor->needs_aperture_sample())
            aperture_sample = sampler->next_2d(active);

        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0.f)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        Float wavelength_sample = 0.f;
        if constexpr (is_spectral_v<Spectrum>)
            wavelength_sample = sampler->next_1d(active);

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            time, wavelength_sample, adjusted_pos, aperture_sample);

        if (ray.has_differentials)
            ray.scale_differential(diff_scale_factor);

        // Modifications for BDPT
        ScalarVector2u film_size = film->size(),
                       crop_size = film->crop_size();
        uint32_t spp = sampler->sample_count();
        ScalarFloat sample_scale =
            dr::prod(crop_size) / ScalarFloat(spp * dr::prod(film_size));
        // In order to accumulate and splat into a single block
        block->set_normalize(true);
        bool coalesce = block->coalesce();
        block->set_coalesce(false);

        auto [spec, valid] = sample(
            scene, sensor, sampler, ray, block, sample_scale
            );
        spec *= sample_scale;
        block->set_coalesce(coalesce);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(ray_weight * spec);

        Color3f rgb;
        if constexpr (is_spectral_v<Spectrum>)
            rgb = spectrum_to_srgb(spec_u, ray.wavelengths, active);
        else if constexpr (is_monochromatic_v<Spectrum>)
            rgb = spec_u.x();
        else
            rgb = spec_u;

        aovs[0] = rgb.x();
        aovs[1] = rgb.y();
        aovs[2] = rgb.z();

        if (unlikely(has_alpha)) {
            aovs[3] = dr::select(valid, Float(1.f), Float(0.f));
            aovs[4] = 0.f;
        } else {
            aovs[3] = 0.f;
        }
        //

        // With box filter, ignore random offset to prevent numerical instabilities
        block->put(box_filter ? pos : sample_pos, aovs, active);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     const Sensor *sensor,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     ImageBlock *block,
                                     ScalarFloat sample_scale) const {

        UInt32 n_camera_verts = generate_camera_subpath();

        UInt32 n_light_verts = generate_light_subpath();

        Spectrum result(0.f);

        // t = 1, s = 1
        // Sample emitter and sensor and connect

        // t = 1, s > 1
        // Sample sensor and connect
        UInt32 s = 1u;
        while (loop(active)) {
            Vertex vert = dr::gather<Vertex>();
            auto [ds, importance] = sensor->sample_direction();
            Spectrum L = importance * vert.throughput / ds.pdf;
            L *= mis_weight();
            block->put();
            s += 1;
        }
        // s = 0
        // Treat camera subpath as a complete path
        // s = 1, t > 1
        // Sample emitter and connect
        UInt32 t = 1u;
        while (loop(active)) {
            Vertex vert = dr::gather<Vertex>();
            Spectrum L = vert.throughput * vert.emitter();

            scene->sample_emitter_direction();
            L += dr::select(t > 1, L_, 0.f);

            result += L;
        }



        // t > 1, s > 1
        // The general case
        UInt32 t = 2u;
        while (loop(active_t)) {
            UInt32 s = 2u;
            while (loop(active_s)) {
                // Check for invalid combination

                // Connect
                result += connect_bdpt();
            }
        }
    }

    UInt32 random_walk(BSDFContext bsdf_ctx,
                       const Scene *scene,
                       const Ray3f &ray,
                       Vertex vertices,
                       UInt32 offset,
                       uint32_t max_depth,
                       Float pdf_fwd,
                       Spectrum throughput = 1.f,
                       Mask active = true) const {
        if (unlikely(max_depth == 0))
            return 0;

        UInt32 n_verts = 0;
        Vertex prev_vert = dr::gather<Vertex>();

        dr::Loop<Bool> loop("Random Walk", n_verts, prev_vert, active);
        loop.set_max_iterations(max_depth);

        while (loop(active)) {
            SurfaceInteraction3f si =
                scene->ray_intersect(ray);
            Vertex curr_vert(si);
            si.pdf_fwd = pdf_fwd;
            BSDFPtr bsdf = si.bsdf(ray);

            bsdf_ctx->reverse();
            bsdf->pdf(bsdf_ctx_rev, si, si.wi);
            bsdf_ctx->reverse();

        }
    }

    UInt32 generate_camera_subpath() const {

        // Perform random walk to construct subpath
    }

    UInt32 generate_light_subpath() const;

    Spectrum connect_bdpt() const {
        // Handle the case where the camera vertex is on a light and s != 0
        // by returning zero

        // Handle the case where s == 0
        // by treating the camera subpath as a complete path


        // Handle the case where t == 1
        // by connecting the light subpath to the camera

        // Handle the case where s == 1
        // by connecting the camera subpath to a light

        // Handle the general case

        // Multiply radiance by MIS weight
    }

    Float mis_weight() const;

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(BDPTIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(BDPTIntegrator, "Bidirectional Path Tracer integrator")
NAMESPACE_END(mitsuba)