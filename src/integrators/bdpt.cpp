#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/vertex.h>

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
        auto [_, wav_weight] = sensor->sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
            wavelength_sample);

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
            scene, sensor, sampler, ray, wav_weight, block, sample_scale
            );
        spec *= sample_scale;
        block->set_coalesce(coalesce);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(spec * ray_weight);

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
                                     Spectrum wav_weight,
                                     ImageBlock *block,
                                     ScalarFloat sample_scale) const {

        UInt32 n_camera_verts = generate_camera_subpath(scene, sensor, sampler, ray);

        UInt32 n_light_verts = generate_light_subpath();

        Spectrum result(0.f);

        // t = 1, s = 1
        // Sample emitter and sensor and connect

        // t = 1, s > 1
        // Sample sensor and connect
        UInt32 s = 1u;
        while (loop(active)) {
            Vertex vert = dr::gather<Vertex3f>();
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

    // TODO: Delta distributions ignored for now
    UInt32 random_walk(BSDFContext bsdf_ctx,
                       const Scene *scene,
                       Sampler *sampler,
                       const Ray3f &ray_,
                       Vertex3f &vertices,
                       Vertex3f prev_vert,
                       UInt32 offset,
                       uint32_t max_depth,
                       Float pdf_fwd,
                       Mask active = true) const {
        if (unlikely(max_depth == 0))
            return 0;

        Ray3f ray = Ray3f(ray_);
        UInt32 n_verts = 0;
        dr::Loop<Bool> loop("Random Walk", ray, n_verts, prev_vert, pdf_fwd, active);
        loop.set_max_iterations(max_depth);

        while (loop(active)) {
            // Find next vertex
            SurfaceInteraction3f si =
                scene->ray_intersect(ray);
            Vertex3f curr_vert(si, pdf_fwd);
            BSDFPtr bsdf = si.bsdf(ray);
            auto [bsdf_sample, bsdf_weight] = bsdf->sample(bsdf_ctx, si, sampler->next_1d(), sampler->next_2d());
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);
            curr_vert.throughput = prev_vert.throughput * bsdf_weight;
            curr_vert.d = bsdf_sample.wo;
            curr_vert.emitter = si.emitter(scene);

            // Compute previous vertex's pdf_rev
            bsdf_ctx.reverse();
            // TODO: what if current vertex is the last vertex,
            // and bsdf_sample.wo thus doesn't make sense?
            // How do we compute pdf_rev then?
            Vector3f wo = si.wi;
            si.wi = bsdf_sample.wo;
            Float pdf_bsdf = bsdf->pdf(bsdf_ctx, si, wo);
            Float pdf_pos = pdf_bsdf * dr::rcp(dr::sqr(si.t)) * dr::abs_dot(prev_vert.n, si.to_world(wo));
            if (bsdf_ctx.mode == TransportMode::Radiance) {
                Float pdf_env =
                    dr::rcp(dr::sqr(scene->bbox().bounding_sphere().radius) *
                            dr::Pi<Float>);
                pdf_env *= dr::abs_dot(prev_vert.n, wo);
                prev_vert.pdf_rev =
                    dr::select(si.is_valid(), pdf_pos, pdf_env);
            }
            else {
                Mask is_inf = has_flag(prev_vert.emitter->flags(), EmitterFlags::Infinite);
                prev_vert.pdf_rev = dr::select(is_inf, pdf_bsdf, pdf_pos);
            }
            si.wi = wo;
            bsdf_ctx.reverse();

            // Scatter previous vertex into `vertices`
            UInt32 idx = offset + n_verts;
            dr::scatter(vertices, prev_vert, idx, active);

            // Update loop variables
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
            n_verts++;
            prev_vert = curr_vert;
            pdf_fwd = bsdf_sample.pdf;


            // Check for termination
            active &= si.is_valid();
            active &= n_verts < max_depth;
        }

        // Scatter last vertex
        UInt32 idx = offset + n_verts;
        // We allow environment maps to be the last vertex
        // of camera subpaths but not of light subpaths
        if (bsdf_ctx.mode == TransportMode::Importance) {
            active = dr::neq(prev_vert.dist, dr::Infinity<Float>);
        }
        else {
            active = true;
        }
        dr::scatter(vertices, prev_vert, idx, active);
        n_verts -= dr::select(active, 0, 1);

        return n_verts;
    }

    UInt32 generate_camera_subpath(const Scene *scene,
                                   const Sensor *sensor,
                                   Sampler *sampler,
                                   const RayDifferential3f &ray,
                                   Vertex3f &vertices) const {
        auto [pdf_pos, pdf_dir] = sensor->pdf_ray(ray);
        Vertex3f vert(ray, pdf_pos);
        UInt32 offset = dr::arange<UInt32>(dr::width(ray)) * (m_max_depth + 1);

        return random_walk(BSDFContext(), scene, sampler,
                           ray, vertices, vert,
                           offset, m_max_depth, pdf_dir) + 1;
    }

    // TODO: Delta lights ignored for now
    // TODO: Implement sample_*() for envmap
    UInt32 generate_light_subpath(const Scene *scene,
                                  Sampler *sampler,
                                  Float time,
                                  const Wavelength &wavelengths,
                                  Vertex3f &vertices) const {
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d());
        EmitterPtr emitter =
            dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);

        auto [ps, pos_weight] =
            emitter->sample_position(time, sampler->next_2d());
        auto [ray, ray_weight] =
            emitter->sample_ray_dir(time, wavelengths, sampler->next_2d(), ps);
        ray_weight *= pos_weight * emitter_idx_weight;

        Float pdf_pos = scene->pdf_emitter(emitter_idx) / pos_weight;
        Float pdf_dir = emitter->pdf_ray_dir(ray, ps);

        Vertex3f vert(ray, ps, emitter, pdf_pos, ray_weight);
        UInt32 idx = dr::arange<UInt32>(dr::width(time)) * m_max_depth;

        UInt32 n_verts = random_walk(BSDFContext(TransportMode::Importance),
                                     scene, sampler, ray, vertices, vert, idx, m_max_depth - 1, pdf_dir) + 1;

        // Correct PDF for infinite lights
        Mask is_inf = has_flag(emitter->flags(), EmitterFlags::Infinite);
        vert.pdf_fwd = dr::select(is_inf, pdf_dir, vert.pdf_fwd);
        dr::scatter(vertices, vert, idx, is_inf);
        idx++;
        vert = dr::gather<Vertex3f>(vertices, idx, is_inf);
        vert.pdf_fwd = pdf_pos * dr::abs_dot(vert.n, ray.d);
        dr::scatter(vertices, vert, idx, is_inf);

        return n_verts;
    }

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