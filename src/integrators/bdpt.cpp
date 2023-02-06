#include <tuple>
#include <drjit/struct.h>
#include <drjit/dynamic.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/vertex.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BDPTIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_hide_emitters, m_max_depth, m_rr_depth)
    MI_IMPORT_TYPES(Scene, Film, Sampler, ImageBlock, Emitter, EmitterPtr,
                    Sensor, SensorPtr, BSDF, BSDFPtr)

    BDPTIntegrator(const Properties &props) : Base(props) { }

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
        // Note wav_weight is already included in ray_weight, but we need
        // it when connecting light subpaths to the sensor
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

    // TODO: Take care of scalar mode
    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     const Sensor *sensor,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     Spectrum wav_weight,
                                     ImageBlock *block,
                                     ScalarFloat sample_scale) const {
//        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
        constexpr bool JIT = dr::is_jit_v<Float>;

        auto [n_camera_verts, verts_camera] = generate_camera_subpath(scene, sensor, sampler, ray);

        auto [n_light_verts, verts_light] = generate_light_subpath(scene, sampler, ray.time, ray.wavelengths);

#ifndef USE_MIS
        // When not using MIS, we fix all the vertices of a path
        // to be light vertices except for the first vertex.
        // So effectively the BDPT becomes a particle tracer
        // in this case.
        sample_visible_emitters(scene, sensor, sampler, block, sample_scale,
                                ray.time, ray.wavelengths, wav_weight);
        UInt32 depth = 1;
        UInt32 idx = dr::arange<UInt32>(dr::width(ray)) * m_max_depth;
        Mask active = depth < n_light_verts;
        dr::Loop<Bool> loop("Connect Sensor",
                            depth, idx, active);
        while (loop(active)) {
            idx++;

            Vertex3f vert = dr::gather<Vertex3f>(verts_light, idx);
            SurfaceInteraction3f si(vert);
            Point2f aperture_sample;
            if (sensor->needs_aperture_sample())
                aperture_sample = sampler->next_2d();
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, aperture_sample);
            Spectrum weight = vert.throughput * sensor_weight * wav_weight;
            connect_sensor(scene, si, sensor_ds, vert.bsdf, weight, block, sample_scale);

            // Check for termination
            depth++;
            active &= depth < n_light_verts;
        }

        return { 0.f, false };
#elif !defined(CONNECT)
        // When not connecting subpaths, we fix all the vertices of a path
        // to be camera vertices except for the last vertex.
        // For the last vertex, we use MIS to combine BSDF sampling
        // and light sampling. So effectively the BDPT becomes a path tracer
        // in this case.
#endif

#ifdef USE_MIS
#ifdef CONNECT
        // t = 1, s = 1
        // Sample emitter and sensor and connect
        // Sample emitter
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d());
        EmitterPtr emitter = dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);
        Mask is_inf = has_flag(emitter->flags(), EmitterFlags::Infinite);
        Spectrum throughput = emitter_idx_weight;
        Float pdf_emitter = dr::rcp(emitter_idx_weight);
        SurfaceInteraction si = dr::zeros<SurfaceInteraction3f>();
        // Sample direction or position for infinite or area light respectively
        if (dr::any_or<true>(is_inf)) {
            Interaction3f ref_it(0.f, ray.time, ray.wavelengths,
                                sensor->world_transform().translation());
            auto [ds, dir_weight] = emitter->sample_direction(
                ref_it, sampler->next_2d(is_inf), is_inf);
            // Convert to area measure
            throughput[is_inf] *= dir_weight * dr::sqr(ds.dist);
            pdf_emitter[is_inf] *= ds.pdf;
            si[is_inf] = SurfaceInteraction3f(ds, ray.wavelengths);
        }
        if (dr::any_or<true>(!is_inf)) {
            auto [ps, pos_weight] = emitter->sample_position(ray.time, sampler->next_2d(!is_inf), !is_inf);
            SurfaceInteraction3f si(ps, ray.wavelengths);
            throughput[!is_inf] *= pos_weight * emitter->eval(si, !is_inf);
            pdf_emitter[!is_inf] *= ps.pdf;
            si[!is_inf] = SurfaceInteraction3f(ps, ray.wavelengths);
        }
        Point2f aperture_sample;
        if (sensor->needs_aperture_sample()) {
            aperture_sample = sampler->next_2d();
        }
        auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, aperture_sample);
        // Convert to solid angle measure
        pdf_emitter[!is_inf] *= dr::sqr(sensor_ds.dist) * dr::dot(sensor_ds.d, si.n);
        Float pdf_sensor = sensor_ds.pdf;
        throughput *= sensor_weight;
        Float pdf_fwd = pdf_sensor * pdf_emitter;
        // Compute MIS weight
//        auto [pdf_pos, pdf_dir] = sensor->pdf_ray(Ray3f(sensor_ds.p, -sensor_ds.d));
//        Float pdf_dir = sensor->pdf_ray_dir(Ray3f(sensor_ds.p, -sensor_ds.d), sensor_ds);
//        Float pdf_rev = pdf_pos * pdf_dir;
//        Float mis_weight_ = pdf_fwd / (pdf_fwd + pdf_rev);

        // t = 1, s > 1
        // Sample sensor and connect
        UInt32 s = 1u;
        while (loop(active)) {
            Vertex3f vert = dr::gather<Vertex3f>();
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
#endif
#endif
    }

    /**
     * \brief Perform random walk to constrcut subpath
     *
     * \param bsdf_ctx
     *    Indicates whether subpath starts from sensor or emitter
     *
     * \param prev_vert
     *    Vertex to start from
     *
     * \param throughput
     *    Throughput of next vertex
     *
     * \param max_depth
     *    Maximum number of remaining vertices along subpath
     *
     * \param pdf_fwd
     *    Directional PDF of next vertex
     *
     * \return
     *    Number of vertices along subpath starting from `prev_vert`
     *    and those vertices
     */
    // TODO: Delta vertices ignored for now
    std::pair<UInt32, Vertex3f> random_walk(BSDFContext bsdf_ctx,
                       const Scene *scene,
                       Sampler *sampler,
                       const Ray3f &ray_,
                       Vertex3f prev_vert,
                       Spectrum throughput,
                       uint32_t max_depth,
                       Float pdf_fwd,
                       Mask active = true) const {
        if (unlikely(max_depth == 0))
            return { 1, prev_vert };

        uint32_t width = dr::width(ray_);
        auto vertices = dr::empty<Vertex3f>(width * (max_depth + 1));
        UInt32 offset = dr::arange<UInt32>(width) * (max_depth + 1);
        Ray3f ray = Ray3f(ray_);
        Mask active_ = active;
        UInt32 n_verts = 1;
        dr::Loop<Bool> loop("Random Walk", ray, n_verts, prev_vert, throughput, pdf_fwd, active);
        loop.set_max_iterations(max_depth);

        while (loop(active)) {
            // Find and populate next vertex
            SurfaceInteraction3f si =
                scene->ray_intersect(ray);
            Vertex3f curr_vert(prev_vert, si, pdf_fwd, throughput);
            if (bsdf_ctx.mode == TransportMode::Importance)
                curr_vert.emitter = si.emitter(scene);
            BSDFPtr bsdf = si.bsdf(ray);
            auto [bsdf_sample, bsdf_weight] = bsdf->sample(bsdf_ctx, si, sampler->next_1d(), sampler->next_2d());
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            // Compute previous vertex's pdf_rev
            bsdf_ctx.reverse();
            Vector3f wo = si.wi;
            si.wi = bsdf_sample.wo;
            Float pdf_bsdf = bsdf->pdf(bsdf_ctx, si, wo);
            Float pdf_pos = pdf_bsdf * dr::rcp(dr::sqr(si.t)) * dr::abs_dot(prev_vert.n, si.to_world(wo));
            si.wi = wo;
            bsdf_ctx.reverse();
            if (bsdf_ctx.mode == TransportMode::Radiance) {
                // Handle the case where the current vertex is on the environment map.
                // Note the reverse PDF is not correct for sensor vertices,
                // but we don't need PDFs of sensor vertices anyway
                Float pdf_env =
                    dr::rcp(dr::sqr(scene->bbox().bounding_sphere().radius) *
                            dr::Pi<Float>);
                pdf_env *= dr::abs_dot(prev_vert.n, wo);
                prev_vert.pdf_rev =
                    dr::select(si.is_valid(), pdf_pos, pdf_env);
            }
            else {
                // Use directional PDF for infinite lights
                Mask is_inf = has_flag(prev_vert.emitter->flags(), EmitterFlags::Infinite);
                prev_vert.pdf_rev = dr::select(is_inf, pdf_bsdf, pdf_pos);
            }

            // Scatter previous vertex into `vertices`
            UInt32 idx = offset + n_verts;
            dr::scatter(vertices, prev_vert, idx, active);

            // Update loop variables
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
            n_verts++;
            prev_vert = curr_vert;
            throughput *= bsdf_weight;
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
            Mask is_inf = dr::eq(prev_vert.dist, dr::Infinity<Float>);
            n_verts -= dr::select(active_ & is_inf, 1, 0);
            active_ &= !is_inf;
        }
        dr::scatter(vertices, prev_vert, idx, active_);

        Assert(!dr::any(n_verts > max_depth));

        return { n_verts, vertices };
    }

    auto generate_camera_subpath(
                                   const Scene *scene,
                                   const Sensor *sensor,
                                   Sampler *sampler,
                                   const RayDifferential3f &ray) const {
        auto [pdf_pos, pdf_dir] = sensor->pdf_ray(ray, dr::zeros<PositionSample3f>());
        Vertex3f vert(ray, pdf_pos);
        UInt32 offset = dr::arange<UInt32>(dr::width(ray)) * (m_max_depth + 1);

        return random_walk(BSDFContext(), scene, sampler,
                           ray, vert, 1.f, m_max_depth, pdf_dir);
    }

    // TODO: Delta lights ignored for now
    auto generate_light_subpath(
                                  const Scene *scene,
                                  Sampler *sampler,
                                  Float time,
                                  const Wavelength &wavelengths) const {
        // Sample an emitter
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d());
        EmitterPtr emitter =
            dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);
        Mask is_inf = has_flag(emitter->flags(), EmitterFlags::Infinite);

        // Sample ray from emitter
        // Note ray_weight includes radiance
        auto [ps, pdf_dir, ray, ray_weight] =
            emitter->pdf_sample_ray(time, wavelengths, sampler->next_2d(), sampler->next_2d());
        Float pdf_pos = ps.pdf;

        Spectrum throughput = emitter_idx_weight /
                              dr::select(is_inf,
                                         pdf_dir,
                                         pdf_pos);
        Float pdf_emitter = scene->pdf_emitter(emitter_idx);
        Float pdf_fwd = dr::select(is_inf, pdf_dir, pdf_pos) * pdf_emitter;
        Vertex3f vert(ray, ps, emitter, pdf_fwd, throughput);

        throughput = emitter_idx_weight * ray_weight;
        pdf_fwd = dr::select(is_inf, pdf_pos, pdf_dir);

        return random_walk(BSDFContext(TransportMode::Importance),
                                     scene, sampler, ray, vert,
                                     throughput, m_max_depth - 1, pdf_fwd);
    }

#ifdef CONNECT
    Spectrum connect_bdpt() const {
        return 1.f;
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
#endif

#ifdef USE_MIS
#ifdef CONNECT
    Float mis_weight(const Vertex3f &verts_camera,
                     const Vertex3f &verts_light,
                     UInt32 t, UInt32 s) const {
        return 1.f;

        Float sum_ri = 0;

        // TODO: Update vertex properties for current strategy

        Float ri = 1;
        while (loop(active_t)) {
            Vertex3f vert = dr::gather<>();
            ri *= vert.pdf_rev / vert.pdf_fwd;
            sum_ri += ri;
        }

        ri = 1;
        while (loop(active_s)) {
            Vertex3f vert = dr::gather<>();
            ri *= vert.pdf_rev / vert.pdf_fwd;
            sum_ri += ri;
        }

        return dr::rcp(1 + sum_ri);

    }
#else
    /**
     *  \brief MIS weight of one of two candidate strategies
     *  using the power heuristic
     */
    Float mis_weight() const;
#endif
#endif

#ifndef USE_MIS
    Spectrum connect_sensor(const Scene *scene,
                            const SurfaceInteraction3f &si,
                            const DirectionSample3f &sensor_ds,
                            const BSDFPtr &bsdf,
                            const Spectrum &weight,
                            ImageBlock *block,
                            ScalarFloat sample_scale,
                            Mask active = true) const {
        active &= (sensor_ds.pdf > 0.f) &&
                dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
        if (dr::none_or<false>(active))
            return 0.f;

        // Visibility test
        Ray3f sensor_ray = si.spawn_ray_to(sensor_ds.p);
        active &= !scene->ray_test(sensor_ray, active);
        if (dr::none_or<false>(active))
            return 0.f;

        Spectrum result = 0.f;
        Spectrum surface_weight = 1.f;
        Vector3f local_d = si.to_local(sensor_ray.d);
        Mask on_surface = active && dr::neq(si.shape, nullptr);
        if (dr::any_or<true>(on_surface)) {
            surface_weight[on_surface && dr::eq(bsdf, nullptr)] *=
                dr::maximum(0.f, Frame3f::cos_theta(local_d));

            on_surface &= dr::neq(bsdf, nullptr);
            if (dr::any_or<true>(on_surface)) {
                BSDFContext ctx(TransportMode::Importance);
                Float wi_dot_geo_n = dr::dot(si.n, si.to_world(si.wi)),
                      wo_dot_geo_n = dr::dot(si.n, sensor_ray.d);

                // Prevent light leaks due to shading normals
                Mask valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                             (wo_dot_geo_n * Frame3f::cos_theta(local_d) > 0.f);

                // Adjoint BSDF for shading normals
                Float correction = dr::select(valid,
                    dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                            (Frame3f::cos_theta(local_d) * wi_dot_geo_n)),
                    0.f);

                surface_weight[on_surface] *=
                correction * bsdf->eval(ctx, si, local_d, on_surface);
            }
        }

        Mask not_on_surface = active && dr::eq(si.shape, nullptr) && dr::eq(bsdf, nullptr);
        if (dr::any_or<true>(not_on_surface)) {
            Mask invalid_side = Frame3f::cos_theta(local_d) <= 0.f;
            surface_weight[not_on_surface && invalid_side] = 0.f;
        }

        result = weight * surface_weight * sample_scale;

        Float alpha = dr::select(dr::neq(bsdf, nullptr), 1.f, 0.f);
        Vector2f adjusted_position = sensor_ds.uv + block->offset();

        block->put(adjusted_position, si.wavelengths, result, alpha, 0.f, active);

        return result;
    }

    // TODO: Delta lights ignored for now
    void sample_visible_emitters(const Scene *scene,
                                 const Sensor *sensor,
                                 Sampler *sampler,
                                 ImageBlock *block,
                                 ScalarFloat sample_scale,
                                 Float time,
                                 const Wavelength &wavelengths,
                                 const Spectrum &wav_weight) const {
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d());
        EmitterPtr emitter =
            dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);

        Spectrum emitter_weight = dr::zeros<Spectrum>();
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        // Sample direction for infinite emitters
        Mask is_infinite = has_flag(emitter->flags(), EmitterFlags::Infinite);
        if (dr::any_or<true>(is_infinite)) {
            Interaction3f ref_it(0.f, time, wavelengths,
                                 sensor->world_transform().translation());
            auto [ds, dir_weight] = emitter->sample_direction(
                ref_it, sampler->next_2d(is_infinite), is_infinite);

            // Convert solid angle measure to area measure
            emitter_weight[is_infinite] =
                dr::select(ds.pdf > 0.f, dr::rcp(ds.pdf), 0.f) *
                dr::sqr(ds.dist);
            si[is_infinite] = SurfaceInteraction3f(ds, wavelengths);
        }

        // Sample position for finite emitters
        if (dr::any_or<true>(!is_infinite)) {
            auto [ps, pos_weight] =
            emitter->sample_position(time, sampler->next_2d(!is_infinite), !is_infinite);

            emitter_weight[!is_infinite] = pos_weight;
            si[!is_infinite] = SurfaceInteraction3f(ps, wavelengths);
        }

        // Sample direction toward sensor
        Point2f aperture_sample;
        if (sensor->needs_aperture_sample()) {
            aperture_sample = sampler->next_2d();
        }
        auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, aperture_sample);
        si.wi = sensor_ds.d;
        si.shape = emitter->shape();

        Spectrum radiance = emitter->eval(si);
        Spectrum weight = emitter_idx_weight * emitter_weight * sensor_weight * radiance * wav_weight;

        connect_sensor(scene, si, sensor_ds, nullptr, weight, block, sample_scale);
    }
#endif

    std::string to_string() const override {
        return tfm::format("BDPTIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(BDPTIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(BDPTIntegrator, "Bidirectional Path Tracer integrator")
NAMESPACE_END(mitsuba)