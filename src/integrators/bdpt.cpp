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

#define USE_MIS
#define CONNECT

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BDPTIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_hide_emitters, m_max_depth, m_rr_depth)
    MI_IMPORT_TYPES(Scene, Film, Sampler, ImageBlock, Emitter, EmitterPtr,
                    Sensor, SensorPtr, BSDF, BSDFPtr, Medium)

    using Vertex3fArray = std::unique_ptr<Vertex3f[]>;

    BDPTIntegrator(const Properties &props) : Base(props) {}

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
            scene, sensor, sampler, ray, wav_weight, block, sample_scale, active
            );

#if defined(CONNECT) || defined(USE_MIS)
        spec *= ray_weight;
#ifdef CONNECT
        spec *= sample_scale;
        Float weight = 0.f;
#else
        block->set_normalize(false);
        Float weight = 1.f;
#endif
        block->set_coalesce(coalesce);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(spec);

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
            aovs[4] = weight;
        } else {
            aovs[3] = weight;
        }
        //

        // With box filter, ignore random offset to prevent numerical instabilities
        block->put(box_filter ? pos : sample_pos, aovs, active);
#else
        DRJIT_MARK_USED(has_alpha);
        DRJIT_MARK_USED(box_filter);
        DRJIT_MARK_USED(coalesce);
        DRJIT_MARK_USED(aovs);
        DRJIT_MARK_USED(spec);
        DRJIT_MARK_USED(valid);
#endif
    }

    std::pair<Spectrum, Mask> sample(const Scene * /* scene */,
                                     Sampler * /* sampler */,
                                     const RayDifferential3f & /* ray */,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask /* active */) const override {
        NotImplementedError("sample");
    }

    // TODO: Doesn't perform correctly when `m_hide_emitters` is true
    // TODO: Can we still use wavefront-style recorded loop instead of for-loop?
    // TODO: Take care of scalar mode
    // TODO: Null materials ignored for now
    // TODO: Not computing uv partials at vertices
    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     const Sensor *sensor,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     Spectrum wav_weight,
                                     ImageBlock *block,
                                     ScalarFloat sample_scale,
                                     Mask active = true) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

#if defined(CONNECT) || defined(USE_MIS)
        Vertex3fArray verts_camera(new Vertex3f[m_max_depth]);
        UInt32 n_camera_verts = generate_camera_subpath(scene, sensor, sampler, verts_camera.get(), ray, active);
#endif
#if defined(CONNECT) || !defined(USE_MIS)
        Vertex3fArray verts_light(new Vertex3f[m_max_depth]);
        UInt32 n_light_verts = generate_light_subpath(scene, sampler, verts_light.get(), ray.time, ray.wavelengths, active);
#endif

#ifndef CONNECT
#ifndef USE_MIS
        // When not using MIS, we fix all the vertices of a path
        // to be light vertices except for the first vertex.
        // So effectively the BDPT becomes a particle tracer
        // in this case.
        sample_visible_emitters(scene, sensor, sampler, block, sample_scale,
                                ray.time, ray.wavelengths, wav_weight, active);
        UInt32 i      = 1;
//        UInt32 offset = dr::arange<UInt32>(width) * (m_max_depth - 1);
        active &= i < n_light_verts;
//        dr::Loop<Bool> loop("Connect Sensor", i, active);
//        while (loop(active)) {
        for (uint32_t i_ = 1; i_ < m_max_depth; i_++) {
//            UInt32 idx = offset + i;
//            Vertex3f vert = dr::zeros<Vertex3f>();
//            if constexpr (JIT) {
//                vert = dr::gather<Vertex3f>(verts_light, idx, active);
//            }
            Vertex3f vert = verts_light[i_];
            SurfaceInteraction3f si(vert);
            Point2f aperture_sample;
            if (sensor->needs_aperture_sample())
                aperture_sample = sampler->next_2d(active);
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, aperture_sample, active);
            Spectrum weight = vert.throughput * sensor_weight * wav_weight;
            connect_sensor(scene, si, sensor_ds, vert.bsdf(), weight, block, sample_scale, active);

            // Check for termination
            active &= (i + 1) < n_light_verts;
            i += dr::select(active, 1, 0);
        }

        return { 0.f, false };
#else // #ifdef USE_MIS
        // When not connecting subpaths, we fix all the vertices of a path
        // to be camera vertices except for the last vertex.
        // For the last vertex, we use MIS to combine BSDF sampling
        // and light sampling. So effectively the BDPT becomes a path tracer
        // in this case.
        Mask valid_ray = !m_hide_emitters && dr::neq(scene->environment(), nullptr);
        Vertex3f prev_vert = dr::zeros<Vertex3f>();
        Spectrum result = 0.f;
        UInt32 i = 0;
        active &= i < n_camera_verts;

        for (uint32_t i_ = 0; i_ < m_max_depth; i_++) {
            Vertex3f vert = verts_camera[i_];
            SurfaceInteraction3f prev_si(prev_vert), si(vert);
            DirectionSample3f ds(scene, si, prev_si);

            // BSDF sampling
            Mask active_bsdf = active && i > 0;
            Float pdf_bsdf = vert.pdf_fwd;
            // Convert area measure to solid angle measure
            pdf_bsdf *= dr::select(si.is_valid(),
                                   dr::sqr(vert.dist) / dr::abs_dot(vert.d, vert.n),
                                   1.f);
            Float pdf_em = scene->pdf_emitter_direction(prev_si, ds, active_bsdf);
            Float weight_mis = dr::select(active_bsdf, mis_weight(pdf_bsdf, pdf_em), 1.f);
            result = spec_fma(vert.throughput, si.emitter(scene, active)->eval(si, active) * weight_mis, result);

            // Emitter sampling
            Mask active_em = active && (i + 1) < m_max_depth;
            Spectrum weight_emitter = 0.f, bsdf_val = 0.f;
            pdf_bsdf = 0.f; weight_mis = 0.f;
            DirectionSample3f ds_emitter = dr::zeros<DirectionSample3f>();
            std::tie(ds_emitter, weight_emitter) = scene->sample_emitter_direction(
                                            si, sampler->next_2d(active_em), true, active_em);
            active_em &= dr::neq(ds_emitter.pdf, 0.f);
            Vector3f wo = si.to_local(ds_emitter.d);
            std::tie(bsdf_val, pdf_bsdf) = vert.bsdf()->eval_pdf(BSDFContext(), si, wo, active_em);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
            weight_mis = mis_weight(ds_emitter.pdf, pdf_bsdf);
            result[active_em] = spec_fma(vert.throughput,
                              bsdf_val * weight_emitter * weight_mis,
                              result);
            // Update loop variables
            prev_vert = vert;
            i += dr::select(active, 1, 0);
            active &= i < n_camera_verts;
        }

        DRJIT_MARK_USED(wav_weight);
        DRJIT_MARK_USED(block);
        DRJIT_MARK_USED(sample_scale);
        return { result, valid_ray };
#endif // #ifndef USE_MIS
#endif // #ifndef CONNECT

#ifdef CONNECT
        // t = 0, s = 1
        // Sample emitter and sensor
        dr::eval(
            sample_visible_emitters(scene, sensor, sampler, block, sample_scale,
                                ray.time, ray.wavelengths, wav_weight, active)
            );

        // t = 0, s > 1
        // Sample sensor and connect
        for (uint32_t s = 2; s < m_max_depth + 1; s++) {
            Mask active_s = active && (s - 1) < n_light_verts;
            Vertex3f vert = verts_light[s - 1];
            SurfaceInteraction3f si(vert);
            active_s &= vert.is_connectible();

            // Sample a point on sensor aperture
            Point2f aperture_sample;
            if (sensor->needs_aperture_sample())
                aperture_sample = sampler->next_2d(active_s);
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, aperture_sample, active_s);
            active_s &= sensor_ds.pdf > 0.f;

            // Compute reverse PDF of last vertex of light subpath
            Ray3f ray(sensor_ds.p, -sensor_ds.d);
            auto [_, pdf_rev] = sensor->pdf_ray(ray, sensor_ds, active_s);
            pdf_rev *= dr::rcp(dr::sqr(sensor_ds.dist)) * dr::abs_dot(si.n, sensor_ds.d);
            vert.pdf_rev = pdf_rev;
            Float weight_mis =
                mis_weight(nullptr, verts_light.get(),
                           0, s, vert, active_s);

            Spectrum weight = vert.throughput * sensor_weight * wav_weight * weight_mis;
            dr::eval(
                connect_sensor(scene, si, sensor_ds, vert.bsdf(), weight, block, sample_scale, active_s)
                );
        }

        Spectrum result = 0.f;
        // s = 0, t > 0
        // Treat camera subpath as a complete path
        Vertex3f vert_prev(ray, 0.f);
        for (uint32_t t = 1; t < m_max_depth + 1; t++) {
            Vertex3f vert = verts_camera[t - 1];
            SurfaceInteraction3f si(vert), si_prev(vert_prev);
            DirectionSample3f ds(scene, si, si_prev);
            Mask active_t = active && (t - 1) < n_camera_verts;
            active_t &= dr::neq(vert.emitter, nullptr);
            Mask is_inf = dr::eq(vert.dist, dr::Infinity<Float>);

            // Compute reverse PDF of last vertex of camera subpath
            Float pdf_rev = scene->pdf_emitter_direction(si_prev, ds, active_t);
            pdf_rev *= dr::select(is_inf, 1.f, dr::rcp(dr::sqr(ds.dist)) * dr::abs_dot(ds.n, ds.d));
            vert.pdf_rev = pdf_rev;
            Float weight_mis = mis_weight(verts_camera.get(), nullptr,
                                          t, 0, vert, active_t);

            result = spec_fma(vert.throughput,
                              vert.emitter->eval(si, active_t) * weight_mis,
                              result);
            vert_prev = vert;
            dr::eval(result);
        }

        // s = 1, t > 0
        // Sample emitter and connect
        for (uint32_t t = 1; t < m_max_depth; t++) {
            Mask active_t = active && (t - 1) < n_camera_verts;
            Vertex3f vert_camera = verts_camera[t - 1];
            SurfaceInteraction3f si(vert_camera);
            active_t &= vert_camera.is_connectible();

            // Create the light vertex
            auto [ds_emitter, weight_emitter] =
                scene->sample_emitter_direction(si, sampler->next_2d(active_t), true, active_t);
            active_t &= ds_emitter.pdf > 0;
            Ray3f ray(ds_emitter.p, -ds_emitter.d);
            Float pdf_fwd = ds_emitter.pdf;
            Vertex3f vert_light(ray, ds_emitter, ds_emitter.emitter, pdf_fwd, weight_emitter);

            Float weight_mis = mis_weight(verts_camera.get(), nullptr,
                                          t, 1, vert_light, active_t);

            BSDFPtr bsdf = vert_camera.bsdf();
            Vector3f wo = si.to_local(ds_emitter.d);
            Spectrum bsdf_val = bsdf->eval(BSDFContext(), si, wo, active_t);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

            result = spec_fma(vert_camera.throughput, bsdf_val * weight_emitter * weight_mis, result);
            dr::eval(result);
        }

        // t > 0, s > 1
        // Connect in the middle
        for (uint32_t t = 1; t < m_max_depth - 1; t++) {
            Vertex3f vert_camera = verts_camera[t - 1];
            Mask active_t = active && (t - 1) < n_camera_verts;
            active_t &= vert_camera.is_connectible();

            for (uint32_t s = 2; s < m_max_depth + 1 - t; s++) {
                Vertex3f vert_light = verts_light[s - 1];
                Mask active_s = active_t && (s - 1) < n_light_verts;
                active_s &= vert_light.is_connectible();

                Float weight_mis = mis_weight(verts_camera.get(), verts_light.get(),
                                              t, s, dr::zeros<Vertex3f>(), active_s);

                Spectrum throughput = connect_bdpt(scene, vert_camera, vert_light, active_s);

                result = spec_fma(throughput, weight_mis, result);
                dr::eval(result);
            }
        }

        return { result, n_camera_verts > 0 };
#endif // #ifdef CONNECT
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
     *    Maximum number of vertices along subpath
     *
     * \param pdf_fwd
     *    Directional PDF of next vertex
     *
     * \return
     *    Number of valid vertices stored
     */
    // TODO: Delta lights ignored for now
    // TODO: Can we apply Russian Roulette?
    UInt32 random_walk(BSDFContext bsdf_ctx,
                       const Scene *scene,
                       Sampler *sampler,
                       uint32_t max_depth,
                       Vertex3f* vertices,
                       const Ray3f &ray_,
                       const Vertex3f &prev_vert_,
                       const Spectrum &throughput_,
                       Float pdf_fwd_,
                       Mask active_ = true) const {
        if (bsdf_ctx.mode == TransportMode::Radiance)
            // We don't need to store the first vertex for camera subpaths
            max_depth--;
        if (unlikely(max_depth == 0))
            return 0;

/**********Using a single dynamic array to store the vertices costs too much memory *********/
//        constexpr bool JIT = dr::is_jit_v<Float>;
//        uint32_t width = dr::width(ray_);
//        auto vertices = dr::empty<Vertex3f>(width * max_depth);
//        UInt32 offset = dr::arange<UInt32>(width) * max_depth;
/********************************************************************************************/

        Ray3f ray = Ray3f(ray_);
        UInt32 n_verts = 0;
        Vertex3f prev_vert = prev_vert_;
        Spectrum throughput = throughput_;
        Float pdf_fwd = pdf_fwd_;
        Mask active = active_;

        // We start from the second vertex for camera subpaths
        // since we don't need to store the sensor
        if (bsdf_ctx.mode == TransportMode::Radiance) {
            Mask active_next = active;
            active_next &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            active_next &= !dr::eq(prev_vert.dist, dr::Infinity<Float>);
            SurfaceInteraction3f si = scene->ray_intersect(ray, active_next);
            if (!(bsdf_ctx.mode == TransportMode::Radiance && scene->environment() && !m_hide_emitters)) {
                active_next &= si.is_valid();
            }

            Vertex3f curr_vert = Vertex3f(prev_vert, si, pdf_fwd, throughput);
            if (bsdf_ctx.mode == TransportMode::Radiance)
                curr_vert.emitter = si.emitter(scene, active_next);

            BSDFPtr bsdf = si.bsdf(ray);
            auto [bsdf_sample, bsdf_weight] =
                bsdf->sample(bsdf_ctx, si, sampler->next_1d(active_next), sampler->next_2d(active_next), active_next);
            if (bsdf_ctx.mode == TransportMode::Radiance) {
                bsdf_weight =
                    si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);
            }
            else {
                bsdf_weight =
                    si.to_world_mueller(bsdf_weight, -si.wi, bsdf_sample.wo);

                // Ajoint BSDF for shading normals
                Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                      wo_dot_geo_n = dr::dot(si.n, si.to_world(bsdf_sample.wo));
                Mask valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                             (wo_dot_geo_n * Frame3f::cos_theta(bsdf_sample.wo) > 0.f);
                Float correction = dr::select(valid & active_next,
                                              dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                                      (Frame3f::cos_theta(bsdf_sample.wo) * wi_dot_geo_n)),
                                              0.f);
                bsdf_weight *= correction;
            }

            prev_vert = dr::select(active_next, curr_vert, prev_vert);
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
            throughput *= bsdf_weight;
            pdf_fwd = bsdf_sample.pdf;
            active &= active_next;
        }

//        dr::Loop<Bool> loop("Random Walk", sampler, ray, n_verts, prev_vert, throughput, pdf_fwd, active);
//        loop.set_max_iterations(max_depth);
//        while (loop(active)) {

        // Collect vertices
        for (uint32_t i = 0; i < max_depth; i++) {
            Mask is_inf = dr::eq(prev_vert.dist, dr::Infinity<Float>);
            Mask not_zero = dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            Mask active_next = active && !is_inf && not_zero && ((n_verts + 1) < max_depth);

            // Find next vertex
            SurfaceInteraction3f si =
                scene->ray_intersect(ray, active_next);
            // Only in camera subpaths do we allow environment map vertices to be valid
            if (!(scene->environment() && bsdf_ctx.mode == TransportMode::Radiance))
                active_next &= si.is_valid();
            Vertex3f curr_vert(prev_vert, si, pdf_fwd, throughput);
            // We allow camera subpaths to be complete paths on their own.
            // So store emitters to query radiance later
            if (bsdf_ctx.mode == TransportMode::Radiance)
                curr_vert.emitter = si.emitter(scene, active_next);

            // Sample next direction
            BSDFPtr bsdf = si.bsdf(ray);
            auto [bsdf_sample, bsdf_weight] =
                bsdf->sample(bsdf_ctx, si, sampler->next_1d(active_next), sampler->next_2d(active_next), active_next);
            if (bsdf_ctx.mode == TransportMode::Radiance) {
                bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);
            }
            else {
                bsdf_weight = si.to_world_mueller(bsdf_weight, -si.wi, bsdf_sample.wo);
                Float correction = bsdf_correction(si, bsdf_sample.wo);
                bsdf_weight *= correction;
            }

            // Compute previous vertex's pdf_rev
            bsdf_ctx.reverse();
            Vector3f wo = si.wi;
            si.wi = bsdf_sample.wo;
            Float pdf_dir = bsdf->pdf(bsdf_ctx, si, wo, active_next);
            Float pdf_pos = pdf_dir * dr::rcp(dr::sqr(si.t)) * dr::abs_dot(prev_vert.n, si.to_world(wo));
            si.wi = wo;
            bsdf_ctx.reverse();
            if (bsdf_ctx.mode == TransportMode::Radiance) {
                if (scene->environment()) {
                    // If the current vertex is on the environment map,
                    // the previous vertex's reverse PDF is computed in a different way
                    auto envmap                     = scene->environment();
                    Ray3f ray(si.p, si.wi);
                    auto [pdf_env, _] = envmap->pdf_ray(ray, dr::zeros<PositionSample3f>(), active_next && !si.is_valid());
                    pdf_env *= dr::abs_dot(prev_vert.n, -ray.d);
                    pdf_pos = dr::select(si.is_valid(), pdf_pos, pdf_env);
                }
                prev_vert.pdf_rev = pdf_pos;
            }
            else {
                if (i == 0) {
                    // When in importance mode
                    // Use directional PDF for infinite lights
                    Mask is_inf       = has_flag(prev_vert.emitter->flags(),
                                                 EmitterFlags::Infinite);
                    prev_vert.pdf_rev = dr::select(is_inf, pdf_dir, pdf_pos);
                }
                else {
                    prev_vert.pdf_rev = pdf_pos;
                }
            }
            prev_vert.pdf_rev = dr::select(active_next, prev_vert.pdf_rev, 0.f);

            // Store previous vertex
//            UInt32 idx = offset + n_verts;
//            if constexpr (JIT) {
//                dr::scatter(vertices, prev_vert, idx);
//            }
            vertices[i] = dr::select(active, prev_vert, dr::zeros<Vertex3f>());

            // Update loop variables
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
            n_verts += dr::select(active, 1, 0);
            // Ensure `prev_vert` stores the last vertex
            prev_vert = dr::select(active_next, curr_vert, prev_vert);
            throughput *= bsdf_weight;
            pdf_fwd = bsdf_sample.pdf;
            active &= active_next;
        }

        Assert(!dr::any(n_verts > max_depth));

        return n_verts;
    }

#if defined(CONNECT) || defined(USE_MIS)
    auto generate_camera_subpath(const Scene *scene,
                                 const Sensor *sensor,
                                 Sampler *sampler,
                                 Vertex3f *vertices,
                                 const RayDifferential3f &ray,
                                 Mask active) const {
        auto [pdf_pos, pdf_dir] = sensor->pdf_ray(ray, dr::zeros<PositionSample3f>());
        Vertex3f vert(ray, pdf_pos);

        return random_walk(BSDFContext(), scene, sampler, m_max_depth + 1,
                           vertices, ray, vert, 1.f, pdf_dir, active);
    }
#endif

#if defined(CONNECT) || !defined(USE_MIS)
    // TODO: Delta lights ignored for now
    auto generate_light_subpath(const Scene *scene,
                                Sampler *sampler,
                                Vertex3f *vertices,
                                Float time,
                                const Wavelength &wavelengths,
                                Mask active = true) const {
        Spectrum emitter_idx_weight = 1.f;
        Float pdf_emitter = 1.f;
        EmitterPtr emitter;

        bool vcall_inline = true;
        if constexpr (dr::is_jit_v<Float>) {
            vcall_inline = jit_flag(JitFlag::VCallInline);
        }

        // Sample an emitter
        size_t emitter_count = scene->emitters().size();
        if (emitter_count > 1 || (emitter_count == 1 && !vcall_inline)) {
            auto [emitter_idx, weight, _] =
                scene->sample_emitter(sampler->next_1d(active), active);
            emitter_idx_weight = weight;
            pdf_emitter = scene->pdf_emitter(emitter_idx, active);
            emitter = dr::gather<EmitterPtr>(scene->emitters_dr(),
                                                        emitter_idx, active);
        }
        else {
            emitter = dr::gather<EmitterPtr>(scene->emitters_dr(),
                                             UInt32(0u), active);
        }
        Mask is_inf = has_flag(emitter->flags(), EmitterFlags::Infinite);

        // Sample ray from emitter
        // Note ray_weight includes radiance
        auto [ps, pdf_dir, ray, ray_weight] =
            emitter->pdf_sample_ray(time, wavelengths, sampler->next_2d(active), sampler->next_2d(active), active);
        Float pdf_pos = ps.pdf;

        Float pdf = dr::select(is_inf, pdf_dir, pdf_pos);
        Spectrum throughput = emitter_idx_weight * dr::select(pdf > 0.f, dr::rcp(pdf), 0.f);
        active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
        Float pdf_fwd = pdf * pdf_emitter;
        Vertex3f vert(ray, ps, emitter, pdf_fwd, throughput);

        throughput = emitter_idx_weight * ray_weight;
        pdf_fwd = dr::select(is_inf, pdf_pos, pdf_dir);

        return random_walk(BSDFContext(TransportMode::Importance),
                           scene, sampler, m_max_depth, vertices,
                           ray, vert, throughput, pdf_fwd, active);
    }
#endif

#ifdef CONNECT
    Spectrum connect_bdpt(const Scene *scene,
                          const Vertex3f &vert_camera,
                          const Vertex3f &vert_light,
                          Mask active = true) const {
        SurfaceInteraction3f si_camera(vert_camera);
        SurfaceInteraction3f si_light(vert_light);
        BSDFPtr bsdf_camera = vert_camera.bsdf();
        BSDFPtr bsdf_light = vert_light.bsdf();

        // Test visibility
        Ray3f ray = si_camera.spawn_ray_to(vert_light.p);
        Mask occluded = scene->ray_test(ray, active);
        active &= !occluded;

        Vector3f cam_to_light = vert_light.p - vert_camera.p;
        Float dist2 = dr::squared_norm(cam_to_light);
        cam_to_light = cam_to_light * dr::rsqrt(dist2);

        Vector3f wo_camera = si_camera.to_local(cam_to_light);
        Spectrum bsdf_val_camera =
            bsdf_camera->eval(BSDFContext(), si_camera, wo_camera, active);
        bsdf_val_camera =
            si_camera.to_world_mueller(bsdf_val_camera, -wo_camera, si_camera.wi);

        Vector3f wo_light = si_light.to_local(-cam_to_light);
        Spectrum bsdf_val_light = bsdf_light->eval(
            BSDFContext(TransportMode::Importance), si_light, wo_light, active);
        Float correction = bsdf_correction(si_light, wo_light);
        bsdf_val_light *= correction;
        bsdf_val_light = si_light.to_world_mueller(bsdf_val_light, -si_light.wi, wo_light);

        Spectrum result = vert_camera.throughput * bsdf_val_camera *
                          vert_light.throughput * bsdf_val_light;
        result *= dr::select(dr::eq(dist2, 0.f), 0.f, dr::rcp(dist2));
        return dr::select(active, result, 0.f);
    }
#endif

#ifdef CONNECT
    // TODO: Delta vertices ignored for now
    Float mis_weight(const Vertex3f *verts_camera,
                     const Vertex3f *verts_light,
                     uint32_t t, uint32_t s,
                     const Vertex3f &vert_aux,
                     Mask active_ = true) const {
#ifndef USE_MIS
        UInt32 n_techniques = 1;

        for (uint32_t i = t; i > 0; --i) {
            const Vertex3f &vert = verts_camera[i - 1];
            n_techniques += dr::select(vert.is_delta(), 0, 1);
        }

        if ( s == 1 ) {
            n_techniques++;
        }
        else {
            for (uint32_t i = s; i > 0; --i) {
                const Vertex3f &vert = verts_light[i - 1];
                n_techniques += dr::select(vert.is_delta(), 0, 1);
            }
        }

        DRJIT_MARK_USED(verts_camera);
        DRJIT_MARK_USED(verts_light);
        DRJIT_MARK_USED(vert_aux);
        return dr::select(active_, dr::rcp(Float(n_techniques)), 0.f);
#else
        uint32_t n = s + t;
        if ( n == 1 ) {
            return dr::select(active_,
                              vert_aux.pdf_fwd * dr::rcp(vert_aux.pdf_fwd + vert_aux.pdf_rev),
                              0.f);
        }

        Float sum_ri = 0.f,
              ri_camera = 1.f,
              ri_light = 1.f;
        uint32_t i_camera = t,
                 i_light = s;
        Vertex3f vert_camera = dr::zeros<Vertex3f>(),
                 vert_light  = dr::zeros<Vertex3f>();

        // When light subpath is a complete path
        if ( t == 0 ) {
            vert_light = vert_aux;
            ri_light *= vert_light.pdf_rev / vert_light.pdf_fwd;
            sum_ri += ri_light;
            i_light--;
        }

        // When camera subpath is a complete path
        else if ( s == 0 ) {
            // Compute reverse PDF for second last vertex
            vert_camera = verts_camera[t - 2];
            vert_light = vert_aux;
            Mask is_inf = dr::eq(vert_light.dist, dr::Infinity<Float>);
            SurfaceInteraction3f si_light(vert_light);
            PositionSample3f ps(si_light);
            Ray3f ray(ps.p, si_light.to_world(si_light.wi));

            auto [_, pdf_rev] = vert_light.emitter->pdf_ray(ray, ps, active_ && !is_inf);
            pdf_rev *= dr::rcp(dr::sqr(vert_light.dist)) * dr::abs_dot(vert_camera.n, -ray.d);
            vert_camera.pdf_rev = dr::select(is_inf, vert_camera.pdf_rev, pdf_rev);

            ri_camera *= vert_light.pdf_rev / vert_light.pdf_fwd;
            sum_ri += ri_camera;
            ri_camera *= vert_camera.pdf_rev / vert_camera.pdf_fwd;
            sum_ri += ri_camera;
            i_camera -= 2;
        }

        // When there is only a single light vertex
        else if ( s == 1 ) {
            vert_camera = verts_camera[t - 1];
            vert_light = vert_aux;
            Mask is_inf = has_flag(vert_light.emitter->flags(), EmitterFlags::Infinite);
            SurfaceInteraction3f si_camera(vert_camera),
                                 si_light(vert_light);
            Vector3f cam_to_light = si_light.p - si_camera.p;
            Float dist2 = dr::squared_norm(cam_to_light);
            cam_to_light *= dr::rsqrt(dist2);

            PositionSample3f ps(si_light);
            Ray3f ray(ps.p, -cam_to_light);
            auto [pdf_pos, pdf_dir] = vert_light.emitter->pdf_ray(ray, ps, active_);
            Float pdf_rev = dr::select(is_inf, pdf_pos, pdf_dir);
            pdf_rev *= dr::abs_dot(vert_camera.n, cam_to_light) *
                       dr::select(is_inf, 1.f, dr::rcp(dist2));
            vert_camera.pdf_rev = pdf_rev;

            pdf_rev = 0.f;
            pdf_rev = vert_camera.bsdf()->pdf(BSDFContext(), si_camera, si_camera.to_local(cam_to_light), active_);
            vert_light.pdf_rev = pdf_rev;

            ri_camera *= vert_camera.pdf_rev / vert_camera.pdf_fwd;
            sum_ri += ri_camera;
            ri_light *= vert_light.pdf_rev / vert_light.pdf_fwd;
            sum_ri += ri_light;
            i_camera--;
            i_light--;
        }
        // General case where t > 0 and s > 1
        else {
            vert_camera = verts_camera[t - 1];
            vert_light = verts_light[s - 1];
            BSDFPtr bsdf_camera = vert_camera.bsdf(),
                    bsdf_light  = vert_light.bsdf();
            SurfaceInteraction3f si_camera(vert_camera), si_light(vert_light);
            active_ &= vert_camera.is_connectible() && vert_light.is_connectible();

            Vector3f cam_to_light = vert_light.p - vert_camera.p;
            Float dist2           = dr::squared_norm(cam_to_light);
            cam_to_light         *= dr::rsqrt(dist2);
            vert_camera.pdf_rev   = bsdf_light->pdf(
                BSDFContext(TransportMode::Importance), si_light,
                si_light.to_local(-cam_to_light), active_);
            vert_light.pdf_rev =
                bsdf_camera->pdf(BSDFContext(), si_camera,
                                 si_camera.to_local(cam_to_light), active_);
            vert_camera.pdf_rev *=
                dr::abs_dot(vert_camera.n, cam_to_light) * dr::rcp(dist2);
            vert_light.pdf_rev *=
                dr::abs_dot(vert_light.n, -cam_to_light) * dr::rcp(dist2);

            ri_camera *= vert_camera.pdf_rev / vert_camera.pdf_fwd;
            sum_ri += ri_camera;
            ri_light *= vert_light.pdf_rev / vert_light.pdf_fwd;
            sum_ri += ri_light;
            i_camera--;
            i_light--;
        }

        for (uint32_t i = i_camera; i > 0; --i) {
            Vertex3f vert = verts_camera[i - 1];
            Float pdf_fwd = dr::select(dr::eq(vert.pdf_fwd, 0.f),
                                       1.f, vert.pdf_fwd);
            Float pdf_rev = dr::select(dr::eq(vert.pdf_rev, 0.f),
                                       1.f, vert.pdf_rev);
            ri_camera *= pdf_rev / pdf_fwd;
            sum_ri += ri_camera;
        }
        if ( i_camera > 0)
            dr::eval(sum_ri);

        for (uint32_t i = i_light; i > 0; --i) {
            Vertex3f vert = verts_light[i - 1];
            Float pdf_fwd = dr::select(dr::eq(vert.pdf_fwd, 0.f),
                                       1.f, vert.pdf_fwd);
            Float pdf_rev = dr::select(dr::eq(vert.pdf_rev, 0.f),
                                       1.f, vert.pdf_rev);
            ri_light *= pdf_rev / pdf_fwd;
            sum_ri += ri_light;
        }
        if ( i_light > 0 )
            dr::eval(sum_ri);

        return dr::select(active_, dr::rcp(1 + sum_ri), 0.f);
#endif
    }
#elif defined(USE_MIS) // #if !defined(CONNECT) && defined(USE_MIS)
    /**
     *  \brief MIS weight of one of two candidate strategies
     *  using the power heuristic
     */
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }
#endif // #ifdef CONNECT

#if defined(CONNECT) || !defined(USE_MIS)
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
                Float correction = bsdf_correction(si, local_d);
                BSDFContext ctx(TransportMode::Importance);
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
    Spectrum sample_visible_emitters(const Scene *scene,
                                 const Sensor *sensor,
                                 Sampler *sampler,
                                 ImageBlock *block,
                                 ScalarFloat sample_scale,
                                 Float time,
                                 const Wavelength &wavelengths,
                                 const Spectrum &wav_weight,
                                 Mask active = true) const {
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(sampler->next_1d(active));
        EmitterPtr emitter =
            dr::gather<EmitterPtr>(scene->emitters_dr(), emitter_idx);

        Spectrum emitter_weight = dr::zeros<Spectrum>();
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
#ifdef CONNECT
        Float pdf_fwd           = scene->pdf_emitter(emitter_idx, active);
        Normal3f n = 0.f;
#endif

        // Sample direction for infinite emitters
        Mask is_infinite = has_flag(emitter->flags(), EmitterFlags::Infinite);
        if (dr::any_or<true>(is_infinite)) {
            Interaction3f ref_it(0.f, time, wavelengths,
                                 sensor->world_transform().translation());
            auto [ds, dir_weight] = emitter->sample_direction(
                ref_it, sampler->next_2d(is_infinite && active), is_infinite && active);

#ifdef CONNECT
            pdf_fwd *= dr::select(is_infinite, ds.pdf, 1.f);
#endif

            // Convert solid angle measure to area measure
            emitter_weight[is_infinite] =
                dr::select(ds.pdf > 0.f, dr::rcp(ds.pdf), 0.f) *
                dr::sqr(ds.dist);
            si[is_infinite] = SurfaceInteraction3f(ds, wavelengths);
        }

        // Sample position for finite emitters
        if (dr::any_or<true>(!is_infinite)) {
            auto [ps, pos_weight] =
            emitter->sample_position(time, sampler->next_2d(!is_infinite && active), !is_infinite && active);

#ifdef CONNECT
            pdf_fwd *= dr::select(!is_infinite, ps.pdf , 1.f);
            n = ps.n;
#endif

            emitter_weight[!is_infinite] = pos_weight;
            si[!is_infinite] = SurfaceInteraction3f(ps, wavelengths);
        }

        // Sample direction toward sensor
        Point2f aperture_sample;
        if (sensor->needs_aperture_sample()) {
            aperture_sample = sampler->next_2d(active);
        }
        auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, aperture_sample, active);
        active &= sensor_ds.pdf > 0.f;
        si.wi = sensor_ds.d;
        si.shape = emitter->shape();

#ifdef CONNECT
        Vertex3f vert = dr::zeros<Vertex3f>();
        vert.pdf_fwd = pdf_fwd;
        Ray3f ray(sensor_ds.p, -sensor_ds.d);
        Float pdf_rev = 0.f;
        std::tie(_, pdf_rev) = sensor->pdf_ray(ray, sensor_ds, active);
        pdf_rev *= dr::select(is_infinite, 1.f, dr::rcp(dr::sqr(sensor_ds.dist)) * dr::abs_dot(n, sensor_ds.d));
        vert.pdf_rev = pdf_rev;
        Float weight_mis = mis_weight(nullptr, nullptr, 0, 1, vert, active);
#else
        Float weight_mis = 1.f;
#endif

        Spectrum radiance = emitter->eval(si, active);
        Spectrum weight = emitter_idx_weight * emitter_weight * sensor_weight * radiance * wav_weight * weight_mis;
        weight = dr::select(active, weight, 0.f);

        return connect_sensor(scene, si, sensor_ds, nullptr, weight, block, sample_scale, active);
    }
#endif // #if !defined(CONNECT) && !defined(USE_MIS)

    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    /**
     * \brief Correction factor for adjoint BSDF for shading normals
     * */
    Float bsdf_correction(const SurfaceInteraction3f &si,
                          const Vector3f &wo) const {
        Float wi_dot_geo_n = dr::dot(si.n, si.to_world(si.wi)),
              wo_dot_geo_n = dr::dot(si.n, si.to_world(wo));
        // Prevent light leaks due to shading normals
        Mask valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                     (wo_dot_geo_n * Frame3f::cos_theta(wo) > 0.f);
        // [Veach, p. 155]
        Float correction = dr::select(valid,
                                      dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                              (Frame3f::cos_theta(wo) * wi_dot_geo_n)),
                                      0.f);
        return correction;
    }

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

#undef USE_MIS
#undef CONNECT