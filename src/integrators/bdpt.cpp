#include <mitsuba/render/integrator.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BDPTIntegrator : public AdjointIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(AdjointIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Film, Sampler, ImageBlock, Emitter, EmitterPtr,
                    Sensor, SensorPtr, BSDF, BSDFPtr)

    BDPTIntegrator(const Properties &props) : Base(props) {
    }

    void sample(const Scene *scene, const Sensor *sensor, Sampler *sampler,
                ImageBlock *block, ScalarFloat sample_scale) const override {
        auto [camera_verts, n_camera_verts] = generate_camera_subpath();

        auto [light_verts, n_light_verts] = generate_light_subpath();

        Spectrum result(0.f);
        UInt32 t = 1;
        while (loop(active_t)) {
            UInt32 s = 0;
            while (loop(active_s)) {
                // Check for invalid combination

                // Connect
                result += connect_bdpt();
            }
        }

        // Splat
    }


    UInt32 generate_camera_subpath() const;

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

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(BDPTIntegrator, AdjointIntegrator);
MI_EXPORT_PLUGIN(BDPTIntegrator, "Bidirectional Path Tracer integrator")
NAMESPACE_END(mitsuba)