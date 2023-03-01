#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(Endpoint) {
    MI_PY_IMPORT_TYPES()
    MI_PY_CLASS(Endpoint, Object)
        .def("sample_ray", &Endpoint::sample_ray,
            "time"_a, "sample1"_a, "sample2"_a, "sample3"_a, "active"_a = true,
            D(Endpoint, sample_ray))
        .def("sample_ray_1", &Endpoint::sample_ray_1,
             "time"_a, "wavelengths"_a, "sample2"_a, "sample3"_a, "active"_a = true,
             D(Endpoint, sample_ray_1))
        .def("sample_ray_3", &Endpoint::sample_ray_3,
             "time"_a, "sample1"_a, "sample2"_a, "ps"_a, "active"_a = true,
             D(Endpoint, sample_ray_3))
        .def("sample_ray_13", &Endpoint::sample_ray_13,
             "time"_a, "wavelengths"_a, "sample2"_a, "ps"_a, "active"_a = true,
             D(Endpoint, sample_ray_13))
        .def("pdf_ray", &Endpoint::pdf_ray,
             "ray"_a, "ps"_a, "active"_a = true,
             D(Endpoint, pdf_ray))
        .def("pdf_sample_ray", &Endpoint::pdf_sample_ray,
             "time"_a, "wavelengths"_a, "sample2"_a, "sample3"_a, "active"_a = true,
             D(Endpoint, pdf_sample_ray))
        .def("sample_direction", &Endpoint::sample_direction,
             "it"_a, "sample"_a, "active"_a = true, D(Endpoint, sample_direction))
        .def("pdf_direction", &Endpoint::pdf_direction,
             "it"_a, "ds"_a, "active"_a = true, D(Endpoint, pdf_direction))
        .def("eval_direction", &Endpoint::eval_direction,
             "it"_a, "ds"_a, "active"_a = true, D(Endpoint, eval_direction))
        .def("sample_position", &Endpoint::sample_position,
             "ref"_a, "ds"_a, "active"_a = true, D(Endpoint, sample_position))
        .def("pdf_position", &Endpoint::pdf_position,
             "ps"_a, "active"_a = true, D(Endpoint, pdf_position))
        .def("eval", &Endpoint::eval,
             "si"_a, "active"_a = true, D(Endpoint, eval))
        .def("sample_wavelengths", &Endpoint::sample_wavelengths,
             "si"_a, "sample"_a, "active"_a = true, D(Endpoint, sample_wavelengths))
        .def("pdf_wavelengths", &Endpoint::pdf_wavelengths,
             "wavelengths"_a, "active"_a = true, D(Endpoint, pdf_wavelengths))
        .def_method(Endpoint, world_transform)
        .def_method(Endpoint, needs_sample_2)
        .def_method(Endpoint, needs_sample_3)
        .def("shape",  py::overload_cast<>(&Endpoint::shape, py::const_),  D(Endpoint, shape))
        .def("medium", py::overload_cast<>(&Endpoint::medium, py::const_), D(Endpoint, medium))
        .def_method(Endpoint, set_shape, "shape"_a)
        .def_method(Endpoint, set_medium, "medium"_a)
        .def_method(Endpoint, set_scene, "scene"_a)
        .def_method(Endpoint, bbox);
}
