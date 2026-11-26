// Transfer/dispatch and physical stage invariants, independently of fonts or
// grain. The full seeded film/date/DX suite lives in overscan_checks.py.
#include "gpu/gpu.hpp"
#include "gpu_vulkan_read.hpp"
#include "core/json.hpp"
#include "spektrafilm/spk_engine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void check(bool ok,const std::string& error){if(!ok)throw std::runtime_error(error);}
using P=std::array<float,261>;
std::vector<float> run(spk::gpu::Gpu* gpu,const char* kernel,const std::vector<float>& in,const std::vector<float>& p,
                       std::vector<float> initial,uint32_t count){
    std::string e;initial.insert(initial.end(),51,-12345.f);
    auto source=gpu->upload(in.data(),in.size()*4,e),params=gpu->upload(p.data(),p.size()*4,e),dest=gpu->upload(initial.data(),initial.size()*4,e);
    check(source&&params&&dest,e);
    const bool ok=gpu->dispatch(kernel,{spk::gpu::Arg::buf(source),spk::gpu::Arg::buf(params),spk::gpu::Arg::inline_bytes(&count,1),spk::gpu::Arg::buf(dest)},count+31,e)&&gpu->flush(e);
    check(ok,e);std::vector<float> out;check(read_test_buffer(gpu,dest,out,e),e);
    for(size_t i=initial.size()-51;i<initial.size();++i)check(out[i]==-12345.f,"overscan dispatch wrote past output");
    out.resize(initial.size()-51);return out;
}
std::vector<float> run(spk::gpu::Gpu* gpu,const char* kernel,const std::vector<float>& in,const P& p){return run(gpu,kernel,in,{p.begin(),p.end()},std::vector<float>(in.size()),uint32_t(in.size()/3));}

void check_picture_edr(const char* resources) {
    std::unique_ptr<spk_engine, decltype(&spk_engine_destroy)> engine(
        spk_engine_create(resources, nullptr), spk_engine_destroy);
    check(bool(engine), spk_last_error(nullptr));
    for (const bool pair : {false, true}) {
        const unsigned width = pair ? 370 : 360;
        const unsigned height = 240;
        std::vector<float> input(width * height * 3, 0.004f);
        std::fill_n(input.begin(), input.size() / 2, 8.5f);
        const spk_image image{input.data(), width, height, 3};
        spk::Json params;
        std::string error;
        check(spk::Json::parse(R"({"auto_exposure":false,"grain_active":false,"glare_active":false,
             "input_cctf_decoding":false,"overscan_active":true,"overscan_holes":"white",
             "overscan_edge_text":"KODAK PORTRA 400","overscan_frame_seed":13})", params, error), error);
        params.set("overscan_format", spk::Json(std::string(pair ? "135_half" : "135")));
        params.set("overscan_pair", spk::Json(pair));
        std::unique_ptr<spk_session, decltype(&spk_session_release)> session(
            spk_open(engine.get(), &image, params.dump().c_str(), nullptr), spk_session_release);
        check(bool(session), spk_last_error(engine.get()));
        spk_result off{}, on{};
        check(spk_render(session.get(), "full", &off) == SPK_OK, spk_last_error(engine.get()));
        char* raw_geometry = nullptr;
        check(spk_overscan_geometry(session.get(), &raw_geometry) == SPK_OK && raw_geometry,
              "EDR test has no film geometry");
        std::unique_ptr<char, decltype(&spk_string_free)> geometry_json(raw_geometry, spk_string_free);
        spk::Json geometry;
        check(spk::Json::parse(geometry_json.get(), geometry, error), error);
        check(spk_set_params(session.get(), R"({"extended_dynamic_range":true})", nullptr) == SPK_OK,
              spk_last_error(engine.get()));
        check(spk_reprint(session.get(), "full", &on) == SPK_OK, spk_last_error(engine.get()));
        check(on.negative_was_cached && off.rgba16 && on.rgba16 && off.width == on.width && off.height == on.height,
              "EDR must reprint the same cached film canvas");
        const auto same_pixel = [&](unsigned x, unsigned y) {
            const auto* a = off.rgba16 + (size_t(y) * off.row_stride_px + x) * 4;
            const auto* b = on.rgba16 + (size_t(y) * on.row_stride_px + x) * 4;
            return std::equal(a, a + 4, b);
        };
        const auto& gates = geometry.at("gates").items();
        check(gates.size() == (pair ? 2u : 1u), "unexpected number of picture gates");
        std::vector<std::array<double, 4>> boxes;
        for (const auto& gate : gates) {
            const auto& corners = gate.items();
            check(corners.size() == 8, "invalid picture gate corners");
            double left = on.width, right = 0, top = on.height, bottom = 0;
            for (size_t i = 0; i < corners.size(); i += 2) {
                const double x = corners[i].as_double() * on.width;
                const double y = corners[i + 1].as_double() * on.height;
                left = std::min(left, x); right = std::max(right, x);
                top = std::min(top, y); bottom = std::max(bottom, y);
            }
            boxes.push_back({left, right, top, bottom});
            check(!same_pixel(unsigned((left + right) / 2), unsigned((top * 3 + bottom) / 4)),
                  "EDR did not change the print inside a picture gate");
        }
        // Beyond 1 mm the optical blur has settled to zero, including the perforations.
        const double margin = 1.0 / geometry.at("mm_per_px").as_double();
        size_t outside = 0, changed = 0;
        for (unsigned y = 0; y < on.height; ++y) {
            for (unsigned x = 0; x < on.width; ++x) {
                if (std::any_of(boxes.begin(), boxes.end(), [&](const auto& box) {
                    return x >= box[0] - margin && x <= box[1] + margin &&
                           y >= box[2] - margin && y <= box[3] + margin;
                })) continue;
                ++outside;
                changed += !same_pixel(x, y);
            }
        }
        spk_result_free(&on);
        spk_result_free(&off);
        check(outside > 1000 && changed == 0,
              "EDR changed " + std::to_string(changed) + " pixels outside the picture gates");
    }
}
}
int main(int argc,char** argv){try{
    check(argc==2,"usage: spk_vulkan_overscan resources_dir");std::string e;
    std::unique_ptr<spk::gpu::Gpu> gpu(spk::gpu::Gpu::create_vulkan(std::string(argv[1])+"/vulkan",e));check(bool(gpu),e);
    constexpr int w=17,h=13;std::vector<float> input(w*h*3);for(size_t i=0;i<input.size();++i)input[i]=float(1u<<(i%3))*.2f;
    P p{};p[0]=w;p[1]=h;p[2]=.1f;p[3]=w;p[4]=h;p[5]=p[8]=1;p[16]=w*p[2];p[17]=h*p[2];p[44]=h*p[2];p[199]=2;
    const auto canvas=run(gpu.get(),"spk_overscan_canvas",input,p);
    check(canvas==input,"unmodified gate did not preserve uniform raw exposure");
    check(run(gpu.get(),"spk_overscan_film_present",input,p)==input,"unperforated no-carrier film changed density");
    const auto identity_light=run(gpu.get(),"spk_overscan_light",input,p);
    // GLSL mix lowers to arithmetic; the identity endpoint may round by 1 ulp.
    for(size_t i=0;i<input.size();++i)check(std::abs(identity_light[i]-input[i])<2e-7f,
        "unperforated no-carrier scan changed RGB");
    // Shift film relative to the canvas, exercising full film, half coverage
    // and no film. Density removal precedes print; scan carrier is later.
    p[199]=0;p[10]=-.4f;p[44]=.8f;p[90]=.05f;p[91]=.1f;p[92]=.15f;
    const auto film=run(gpu.get(),"spk_overscan_film_present",input,p),light=run(gpu.get(),"spk_overscan_light",input,p);
    for(int y=0;y<h;++y){const float t=(float(y)+.5f)*p[2]+p[10];const float a=std::clamp(std::min(t,p[44]-t)/p[2]+.5f,0.f,1.f);
        for(int x=0;x<w;++x)for(int c=0;c<3;++c){const size_t i=(size_t(y)*w+x)*3+c;
            check(std::abs(film[i]-(a*input[i]+(1-a)*p[90+c]))<2e-6f,"film coverage changed physical density blend");
            check(std::abs(light[i]-a*input[i])<2e-6f,"scan carrier coverage is wrong");
        }
    }
    // The source mask is clipped using its own offset; negative coordinates
    // must not underflow, and non-positive samples leave the canvas alone.
    std::vector<float> mask(5*7);for(size_t i=0;i<mask.size();++i)mask[i]=float(int(i%9)-2)/8;
    const std::vector<float> m{-2,9,5,7,float(w),float(h),.2f,.4f,.6f};
    const auto added=run(gpu.get(),"spk_overscan_add_mask",mask,m,input,uint32_t(mask.size()));auto want=input;
    for(int y=0;y<7;++y)for(int x=0;x<5;++x){const int xx=x-2,yy=y+9;if(xx<0||yy<0||xx>=w||yy>=h)continue;
        const float a=mask[y*5+x];if(a<=0)continue;for(int c=0;c<3;++c)want[(yy*w+xx)*3+c]+=a*m[6+c];}
    for(size_t i=0;i<want.size();++i)check(std::abs(added[i]-want[i])<2e-7f,"mask exposure/offset clipping differs from CPU reference");
    gpu.reset();
    check_picture_edr(argv[1]);
    std::puts("Overscan passed exposure, film coverage, dispatch guards and picture-only EDR");return 0;
}catch(const std::exception& ex){std::fprintf(stderr,"%s\n",ex.what());return 1;}}
