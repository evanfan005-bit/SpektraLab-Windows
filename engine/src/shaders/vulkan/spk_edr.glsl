// Calibrated EDR log2-Y LUT and identity toe/shoulder joins, nodes.metal.
layout(set = 0, binding = 1, std430) readonly buffer EdrParameters { float value[]; } parameters;
layout(set = 0, binding = 2, std430) readonly buffer Lut { float value[]; } lut_data;

float edr_scale(float r, float g, float b) {
    float y = max(parameters.value[0] * r + parameters.value[1] * g + parameters.value[2] * b, 0.0);
    float toe = parameters.value[5], shoulder = parameters.value[6];
    float mapped = y;
    if (y > 0.0 && (y < toe || y > shoulder)) {
        uint count = max(uint(parameters.value[7]), 2u);
        float u = clamp((log2(y) - parameters.value[3]) * parameters.value[4], 0.0, 1.0) * float(count - 1u);
        uint i0 = min(uint(floor(u)), count - 2u);
        float t = u - float(i0);
        mapped = exp2(mix(lut_data.value[i0], lut_data.value[i0 + 1u], t));
    }
    return mapped / (y + 1e-10);
}
