#include "simulation.hpp"

float calc_next_c(float C, float C_left, float C_right, float C_up, float C_down) {
    float d2x = (C_right - 2.0f * C + C_left) / (DX * DX);
    float d2y = (C_down  - 2.0f * C + C_up)   / (DY * DY);
    float diffusion = DIFF_D * (d2x + d2y);

    float adv_x = UX * (C - C_left) / DX;
    float adv_y = UY * (C - C_up)   / DY;
    float advection = adv_x + adv_y;

    float decay = (LAMBDA_DECAY + K_DEP) * C;

    float dC_dt = diffusion - advection - decay;
    float next_C = C + DT * dC_dt;

    if (next_C < 0.0f || !std::isfinite(next_C)) {
        return 0.0f;
    }
    return next_C;
}