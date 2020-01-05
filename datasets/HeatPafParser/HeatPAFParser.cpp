/*
 * Adapted from caffe2: github.com/caffe2/caffe2
 */
#include "bilinear.h"
#include "mobula_op.h"
#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>
#include <tuple>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)
#include <cstdio>

class HeatPeak(){
    public:
    size_t x;
    size_t y;
    float score;
    HeatPeak(size_t x, size_t y, float score){
        this->x = x;
        this->y = y;
        this-score = score;
    };
};
class ConnectionCandidate{
public:
    size_t nA;
    size_t nB;
    float score0;
    float score1;
    ConnectionCandidate(size_t nA, size_t nB, float score0, float score1){
        this->nA = nA;
        this->nB = nB;
        this->score0 = score0;
        this->score1 = score1;
    }
}

void f_compare_connection_candidates(const ConnectionCandidate &c0, const ConnectionCandidate &c1){
    return c0.score0 > c1.score0;
}


template <typename T, typename T_index>
MOBULA_KERNEL heat_paf_parser_kernel(const T* p_heat, const T* p_paf, const T_index *limbs, const T_index number_of_parts,
                                     const T_index number_of_limbs, const T_index image_width,
                                     const T_index image_height ) {
    const T threshold1 = .1;
    const T threshold2 = 0.05;
    const T mid_num = 10;
    // find sub-max positions
    std::vector<std::vector<HeatPeak>> heatPeaks;
    for(auto i=0; i< number_of_parts; i++){
        // Escape the first and the last row/column.
        heatPeaks.push_back(std::vector<HeatPeak>);
        const size_t channel_offset = i * (image_height * image_width);
        for(auto m=1; m < image_height; m ++){
            for(auto n=1; n < image_width; n++){
                auto currentValue = p_heat[ channel_offset + m * image_height + n * image_width];
                if(currentValue > threshold1){
                    auto upValue = p_heat[ channel_offset + (m-1) * image_height + n * image_width];
                    auto bottomValue = p_heat[ channel_offset + (m+1) * image_height + n * image_width];
                    auto rightValue = p_heat[ channel_offset + m * image_height + (n + 1) * image_width];
                    auto leftValue = p_heat[ channel_offset + m * image_height + (n - 1) * image_width];
                    if(currentValue >= upValue && currentValue >= bottomValue
                        && currentValue >= leftValue && currentValue >= rightValue){
                        heatPeaks[i].emplace_back(n, m, currentValue);
                    }
                }

            }
        }
    }
    std::vector<std::vector<std::tuple<T, T, T, size_t, size_t>>> connection_all;
    for(auto i=0; i< number_of_limbs; i++){
        size_t indexA = limbs[i * 2 + 0];
        size_t indexB = limbs[i * 2 + 1];
        T *p_score_mid_x = p_paf + (i * 2 + 0) * image_height * image_width;
        T *p_score_mid_y = p_paf + (i * 2 + 1) * image_height * image_width;
        auto connection_candidates = std::vector<ConnectionCandidate>();
        for(auto nA = 0; nA < heatPeaks[indexA].size(); nA ++){
            for(auto nB=0; nB < heatPeaks[indexB].size(); nB ++){
                auto& p0 = heatPeaks[indexA][nA];
                auto& p1 = heatPeaks[indexB][nB];
                auto vec_x = p1.x - p0.x;
                auto vec_y = p1.y - p0.y;
                auto norm = std::sqrt(vec_x * vec_x + vec_y *vec_y);
                if(norm < 0.1){
                    norm += 1;
                    std::puts("norm is too small, adding one to avoid NAN.");
                }
                vec_x /= norm;
                vec_y /= norm;
                T score_with_dist_prior = 0.0;
                int count_satisfy_thre2 = 0;
                for(auto t=0; t< mid_num; t++){
                    size_t integral_x = static_cast<size_t>(std::round(p0.x + (p1.x - p0.x) / (mid_num-1) * t));
                    size_t integral_y = static_cast<size_t>(std::round(p0.y + (p1.y - p0.y) / (mid_num-1) * t));
                    auto paf_predict_x = p_score_mid_x[integral_y * image_width + integral_x];
                    auto paf_predict_y = p_score_mid_y[integral_y * image_width + integral_x];
                    auto score = vec_x * paf_predict_x + vec_y * paf_predict_y;
                    score_with_dist_prior += score;
                    if(score > threshold2){
                        count_satisfy_thre2 += 1;
                    }
                }
                score_with_dist_prior /= mid_num;
                score_with_dist_prior += std::min(.5 * image_height / norm, 0);
                if(count_satisfy_thre2 > 0.8 * mid_num && score_with_dist_prior >0){
                    connection_candidates.emplace_back(nA, nB, score_with_dist_prior, score_with_dist_prior + p0.score + p1.score);
                }
            }
        } // End calculate scores of all possible connections;
        // Remove redundant connections
        // Sort all candidates according to score0
        std::vector<std::tuple<T, T, T, size_t, size_t>> connections;
        std::sort(connection_candidates.begin(); connection_candidates.end(); f_compare_connection_candidates);
        for(auto nc=0; nc < connection_candidates.size(); nc ++){
            auto &nA = connection_candidates[nc].nA;
            auto &nB = connection_candidates[nc].nB;
            bool exist_flag = false;
            for(const auto &c:connections){
                if(nA == std::get<3>(c) || nB == std::get<4>(c)){
                    exist_flag = true;
                    break;
                }
            }
            // if one parts connected to more than one other parts, only keep the one with the largest score.
            if(!exist_flag){
                connections.emplace_back(heatPeaks[indexA][nA].score, heatPeaks[indexA][nB].score, connection_candidates[nc].score);
            }
            if(connections.size() > std::min(heatPeaks[indexA].size(), heatPeaks[indexAB].size())){
                break;
            }
        }
        connection_all.push_back(connections);
    }

    // parts connected with each other should be connected.
    for(auto k=0; k< number_of_limbs; k++){

    }
} // paf_gen_kernel


}  // namespace mobula
