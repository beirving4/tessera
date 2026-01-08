#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace asymptotic_tetra {
namespace math {

/**
 * Maximum dimensions for Sobol sequences.
 */
constexpr uint32_t SOBOL_MAX_DIM = 6;
constexpr uint32_t SOBOL_MAX_BIT = 30;
constexpr uint32_t SOBOL_MAX_SEQ_NUM = 1U << SOBOL_MAX_BIT;
constexpr double SOBOL_FAC = 1.0 / SOBOL_MAX_SEQ_NUM;

/**
 * Sobol quasi-random sequence generator.
 * 
 * Sobol sequences are low-discrepancy sequences that provide better
 * coverage of the sample space than pseudo-random numbers, making them
 * ideal for Monte Carlo integration.
 * 
 * Based on Press et al. 2007 (Numerical Recipes).
 * 
 * Example:
 *     SobolSequence seq;
 *     auto point = seq.next(3);  // Get 3D point
 */
class SobolSequence {
public:
    SobolSequence() { init(); }
    
    /**
     * Initialize/reset the sequence.
     */
    void init() {
        for (auto& x : ix_) x = 0;
        
        if (is_init_) {
            seq_num_ = 0;
            return;
        }
        
        seq_num_ = 0;
        
        // Initialize from static data
        iv_ = init_iv_;
        ip_ = init_ip_;
        mdeg_ = init_mdeg_;
        
        for (uint32_t k = 0; k < SOBOL_MAX_DIM; ++k) {
            for (uint32_t j = 0; j < mdeg_[k]; ++j) {
                iv_[SOBOL_MAX_DIM * j + k] <<= (SOBOL_MAX_BIT - j - 1);
            }
            
            uint32_t deg = mdeg_[k];
            for (uint32_t j = deg; j < SOBOL_MAX_BIT; ++j) {
                uint32_t ipp = ip_[k];
                uint32_t i = iv_[SOBOL_MAX_DIM * (j - deg) + k];
                i ^= (i >> deg);
                
                for (uint32_t l = deg - 1; l >= 1; --l) {
                    if ((ipp & 1) == 1) {
                        i ^= iv_[SOBOL_MAX_DIM * (j - l) + k];
                    }
                    ipp >>= 1;
                }
                
                iv_[SOBOL_MAX_DIM * j + k] = i;
            }
        }
        
        is_init_ = true;
    }
    
    /**
     * Reset the sequence to the beginning.
     */
    void reset() {
        seq_num_ = 0;
        for (auto& x : ix_) x = 0;
    }
    
    /**
     * Get the next point in the sequence.
     * 
     * @param dim Number of dimensions (1 to SOBOL_MAX_DIM)
     * @return Vector of quasi-random values in [0, 1)
     */
    std::vector<double> next(int dim) {
        std::vector<double> result(dim);
        next_at(result);
        return result;
    }
    
    /**
     * Get the next point in the sequence, storing in-place.
     * 
     * @param target Output vector (size determines dimensionality)
     */
    void next_at(std::vector<double>& target) {
        uint32_t dim = static_cast<uint32_t>(target.size());
        
        if (dim > SOBOL_MAX_DIM) {
            throw std::runtime_error("Dimension exceeds SOBOL_MAX_DIM");
        }
        if (seq_num_ >= SOBOL_MAX_SEQ_NUM) {
            throw std::runtime_error("Exceeded maximum sequence number");
        }
        
        seq_num_++;
        
        // Find the rightmost zero bit
        uint32_t zero_idx = 0;
        for (zero_idx = 0; zero_idx < SOBOL_MAX_BIT; ++zero_idx) {
            if ((seq_num_ & (1U << zero_idx)) == 0) break;
        }
        
        uint32_t im = zero_idx * SOBOL_MAX_DIM;
        for (uint32_t k = 0; k < dim; ++k) {
            ix_[k] ^= iv_[im + k];
            target[k] = static_cast<double>(ix_[k]) * SOBOL_FAC;
        }
    }
    
    /**
     * Get the next point using a fixed-size array.
     */
    template<size_t N>
    void next_at(std::array<double, N>& target) {
        static_assert(N <= SOBOL_MAX_DIM, "Dimension exceeds SOBOL_MAX_DIM");
        
        if (seq_num_ >= SOBOL_MAX_SEQ_NUM) {
            throw std::runtime_error("Exceeded maximum sequence number");
        }
        
        seq_num_++;
        
        uint32_t zero_idx = 0;
        for (zero_idx = 0; zero_idx < SOBOL_MAX_BIT; ++zero_idx) {
            if ((seq_num_ & (1U << zero_idx)) == 0) break;
        }
        
        uint32_t im = zero_idx * SOBOL_MAX_DIM;
        for (uint32_t k = 0; k < N; ++k) {
            ix_[k] ^= iv_[im + k];
            target[k] = static_cast<double>(ix_[k]) * SOBOL_FAC;
        }
    }
    
    /**
     * Get current sequence number.
     */
    uint32_t sequence_number() const { return seq_num_; }
    
    /**
     * Get maximum sequence length.
     */
    static constexpr uint32_t max_sequence() { return SOBOL_MAX_SEQ_NUM; }
    
    /**
     * Get maximum supported dimensions.
     */
    static constexpr uint32_t max_dim() { return SOBOL_MAX_DIM; }

private:
    uint32_t seq_num_ = 0;
    std::array<uint32_t, SOBOL_MAX_DIM> ix_{};
    std::array<uint32_t, SOBOL_MAX_DIM> mdeg_{};
    std::array<uint32_t, SOBOL_MAX_DIM> ip_{};
    std::array<uint32_t, SOBOL_MAX_BIT * SOBOL_MAX_DIM> iv_{};
    bool is_init_ = false;
    
    // Static initialization data from Numerical Recipes
    static constexpr std::array<uint32_t, SOBOL_MAX_DIM> init_mdeg_ = {1, 2, 3, 3, 4, 4};
    static constexpr std::array<uint32_t, SOBOL_MAX_DIM> init_ip_ = {0, 1, 1, 2, 1, 4};
    static constexpr std::array<uint32_t, SOBOL_MAX_BIT * SOBOL_MAX_DIM> init_iv_ = {
        1, 1, 1, 1, 1, 1,
        3, 1, 3, 3, 1, 1,
        5, 7, 7, 3, 3, 5,
        15, 11, 5, 15, 13, 9,
        // Rest initialized to 0
    };
};

} // namespace math
} // namespace asymptotic_tetra
