#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>

namespace asymptotic_tetra {
namespace math {

constexpr size_t DEFAULT_BUF_SIZE = 1 << 10;

/**
 * GeneratorType enumerates the available random number generators.
 */
enum class GeneratorType {
    Xorshift,
    Golang,
    Tausworthe,
    Default = Tausworthe
};

/**
 * GeneratorBackend is the abstract interface for random number generators.
 */
class GeneratorBackend {
public:
    virtual ~GeneratorBackend() = default;
    virtual void init(uint64_t seed) = 0;
    virtual double next() = 0;
    virtual void next_sequence(std::vector<double>& target) = 0;
};

/**
 * XorshiftGenerator implements a simple xorshift PRNG.
 */
class XorshiftGenerator : public GeneratorBackend {
public:
    void init(uint64_t seed) override;
    double next() override;
    void next_sequence(std::vector<double>& target) override;

private:
    uint64_t state = 0;
};

/**
 * GolangGenerator wraps the standard library's mt19937_64 (similar to Go's default).
 */
class GolangGenerator : public GeneratorBackend {
public:
    void init(uint64_t seed) override;
    double next() override;
    void next_sequence(std::vector<double>& target) override;

private:
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist{0.0, 1.0};
};

/**
 * TauswortheGenerator implements a Tausworthe-style lagged Fibonacci generator.
 * This is the default and recommended generator for Monte Carlo sampling.
 */
class TauswortheGenerator : public GeneratorBackend {
public:
    void init(uint64_t seed) override;
    double next() override;
    void next_sequence(std::vector<double>& target) override;

private:
    static constexpr int DIGITS_RANDOMIZED = 15;
    static constexpr size_t SEQ_LEN = 9689;
    static constexpr size_t FIRST_OFFSET = 2444;
    static constexpr size_t SECOND_OFFSET = 4187;

    std::vector<double> seq;
    size_t leader = 0;
    size_t first_follower = 0;
    size_t second_follower = 0;
};

/**
 * Generator is the main interface for random number generation.
 * Supports uniform sampling and gaussian sampling.
 */
class Generator {
public:
    Generator(GeneratorType type, uint64_t seed);
    
    static Generator new_time_seed(GeneratorType type = GeneratorType::Default);

    /**
     * Generate a uniform random integer in [low, high).
     */
    int uniform_int(int low, int high);

    /**
     * Generate a uniform random double in [low, high).
     */
    double uniform(double low, double high);

    /**
     * Fill a vector with uniform random doubles in [low, high).
     */
    void uniform_at(double low, double high, std::vector<double>& target);

    /**
     * Generate a standard normal random variable using Box-Muller.
     */
    double gaussian();

private:
    std::unique_ptr<GeneratorBackend> backend;
    bool saved_gaussian = false;
    double next_gaussian_dx = 0.0;
};

} // namespace math
} // namespace asymptotic_tetra
