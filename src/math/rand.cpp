#include "math/rand.h"
#include <stdexcept>

namespace tessera {
namespace math {

// XorshiftGenerator implementation
void XorshiftGenerator::init(uint64_t seed) {
    state = seed;
    if (state == 0) state = 1;  // Avoid zero state
}

double XorshiftGenerator::next() {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return static_cast<double>(x) / static_cast<double>(UINT64_MAX);
}

void XorshiftGenerator::next_sequence(std::vector<double>& target) {
    for (auto& val : target) {
        val = next();
    }
}

// GolangGenerator implementation
void GolangGenerator::init(uint64_t seed) {
    rng.seed(seed);
}

double GolangGenerator::next() {
    return dist(rng);
}

void GolangGenerator::next_sequence(std::vector<double>& target) {
    for (auto& val : target) {
        val = dist(rng);
    }
}

// TauswortheGenerator implementation
void TauswortheGenerator::init(uint64_t seed) {
    seq.resize(SEQ_LEN);
    
    // Use Golang generator to initialize sequence
    GolangGenerator digit_gen;
    digit_gen.init(seed);

    double f = 1.0;
    for (int digit = 0; digit < DIGITS_RANDOMIZED; ++digit) {
        for (size_t i = 0; i < seq.size(); ++i) {
            seq[i] += digit_gen.next() * f;
        }
        f /= 2.0;
    }

    for (auto& val : seq) {
        if (val >= 1.0) val -= 1.0;
    }

    leader = 0;
    first_follower = FIRST_OFFSET;
    second_follower = SECOND_OFFSET;
}

double TauswortheGenerator::next() {
    double next_val = seq[first_follower] - seq[second_follower];
    if (next_val < 0) next_val += 1.0;
    seq[leader] = next_val;

    if (leader == 0) leader = seq.size();
    if (first_follower == 0) first_follower = seq.size();
    if (second_follower == 0) second_follower = seq.size();

    --leader;
    --first_follower;
    --second_follower;

    return next_val;
}

void TauswortheGenerator::next_sequence(std::vector<double>& target) {
    for (auto& val : target) {
        double next_val = seq[first_follower] - seq[second_follower];
        if (next_val < 0) next_val += 1.0;
        seq[leader] = next_val;

        if (leader == 0) leader = seq.size();
        if (first_follower == 0) first_follower = seq.size();
        if (second_follower == 0) second_follower = seq.size();

        --leader;
        --first_follower;
        --second_follower;

        val = next_val;
    }
}

// Generator implementation
Generator::Generator(GeneratorType type, uint64_t seed) {
    switch (type) {
        case GeneratorType::Xorshift:
            backend = std::make_unique<XorshiftGenerator>();
            break;
        case GeneratorType::Golang:
            backend = std::make_unique<GolangGenerator>();
            break;
        case GeneratorType::Tausworthe:
            backend = std::make_unique<TauswortheGenerator>();
            break;
        default:
            throw std::runtime_error("Unrecognized GeneratorType");
    }
    backend->init(seed);
}

Generator Generator::new_time_seed(GeneratorType type) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    uint64_t seed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()
    );
    return Generator(type, seed);
}

int Generator::uniform_int(int low, int high) {
    double f = backend->next();
    return static_cast<int>(std::floor(static_cast<double>(high - low) * f + static_cast<double>(low)));
}

double Generator::uniform(double low, double high) {
    if (low == 0.0 && high == 1.0) return backend->next();
    return backend->next() * (high - low) + low;
}

void Generator::uniform_at(double low, double high, std::vector<double>& target) {
    backend->next_sequence(target);
    if (low == 0.0 && high == 1.0) return;
    for (auto& val : target) {
        val = val * (high - low) + low;
    }
}

double Generator::gaussian() {
    if (saved_gaussian) {
        saved_gaussian = false;
        return next_gaussian_dx;
    }
    
    double u1 = backend->next();
    double u2 = backend->next();
    
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    
    next_gaussian_dx = r * std::sin(theta);
    saved_gaussian = true;
    
    return r * std::cos(theta);
}

} // namespace math
} // namespace tessera
