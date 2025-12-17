#ifndef _BRIAN_RANDOMGENERATOR_H
#define _BRIAN_RANDOMGENERATOR_H

#include <random>
#include <cmath>
#include <iostream>

/**
 * @brief Random number generator class that provides reproducible
 * random sequences across different Brian2 backends.
 *
 * Uses std::mt19937 (Mersenne Twister) with the Jean-Sebastien Roy
 * algorithm for converting to uniform doubles, ensuring identical
 * sequences when seeded with the same value.
 *
 * This class is used by both C++ standalone mode and Cython runtime mode
 * to ensure cross-backend reproducibility.
 */
class RandomGenerator
{
private:
    std::mt19937 gen;
    double stored_gauss;
    bool has_stored_gauss;

public:
    RandomGenerator() : has_stored_gauss(false)
    {
        seed();
    }

    /**
     * @brief Seed with a random value from the system.
     */
    void seed()
    {
        std::random_device rd;
        gen.seed(rd());
        has_stored_gauss = false;
    }

    /**
     * @brief Seed with a specific value for reproducibility.
     * @param seed The seed value.
     */
    void seed(unsigned long seed_value)
    {
        gen.seed(seed_value);
        has_stored_gauss = false;
    }

    /**
     * @brief Generate a uniform random double in [0, 1)
     *
     * Uses the Jean-Sebastien Roy algorithm to extract 53 bits
     * of randomness from two consecutive MT19937 outputs.
     * This ensures reproducibility with older Brian2 versions
     * and across different backends.
     *
     * The algorithm:
     * - Takes two 32-bit outputs from MT19937
     * - Extracts 27 bits from the first (shift right 5)
     * - Extracts 26 bits from the second (shift right 6)
     * - Combines them into a 53-bit integer (full double precision mantissa)
     * - Divides by 2^53 to get a value in [0, 1)
     */
    double rand()
    {
        /* shifts : 67108864 = 0x4000000 = 2^26
         *          9007199254740992 = 0x20000000000000 = 2^53 */
        const long a = gen() >> 5; // Upper 27 bits
        const long b = gen() >> 6; // Upper 26 bits
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    /**
     * @brief Generate a standard normal random double (mean=0, std=1)
     *
     * Uses the Box-Muller transform with rejection sampling.
     * Generates two values at once and caches one for the next call,
     * making every other call essentially free.
     */
    double randn()
    {
        if (has_stored_gauss)
        {
            const double tmp = stored_gauss;
            has_stored_gauss = false;
            return tmp;
        }
        else
        {
            double f, x1, x2, r2;

            do
            {
                x1 = 2.0 * rand() - 1.0;
                x2 = 2.0 * rand() - 1.0;
                r2 = x1 * x1 + x2 * x2;
            } while (r2 >= 1.0 || r2 == 0.0);

            /* Box-Muller transform */
            f = sqrt(-2.0 * log(r2) / r2);
            /* Keep for next call */
            stored_gauss = f * x1;
            has_stored_gauss = true;
            return f * x2;
        }
    }

    // Allow exporting/setting the internal state of the random generator
    friend std::ostream &operator<<(std::ostream &out, const RandomGenerator &rng);
    friend std::istream &operator>>(std::istream &in, RandomGenerator &rng);
};

/**
 * @brief Stream output operator for serializing generator state.
 */
inline std::ostream &operator<<(std::ostream &out, const RandomGenerator &rng)
{
    return out << rng.gen;
}

/**
 * @brief Stream input operator for deserializing generator state.
 */
inline std::istream &operator>>(std::istream &in, RandomGenerator &rng)
{
    return in >> rng.gen;
}

#endif // _BRIAN_RANDOMGENERATOR_H
