#define CHECK(condition)                   \
    do {                                   \
        if (!(condition)) return __LINE__; \
    } while (false)
#include <stdexcept>

#include "cece_output_batch.hpp"
int main() {
    CHECK(cece::OutputBatchSize(100, 30, 10) == 3);
    CHECK(cece::OutputBatchSize(100, 30, 2) == 2);
    CHECK(cece::OutputBatchSize(100000, 1, 10000) == 4096);
    CHECK(cece::OutputBatchSize(100, 30, 0) == 0);
    bool rejected = false;
    try {
        cece::OutputBatchSize(20, 30, 10);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try {
        cece::OutputBatchSize(100, 0, 10);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}
