#ifndef CECE_OUTPUT_BATCH_HPP
#define CECE_OUTPUT_BATCH_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace cece {
// Use the maximum payload among writer ranks so every rank submits identical
// batch boundaries. Each AMIO batch occupies one staging buffer.
inline std::size_t OutputBatchSize(std::size_t capacity, std::size_t max_field_bytes, std::size_t remaining) {
    if (max_field_bytes == 0 || max_field_bytes > capacity) {
        throw std::invalid_argument("output field does not fit staging capacity");
    }
    return std::min({capacity / max_field_bytes, remaining, std::size_t{4096}});
}
}  // namespace cece
#endif
