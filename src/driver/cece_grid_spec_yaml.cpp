// SPDX-License-Identifier: Apache-2.0
// CECE — Chemical Emissions Coupling Engine
// Copyright (c) HELM Project Contributors

/**
 * @file cece_grid_spec_yaml.cpp
 * @brief GridSpec::from_yaml / Validate — the single YAML grid-resolution
 * path shared by both CECE drivers.
 *
 * This is the grid-dimension and coordinate-resolution logic relocated from
 * the standalone driver's main() so the NUOPC cap's config-built branch runs
 * literally the same code (parity by construction). Named-grid resolution via
 * AXIS, explicit/stream-inferred GRIDSPEC coordinate loading through AMIO
 * (with CF unpacking and radian handling), and the uniform-extents fallback
 * all live here; failures throw std::invalid_argument with the same
 * diagnostics the driver has always logged, and the facade converts them to
 * rc<0 (never a silent fallback).
 */

#include <amio/amio.h>

#include <algorithm>
#include <axis/topology/named_grid_registry.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cece/cece_amio_utils.hpp"
#include "cece/cece_grid_detail.hpp"
#include "cece/cece_logger.hpp"
#include "cece/cece_simulation.hpp"

namespace {

// Build the in-memory AMIO coordinate-manifest YAML for reading lon/lat out
// of `path`. Kept as a named helper so the manifest schema (backend, staging
// pool, worker pool, prefetch tuning) lives in exactly one place; the call
// site only opens/reads with the returned string. Writing it to a shared-disk
// file (e.g. Lustre) races when multiple MPI ranks per node truncate/rewrite
// the same path concurrently, which produces torn reads (empty/partial YAML)
// and spurious open failures — hence in-memory.
std::string BuildCoordinateManifest(const std::string& path) {
    std::ostringstream manifest;
    manifest << "backend: netcdf4\n"
             << "path: " << path << "\n"
             << "data_model: enhanced\n"
             << "staging_pool:\n"
             << "  buffer_count: 16\n"
             << "  buffer_capacity_bytes: 33554432\n"
             << "worker_pool:\n"
             << "  threads: 1\n"
             << "prefetch:\n"
             << "  depth: 4\n"
             << "  read_timeout_s: 60\n"
             << "staging_timeout_ms: 10000\n";
    return manifest.str();
}

/// Read one flattened coordinate array (lon or lat) out of an opened AMIO
/// dataset, trying each candidate variable name in order. Applies CF packing,
/// radian conversion (when `is_radian`), and longitude wrapping (when
/// `wrap_lon`). Returns the decoded values plus the extent along the
/// coordinate's leading dimension(s). `total_len` reports the element count.
bool read_coord(amio_dataset_handle dataset, const std::vector<std::string>& names, bool is_radian, bool wrap_lon, std::vector<double>& out,
                int& extent0, int& extent1) {
    extent0 = 0;
    extent1 = 0;
    for (const auto& name : names) {
        amio_view_handle view = nullptr;
        if (amio_read(dataset, name.c_str(), 0, nullptr, &view) != AMIO_OK) {
            continue;
        }
        const void* view_data = nullptr;
        size_t view_size = 0;
        if (amio_view_data(view, &view_data, &view_size) != AMIO_OK) {
            amio_release_view(view);
            return false;
        }
        amio_shape_t shape{};
        if (amio_view_shape(view, &shape) != AMIO_OK) {
            amio_release_view(view);
            return false;
        }
        if (shape.rank == 1) {
            extent0 = static_cast<int>(shape.extents[0]);
        } else if (shape.rank == 2) {
            extent0 = static_cast<int>(shape.extents[0]);
            extent1 = static_cast<int>(shape.extents[1]);
        } else {
            amio_release_view(view);
            return false;
        }
        int total_len = 1;
        for (int r = 0; r < shape.rank; ++r) {
            total_len *= static_cast<int>(shape.extents[r]);
        }
        amio_dtype_t dtype = AMIO_DTYPE_F64;
        double scale = 1.0;
        double offset = 0.0;
        cece::detail::read_cf_packing(dataset, name, scale, offset);
        std::vector<double> widened;
        const bool ok = amio_view_dtype(view, &dtype) == AMIO_OK &&
                        cece::detail::widen_amio_elements(view_data, dtype, static_cast<std::size_t>(total_len), scale, offset, widened);
        if (!ok) {
            // Leaving extent0 set here would let an empty coordinate array
            // pass as a loaded gridspec.
            CECE_LOG_ERROR("Could not decode gridspec coordinate variable '" + name + "'");
            extent0 = 0;
            amio_release_view(view);
            return false;
        }
        out.resize(total_len);
        for (int i = 0; i < total_len; ++i) {
            double val = widened[i];
            if (is_radian) {
                val = cece::detail::radians_to_degrees(val);
            }
            out[i] = wrap_lon ? cece::detail::wrap_longitude(val) : val;
        }
        amio_release_view(view);
        return true;
    }
    return false;
}

}  // namespace

namespace cece {

// Candidate longitude / latitude variable names, in priority order. Shared
// with the historical driver behavior exactly.
static const std::vector<std::string> kLonNames = {"grid_lont", "grid_lon",        "XLONG",         "lonCell",   "geolon",  "clon",
                                                   "glamt",     "mesh2d_face_lon", "lon",           "longitude", "LON",     "lon_rho",
                                                   "nav_lon",   "mesh_node_x",     "mesh2d_node_x", "node_x",    "grid_xt", "x"};
static const std::vector<std::string> kLatNames = {"grid_latt", "grid_lat",        "XLAT",          "latCell",  "geolat",  "clat",
                                                   "gphit",     "mesh2d_face_lat", "lat",           "latitude", "LAT",     "lat_rho",
                                                   "nav_lat",   "mesh_node_y",     "mesh2d_node_y", "node_y",   "grid_yt", "y"};

GridSpec GridSpec::from_yaml(const std::string& config_file, conf::Config& config) {
    GridSpec spec;

    // --- A. Grid dimensions ------------------------------------------------
    int nx = 0;
    int ny = 0;
    int nz = 1;
    std::string grid_name = "";
    if (config.has("driver.grid")) {
        nz = config.get_or("driver.grid.nz", 1);
        grid_name = config.get_or<std::string>("driver.grid.grid_name", "");
        if (grid_name.empty()) {
            nx = config.get_or("driver.grid.nx", 0);
            ny = config.get_or("driver.grid.ny", 0);
        } else {
            try {
                auto parsed = axis::topology::NamedGridRegistry::parse(grid_name);
                if (parsed.family == 'F' || parsed.family == 'R') {
                    int expected_nx = 4 * parsed.number;
                    int expected_ny = 2 * parsed.number;

                    int declared_nx = config.get_or("driver.grid.nx", 0);
                    int declared_ny = config.get_or("driver.grid.ny", 0);
                    if (declared_nx != 0 && declared_ny != 0) {
                        if (declared_nx != expected_nx || declared_ny != expected_ny) {
                            throw std::invalid_argument("Grid dimensions nx=" + std::to_string(declared_nx) + ", ny=" + std::to_string(declared_ny) +
                                                        " do not match the expected dimensions for Named Grid " + grid_name + " (" +
                                                        std::to_string(expected_nx) + "x" + std::to_string(expected_ny) + ")!");
                        }
                    }
                    nx = expected_nx;
                    ny = expected_ny;
                } else {
                    throw std::invalid_argument(
                        "Only regular Gaussian grids (family 'F', e.g. 'F360') and regular lat-lon grids (family 'R', e.g. "
                        "'R360') are currently supported as structured CECE target grids.");
                }
            } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Failed to parse named grid '") + grid_name + "': " + e.what());
            }
        }
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument(
            "driver.grid.nx, driver.grid.ny, and driver.grid.nz must be positive, or "
            "driver.grid.grid_name must specify a supported named grid.");
    }

    CECE_LOG_DEBUG("[GRID] Parsed nx = " + std::to_string(nx) + ", ny = " + std::to_string(ny) + ", grid_name = '" + grid_name + "'");

    spec.nx = nx;
    spec.ny = ny;
    spec.nz = nz;

    // --- B. Coordinate arrays ----------------------------------------------
    std::vector<double> file_lons(static_cast<size_t>(nx), 0.0);
    std::vector<double> file_lats(static_cast<size_t>(ny == 1 ? nx : ny), 0.0);

    if (!grid_name.empty()) {
        // Named grid: generate the mesh via AXIS and extract 1-D center
        // coordinates (sorted).
        try {
            auto mesh = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(grid_name);
            auto coords = mesh.node_coords();
            for (int i = 0; i < nx; ++i) {
                file_lons[static_cast<size_t>(i)] = cece::detail::wrap_longitude(coords(i, 0));
            }
            for (int j = 0; j < ny; ++j) {
                file_lats[static_cast<size_t>(j)] = coords(static_cast<long>(j) * nx, 1);
            }
            std::sort(file_lons.begin(), file_lons.end());
            std::sort(file_lats.begin(), file_lats.end());
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("Failed to retrieve coordinates from named grid '") + grid_name + "': " + e.what());
        }
        spec.topology = GridTopology::Rectilinear;
    } else {
        bool loaded_from_file = false;
        bool is_explicit_gridspec = false;
        std::string input_file_path = "";
        auto gridspec_opt = config.try_string("driver.gridspec_file");
        if (gridspec_opt.has_value() && !gridspec_opt->empty() && *gridspec_opt != "none" && *gridspec_opt != "NONE") {
            input_file_path = *gridspec_opt;
            is_explicit_gridspec = true;
        }
        if (input_file_path.empty() && config.has("cece_data.streams")) {
            auto streams = config.at("cece_data.streams");
            if (streams.size() > 0) {
                auto first_stream = streams[static_cast<std::size_t>(0)];
                auto file_val = first_stream["file"];
                if (file_val.is_defined()) {
                    input_file_path = file_val.as_string();
                }
            }
        }

        if (!input_file_path.empty()) {
            const std::string coord_manifest_content = BuildCoordinateManifest(input_file_path);

            amio_core_handle coord_core = nullptr;
            amio_dataset_handle coord_dataset = nullptr;

            amio_status_t amio_rc = amio_init_from_string(coord_manifest_content.c_str(), "yaml", &coord_core);
            if (amio_rc != AMIO_OK) {
                CECE_LOG_ERROR(std::string("amio_init_from_string failed for coordinate manifest: ") + amio_strerror(amio_rc));
            } else {
                amio_rc = amio_open_dataset_from_string(coord_core, coord_manifest_content.c_str(), "yaml", AMIO_MODE_READ, &coord_dataset);
                if (amio_rc != AMIO_OK) {
                    CECE_LOG_ERROR("amio_open_dataset_from_string failed for dataset '" + input_file_path + "': " + amio_strerror(amio_rc));
                } else {
                    int file_nx = 0;
                    int file_ny = 0;
                    std::vector<double> file_lon_coords;
                    std::vector<double> file_lat_coords;

                    // Longitude: is_radian only for the cell-centered cubed-
                    // sphere names, matching the historical driver rule.
                    int lon_e0 = 0;
                    int lon_e1 = 0;
                    bool lon_found = false;
                    bool is_radian = false;
                    for (const auto& name : kLonNames) {
                        amio_view_handle probe = nullptr;
                        if (amio_read(coord_dataset, name.c_str(), 0, nullptr, &probe) == AMIO_OK) {
                            amio_release_view(probe);
                            if (name == "lonCell" || name == "latCell" || name == "lonVertex" || name == "latVertex") {
                                is_radian = true;
                            }
                            lon_found = true;
                            break;
                        }
                    }
                    if (lon_found && read_coord(coord_dataset, kLonNames, is_radian, /*wrap_lon=*/true, file_lon_coords, lon_e0, lon_e1)) {
                        if (lon_e1 != 0) {
                            file_nx = lon_e1;  // 2-D: extents[1] is the x extent
                        } else {
                            file_nx = lon_e0;
                        }
                    }

                    bool lat_found = false;
                    for (const auto& name : kLatNames) {
                        amio_view_handle probe = nullptr;
                        if (amio_read(coord_dataset, name.c_str(), 0, nullptr, &probe) == AMIO_OK) {
                            amio_release_view(probe);
                            lat_found = true;
                            break;
                        }
                    }
                    int lat_e0 = 0;
                    int lat_e1 = 0;
                    if (lat_found && read_coord(coord_dataset, kLatNames, /*is_radian=*/false, /*wrap_lon=*/false, file_lat_coords, lat_e0, lat_e1)) {
                        // Rank-1 or rank-2 latitude: the y extent is extents[0].
                        file_ny = lat_e0;
                    }

                    // If nx and ny are not specified in the configuration, dynamically
                    // inherit them from the gridspec file. (The dimension guard above
                    // requires them to be declared today; this stays for the
                    // historical behavior contract.)
                    if (nx == 0 && file_nx > 0) {
                        nx = file_nx;
                    }
                    if (ny == 0 && file_ny > 0) {
                        ny = (file_ny == file_nx) ? 1 : file_ny;
                    }

                    if (nx == file_nx && (ny == file_ny || (ny == 1 && file_ny == file_nx)) && file_nx > 0 && file_ny > 0) {
                        file_lons = file_lon_coords;
                        file_lats = file_lat_coords;
                        loaded_from_file = true;
                        spec.gridspec_file = input_file_path;
                        if (ny == 1) {
                            spec.topology = GridTopology::Unstructured;
                        } else if (file_lons.size() == static_cast<size_t>(nx) * static_cast<size_t>(ny) &&
                                   file_lats.size() == static_cast<size_t>(nx) * static_cast<size_t>(ny)) {
                            spec.topology = GridTopology::Curvilinear;
                        } else {
                            spec.topology = GridTopology::Rectilinear;
                        }
                    }
                }
                amio_close(coord_dataset);
            }
            amio_finalize(coord_core);
        }

        if (is_explicit_gridspec && !loaded_from_file) {
            throw std::invalid_argument("[GRID] Failed to load gridspec coordinates from explicitly specified gridspec file '" + input_file_path +
                                        "'");
        }

        if (!loaded_from_file) {
            if (nx <= 0 || ny <= 0) {
                throw std::invalid_argument(
                    "Grid dimensions (nx, ny) were not specified in driver.grid configuration and could not be determined from "
                    "input files!");
            }

            double lon_min = config.get_or("driver.grid.lon_min", -180.0);
            double lon_max = config.get_or("driver.grid.lon_max", 180.0);
            double lat_min = config.get_or("driver.grid.lat_min", -90.0);
            double lat_max = config.get_or("driver.grid.lat_max", 90.0);

            double dlon = (lon_max - lon_min) / nx;
            double dlat = (lat_max - lat_min) / ny;

            file_lons.assign(static_cast<size_t>(nx), 0.0);
            file_lats.assign(static_cast<size_t>(ny == 1 ? nx : ny), 0.0);
            for (int i = 0; i < nx; ++i) {
                file_lons[static_cast<size_t>(i)] = lon_min + dlon * (i + 0.5);
            }
            for (int j = 0; j < ny; ++j) {
                file_lats[static_cast<size_t>(j)] = lat_min + dlat * (j + 0.5);
            }
            spec.topology = (ny == 1) ? GridTopology::Unstructured : GridTopology::Rectilinear;
        }
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Invalid grid dimensions nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny) + ", nz=" + std::to_string(nz));
    }

    spec.nx = nx;
    spec.ny = ny;
    spec.nz = nz;
    spec.lon_coords = std::move(file_lons);
    spec.lat_coords = std::move(file_lats);

    spec.Validate();
    return spec;
}

void GridSpec::Validate() const {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("GridSpec: nx, ny, and nz must be positive (got nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny) +
                                    ", nz=" + std::to_string(nz) + ")");
    }
    const auto expect_size = [&](size_t want, const std::string& which, size_t got) {
        if (got != want) {
            throw std::invalid_argument("GridSpec: " + which + " array length " + std::to_string(got) +
                                        " does not match the declared topology/dimensions (expected " + std::to_string(want) + ")");
        }
    };
    switch (topology) {
        case GridTopology::Rectilinear:
            expect_size(static_cast<size_t>(nx), "lon", lon_coords.size());
            expect_size(static_cast<size_t>(ny == 1 ? nx : ny), "lat", lat_coords.size());
            break;
        case GridTopology::Curvilinear:
            expect_size(static_cast<size_t>(nx) * static_cast<size_t>(ny), "lon", lon_coords.size());
            expect_size(static_cast<size_t>(nx) * static_cast<size_t>(ny), "lat", lat_coords.size());
            break;
        case GridTopology::Unstructured:
            expect_size(static_cast<size_t>(nx), "lon", lon_coords.size());
            expect_size(static_cast<size_t>(nx), "lat", lat_coords.size());
            break;
    }
}

}  // namespace cece
