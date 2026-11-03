# FENGSHA

## Overview

Implements the FENGSHA dust emission scheme, a physically-based saltation model that computes size-resolved dust emissions using friction velocity, soil texture, moisture corrections, and drag partitioning. The scheme includes Fécan moisture correction, Marticorena & Bergametti (1995) vertical-to-horizontal flux ratio, Comola-style emission intermittency, and splits the size-integrated flux across the size bins configured for the model.

The threshold, drag-partition and intermittency formulations follow the Leung_2023 dust emission scheme documented in the [CTSM 5.0 Technical Note, §2.31 "Dust Emission"](https://escomp.github.io/CTSM/tech_note/Dust/CLM50_Tech_Note_Dust.html#dust-emission-flux); equation numbers below refer to that note. Two total dust fluxes are available, selected by `use_brittle_impaction`: the Kok et al. (2014a) brittle-fragmentation flux of Leung_2023 (Eq. 2.31.8), and the original FENGSHA saltation flux of Webb et al. (2020, Eq. 9) combined with the Marticorena & Bergametti (1995) k_v/h ratio. The default is the Webb formulation.

References (bibliography follows the reference list of the CTSM note, §2.35):

- Comola, F., Kok, J.F., Chamecki, M. and Martin, R.L. (2019), *GRL*, 46(22), 13430–13440, doi:10.1029/2019GL085739. (intermittency η, saltation height z_sal, Eq. 2.31.22–2.31.30)
- Darmenova, K., Sokolik, I.N., Shao, Y., Marticorena, B. and Bergametti, G. (2009), *JGR: Atmospheres*, 114(D14), doi:10.1029/2008JD011236. (b1, b2, X rock-drag coefficients)
- Fécan, F., Marticorena, B. and Bergametti, G. (1998), *Annales Geophysicae*, 17(1), 149–157, doi:10.1007/s00585-999-0149-7. (soil moisture correction F_m, Eq. 2.31.5)
- Kok, J.F., Parteli, E.J., Michaels, T.I. and Karam, D.B. (2012), *Reports on Progress in Physics*, 75(10), 106901, doi:10.1088/0034-4885/75/10/106901. (impact threshold ratio B_*it = 0.81, Eq. 2.31.7)
- Kok, J.F., Mahowald, N.M., et al. (2014a), *ACP*, 14(23), 13023–13041, doi:10.5194/acp-14-13023-2014. (brittle-fragmentation emission flux, Eq. 2.31.8–2.31.13)
- Kok, J.F., Albani, S., Mahowald, N.M. and Ward, D.S. (2014b), *ACP*, 14(23), 13043–13061, doi:10.5194/acp-14-13043-2014. (size distribution D(n))
- Leung, D.M., Kok, J.F., et al. (2023), *ACP*, 23(11), 6487–6523, doi:10.5194/acp-23-6487-2023. (scheme structure, η, F_eff)
- Leung, D.M., Kok, J.F., et al. (2024), *ACP*, 24(4), 2287–2318, doi:10.5194/acp-24-2287-2024. (CTSM implementation and tuning)
- Marticorena, B. and Bergametti, G. (1995), *JGR*, 100(D8), 16415–16430, doi:10.1029/95JD00690. (k_v/h ratio, rock drag partition, Eq. 2.31.19)
- Okin, G.S. (2008), *JGR: Earth Surface*, 113(F2), doi:10.1029/2007JF000758. (vegetation drag partition, Eq. 2.31.20)
- Panofsky, H.A., Tennekes, H., Lenschow, D.H. and Wyngaard, J.C. (1977), *Boundary-Layer Meteorology*, 11(3), 355–361, doi:10.1007/BF02186086. (wind fluctuation σ_us, Eq. 2.31.26)
- Pierre, C., Bergametti, G., et al. (2014), *JGR: Earth Surface*, 119(10), 2291–2313, doi:10.1002/2014JF003177. (gap length K, Eq. 2.31.21)
- Prigent, C., Tegen, I., Aires, F., Marticorena, B. and Zribi, M. (2005), *JGR: Atmospheres*, 110(D9), doi:10.1029/2004JD005370. (aeolian roughness z_0a)
- Shao, Y. and Lu, H. (2000), *JGR: Atmospheres*, 105(D17), 22437–22443, doi:10.1029/2000JD900304. (dry fluid threshold u_*ft0, Eq. 2.31.2)
- Zender, C.S., Bian, H. and Newman, D. (2003), *JGR*, 108(D14), 4416, doi:10.1029/2002JD002775. (DEAD; clay breakpoint, ρ_p, f_clay')

Not in the CTSM reference list (from the FENGSHA formulation itself):

- Webb, N.P., et al. (2020), *Current Opinion in Environmental Sustainability*, 44, 138–146. (horizontal saltation flux q, Eq. 9)

## Registration Names

- Native C++: `"fengsha"`
- Fortran bridge: `"fengsha_fortran"`

## Configuration Parameters

| YAML Key | Type | Default | Description |
| --- | --- | --- | --- |
| `alpha` | double | 1.0 | Sandblasting efficiency parameter |
| `gamma` | double | 1.0 | Erodibility exponent for SSM |
| `kvhmax` | double | 2.45e-4 | Maximum vertical-to-horizontal flux ratio |
| `grav` | double | 9.81 | Gravitational acceleration [m/s²] |
| `drylimit_factor` | double | 1.0 | Fécan dry-limit scaling factor (b in w_t = b·f_clay(14 f_clay + 17)) |
| `use_intermittency` | bool | true | Apply the emission intermittency factor η (Eq. 2.31.27) |
| `use_brittle_impaction` | bool | false | Use the Kok et al. (2014a) brittle-fragmentation flux F_d (Eq. 2.31.8) instead of the Webb et al. (2020) saltation flux |
| `c_tune` | double | 0.05 | Tuning constant C_tune of the brittle-impaction flux (Eq. 2.31.8) |
| `particle_radii` | sequence[double] | *(none)* | Representative radius of each size bin [m] |
| `bin_lower_edges` | sequence[double] | *(none)* | Lower edge radius of each size bin [m] (with `particle_radii`) |
| `bin_upper_edges` | sequence[double] | *(none)* | Upper edge radius of each size bin [m] (with `particle_radii`) |
| `bin_weights` | sequence[double] | *(none)* | Emission mass fraction of each bin [-]; overrides the computed Kok weights |

### Size bins

The number of size bins is **not** a scheme option: it is the third (size)
dimension of the `fengsha_dust_emissions` field, so the scheme works for any bin
count the grid is configured with. The deprecated `num_bins` key is rejected with
an error. What the scheme does need is the bin *geometry*, resolved by
`cece::dustbins::Resolve` (see `include/cece/cece_dust_bins.hpp`) in this order:

1. `particle_radii` + `bin_lower_edges` + `bin_upper_edges` — the Kok et al.
   (2014b) brittle-fragmentation distribution is integrated over each bin to give
   the weights (this is where the size-resolved split comes from);
2. the same geometry plus `bin_weights` — your weights are used (renormalised to
   sum to 1) instead of the Kok distribution;
3. `bin_weights` alone — your weights applied to whatever bins the emission field
   has, with the per-bin sizes left entirely to the driver's threshold field;
4. nothing — the flux is split equally over the field bins and a warning is
   logged, so the fallback is visible.

A partial geometry (for example radii without edges), a length mismatch, a
representative radius outside its own bin, or values that cannot be metres are
rejected at initialization rather than silently corrected.

## Import Fields

Required:

| Field Name | Units | Description |
| --- | --- | --- |
| `friction_velocity` | m/s | Surface friction velocity (u\*) |
| `threshold_velocity` | m/s | Dry fluid threshold friction velocity (u\*_ft0) |
| `soil_moisture` | fraction | Volumetric soil liquid water content |
| `clay_fraction` | fraction | Fractional clay content [0–1] |
| `sand_fraction` | fraction | Fractional sand content [0–1] |
| `silt_fraction` | fraction | Fractional silt content [0–1] |
| `erodibility` | dimensionless | Sediment supply map (SSM) |
| `drag_partition` | dimensionless | Drag partition correction factor (F_eff, Eq. 2.31.16) |
| `air_density` | kg/m³ | Surface air density |
| `lake_fraction` | fraction | Lake fraction [0–1] |
| `snow_fraction` | fraction | Snow cover fraction [0–1] |
| `land_mask` | dimensionless | Land mask (1 = land) |

Optional (fallbacks match CTSM's Leung_2023 defaults):

| Field Name | Units | Description | Fallback when absent |
| --- | --- | --- | --- |
| `soil_temperature` | K | Top soil layer temperature | Soil assumed unfrozen (no frozen-soil shutoff) |
| `leaf_area_index` | m²/m² | Leaf area index; with SAI gives VAI for f_v (Eq. 2.31.15) | No vegetation burial (f_v = 0) |
| `stem_area_index` | m²/m² | Stem area index | 0 (VAI = LAI) |
| `obukhov_length` | m | Monin-Obukhov length L in σ_us (Eq. 2.31.26) | Neutral limit (L = 1e6 m) |
| `pbl_height` | m | Planetary boundary layer height z_i (Eq. 2.31.26) | 1000 m (CTSM constant) |

## Export Fields

| Field Name | Units | Description |
| --- | --- | --- |
| `fengsha_dust_emissions` | kg/m²/s | Size-resolved dust emission flux (3D: nx × ny × nbins) |

## Algorithm

1. Skip non-land cells and cells with erodibility (SSM) below threshold (0.01).
2. Compute bare-soil fraction f_bare (Eq. 2.31.14) from lake, snow, and vegetation cover; emissions shut off where top soil temperature ≤ 273.15 K.
3. Apply drag partition: `u*s = u* · F_eff` (Eq. 2.31.16).
4. Convert volumetric to gravimetric soil moisture, then compute the Fécan moisture factor F_m (Eq. 2.31.5) and the wet fluid threshold `u_ft = threshold_velocity · F_m` (Eq. 2.31.1); impact threshold `u_it = 0.81 · threshold_velocity` (Eq. 2.31.7).
5. If `use_intermittency`, compute η (Eq. 2.31.22–2.31.30, Comola et al. 2019): the fraction of the timestep with active emission under turbulent wind fluctuations; η = 0 when u_*s < u_*it. Otherwise η = 1.
6. Compute the total vertical dust mass flux:
   - `use_brittle_impaction: true` — Kok et al. (2014a) brittle fragmentation (Eq. 2.31.8):
     `F_d = η · C_tune · C_d · f_bare · f_clay' · ρ_atm · (u_*s² − u_*it²)/u_*it² · (u_*s/u_*it)^κ`,
     with the transient erodibility `C_d = C_d0 exp[−C_e (u_st − u_st0)/u_st0]` (Eq. 2.31.11), the fragmentation exponent `κ = min(C_κ (u_st − u_st0)/u_st0, 2.5)` (Eq. 2.31.12, C_κ = 2.7), the standardised threshold `u_st = u_ft √(ρ_atm/1.225)` (Eq. 2.31.13) against the constant `u_st0 = 0.16 m/s`, and the CESM-tuned clay factor `f_clay' = max(0.1 + 0.5 f_clay, 0.2)` (Eq. 2.31.10).
   - `use_brittle_impaction: false` — original FENGSHA form:
     `F = η · alpha · f_bare · SSM^gamma · ρ_atm · k_v/h · q`, with the Marticorena & Bergametti (1995) ratio `k_v/h = 10^(13.4 clay − 6)` (capped at `kvhmax` for clay > 0.2) and the Webb et al. (2020, Eq. 9) horizontal saltation flux `q`. **WIP:** `q` is currently a placeholder (`max(0, u_ft / F_eff)`), and the `alpha/g` sandblasting prefactor of the original formulation is not yet applied.
7. Distribute across the size bins of the emission field using the configured
   weights (`bin_weights`, or the Kok et al. (2014b) weights derived from a
   configured bin geometry): `emissions(n) += D(n) · F`. With no bin
   configuration `D(n) = 1/nbins`, so the size-integrated flux is unchanged.

## YAML Configuration Example

```yaml
physics:
  - name: fengsha
    config:
      alpha: 1.0
      gamma: 1.0
      kvhmax: 2.45e-4
      drylimit_factor: 1.0
      use_intermittency: true
      use_brittle_impaction: false
      c_tune: 0.05
      particle_radii: [0.55e-6, 1.75e-6, 3.75e-6, 7.5e-6]
      bin_lower_edges: [0.1e-6, 1.0e-6, 2.5e-6, 5.0e-6]
      bin_upper_edges: [1.0e-6, 2.5e-6, 5.0e-6, 10.0e-6]
```

The emission field must then have a size dimension of 4 to match the configured
bins; with any other size dimension the scheme warns and splits equally.

## Implementation Notes

- Available as both native C++ (Kokkos) and Fortran bridge implementations
- Produces multi-bin (3D) output; the third dimension is the size bin index
- Size bins are configuration-driven (`particle_radii`/`bin_lower_edges`/`bin_upper_edges` or `bin_weights`); the bin count comes from the emission field, and with no bin configuration the flux is split equally across it with a warning
- Requires 12 mandatory import fields plus 5 optional ones — the most input-intensive scheme alongside K14
- Intermittency uses the CTSM-fixed aeolian roughness z_0a = 1e-4 m and saltation height z_sal = 0.1 m (CTSM §2.31.4)
- The brittle-impaction flux ignores the SSM erodibility field: in Leung_2023 the transient soil erodibility C_d replaces the static sediment-supply map (Kok et al. 2014a)
- Frozen soil is a hard shutoff rather than the CTSM liquid/ice water ratio w_liq/(w_liq + w_ice) of Eq. 2.31.14, since FENGSHA carries no separate liquid/ice soil water
- `fengsha_mb95_drag_partition` (Eq. 2.31.19) and `fengsha_leung_drag_partition` (Eq. 2.31.20–2.31.21) provide the rock/vegetation F_eff components; the driver currently supplies the combined `drag_partition` field
