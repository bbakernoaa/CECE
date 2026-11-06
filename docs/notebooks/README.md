# Notebook index

| Notebook | Runtime | What it is |
|---|---|---|
| [fengsha.ipynb](./fengsha.ipynb) | host kernel | FENGSHA driven by a synthetic atmospheric column; shows the scheme in isolation. |
| [fengsha_gfs.ipynb](./fengsha_gfs.ipynb) | `cece-gfs` kernel in `cece/cece-dev` | FENGSHA driven by real GFS data decoded through MONETIO, run with the brittle-impaction flux **on and off** so the two total fluxes can be compared, global scale as well as over the Sahel. Scheme options and the driver window come from [fengsha_gfs_kok.yaml](./fengsha_gfs_kok.yaml) and [fengsha_gfs_webb.yaml](./fengsha_gfs_webb.yaml) (each a `driver:` block plus a `physics_schemes:` block for `fengsha`); the notebook reads no environment variables, so those two files are the scheme's only configuration source — edit them and re-run. See [gfs-docker.md](./gfs-docker.md) for the launcher and container path. |
