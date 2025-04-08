athena
======
<!-- Jenkins Status Badge in Markdown (with view), unprotected, flat style -->
<!-- In general, need to be on Princeton VPN, logged into Princeton CAS, with ViewStatus access to Jenkins instance to click on unprotected Build Status Badge, but server is configured to whitelist GitHub -->
<!-- [![Jenkins Build Status](https://jenkins.princeton.edu/buildStatus/icon?job=athena/PrincetonUniversity_athena_jenkins_master)](https://jenkins.princeton.edu/job/athena/job/PrincetonUniversity_athena_jenkins_master/) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11660592.svg)](https://doi.org/10.5281/zenodo.11660592)
[![codecov](https://codecov.io/gh/PrincetonUniversity/athena/branch/master/graph/badge.svg?token=ZzniY084kP)](https://codecov.io/gh/PrincetonUniversity/athena)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](code_of_conduct.md)

<!--[![Public GitHub  issues](https://img.shields.io/github/issues/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/issues)
[![Public GitHub pull requests](https://img.shields.io/github/issues-pr/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/pulls) -->

<p align="center">
	  <img width="345" height="345" src="https://user-images.githubusercontent.com/1410981/115276281-759d8580-a108-11eb-9fc9-833480b97f95.png">
</p>

Athena++ radiation GRMHD code and adaptive mesh refinement (AMR) framework

Please read [our contributing guidelines](./CONTRIBUTING.md) for details on how to participate.

## Citation
To cite Athena++ in your publication, please use the following BibTeX to refer to the code's [method paper](https://ui.adsabs.harvard.edu/abs/2020ApJS..249....4S/abstract):
```
@article{Stone2020,
	doi = {10.3847/1538-4365/ab929b},
	url = {https://doi.org/10.3847%2F1538-4365%2Fab929b},
	year = 2020,
	month = jun,
	publisher = {American Astronomical Society},
	volume = {249},
	number = {1},
	pages = {4},
	author = {James M. Stone and Kengo Tomida and Christopher J. White and Kyle G. Felker},
	title = {The Athena$\mathplus$$\mathplus$ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers},
	journal = {The Astrophysical Journal Supplement Series},
}
```
Additionally, you can add a reference to `https://github.com/PrincetonUniversity/athena` in a footnote.

Finally, we have minted DOIs for each released version of Athena++ on Zenodo. This practice encourages computational reproducibility, since you can specify exactly which version of the code was used to produce the results in your publication. `10.5281/zenodo.4455879` is the DOI which cites _all_ versions of the code; it will always resolve to the latest release. Click on the Zenodo badge above to get access to BibTeX, etc. info related to these DOIs, e.g.:

```
@software{athena,
  author       = {Athena++ development team},
  title        = {PrincetonUniversity/athena: Athena++ v24.0},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {24.0},
  doi          = {10.5281/zenodo.11660592},
  url          = {https://doi.org/10.5281/zenodo.11660592}
}
```


To cite this multifluid dust module, please use the following BibTeX entry.

```
@ARTICLE{HuangBai2022,
       author = {{Huang}, Pinghui and {Bai}, Xue-Ning},
        title = "{A Multifluid Dust Module in Athena++: Algorithms and Numerical Tests}",
      journal = {\apjs},
     keywords = {Hydrodynamics, Protoplanetary disks, Computational methods, 1963, 1300, 1965, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = sep,
       volume = {262},
       number = {1},
          eid = {11},
        pages = {11},
          doi = {10.3847/1538-4365/ac76cb},
archivePrefix = {arXiv},
       eprint = {2206.01023},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJS..262...11H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

***How to Use the Multifluid Dust Module in Athena++***

***Step 1. Compile***

Run configure.py with the option **--ndustfluids=NDUSTFLUIDS** to set the number of dust fluid species (**NDUSTFLUIDS**) in the code.

***Step 2. Problem Setup in the Problem File***

2.1 Initialize Conservative Variables for Dust Fluids

In the code, dust is treated as multiple neutral, pressureless (or spatially isothermal) fluids. Each dust fluid is characterized by its stopping time. The system includes **4×NDUSTFLUIDS** equations for dust continuity and momentum.

```
Conservative Variables: Dust density and momenta (pmb->pdustfluids->df_u)
Primitive Variables: Dust density and velocities (pmb->pdustfluids->df_w)
```

Users must initialize the conservative variables for each dust fluid in the function **MeshBlock::ProblemGenerator**.
The indices for density and momenta along the three spatial directions are:

```
rho_id, v1_id, v2_id, v3_id.
```

Example:

```
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int n = 0; n < NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4 * dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    pdustfluids->df_u(rho_id, k, j, i) = ...;
    pdustfluids->df_u(v1_id,  k, j, i) = ...;
    pdustfluids->df_u(v2_id,  k, j, i) = ...;
    pdustfluids->df_u(v3_id,  k, j, i) = ...;
  }
}
```

Momentum Correction for Dust Diffusion:

If **Diffusion_Flag = true** and **Momentum_Diffusion_Flag = true** (in the input file), initialize the dust diffusion momentum as follows:

```
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int n = 0; n < NDUSTFLUIDS; ++n) {
    int rho_id  = 4 * n;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    pdustfluids->dfccdif.diff_mom_cc(v1_id, k, j, i) = ...;
    pdustfluids->dfccdif.diff_mom_cc(v2_id, k, j, i) = ...;
    pdustfluids->dfccdif.diff_mom_cc(v3_id, k, j, i) = ...;
  }
}
```

2.2 Set Up Dust Stopping Time

Stopping time for each dust species can be customized using EnrollUserDustStoppingTime in **Mesh::InitUserMeshData**. Stopping time can depend on space, time, or other variables.

Example:

```
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (NDUSTFLUIDS > 0)
    EnrollUserDustStoppingTime(MyStoppingTime);
}
```
```
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
		const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
		const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

  for (int n = 0; n < NDUSTFLUIDS; ++n) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; ++i) {
          stopping_time(n, k, j, i) = ...;
        }
      }
    }
  }
}
```

If **EnrollUserDustStoppingTime** is not used, **constant stopping times** must be specified in the <dust> block of the input file:

```
<dust>
stopping_time_1 = ...
stopping_time_2 = ...
...
```

2.3 Set Up Dust Diffusivities and Dust Sound Speed
If **Diffusion_Flag = true**, dust diffusivity must be set for each species using **EnrollDustDiffusivity**.
Dust sound speed cs​ is related to diffusivity ν as **cs=ν/Teddy**​​, where **Teddy** ​ (eddy turnover time) is user-defined, defaulting to 1.0.

Example:
```
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (NDUSTFLUIDS > 0)
    EnrollDustDiffusivity(MyDustDiffusivity);
}
```
```
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb, const AthenaArray<Real> &w,
			const AthenaArray<Real> &prim_df, const AthenaArray<Real> &stopping_time,
			AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
			int is, int ie, int js, int je, int ks, int ke) {
  for (int n = 0; n < NDUSTFLUIDS; ++n) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          nu_dust(n, k, j, i) = ...;
          cs_dust(n, k, j, i) = ...;
        }
      }
    }
  }
}
```

If **EnrollDustDiffusivity** is not used, specify **constant dust diffusivities** in the <dust> block:
```
<dust>
nu_dust_1 = ...
nu_dust_2 = ...
...
```

2.4 Add Source Terms

Source terms can be added using **EnrollUserExplicitSourceFunction**.

2.5 Boundary Conditions

Specify dust boundary conditions using standard  (e.g., "outflow", "reflecting") or user-defined flags. For user-defined boundary conditions, use **EnrollUserBoundaryFunction**.

***Step 3. Input File Parameters***

Set dust-specific parameters in the <dust> block of the input file, including flags for diffusion, feedback and etc.

```
<dust>
DustFeedback_Flag       = true            # The flag of dust feedback terms on gas, true or false.

Diffusion_Flag          = true            # The flag of dust diffusion, true or false.

Momentum_Diffusion_Flag = true            # The flag of momentum correction of dust diffusion, true or false.
					  # The default value is false. It is valid with "Diffusion_Flag = true".

Dust_SoundSpeed_Flag    = false           # The flag of sound speed of dust, true or false, cs_dust = sqrt(nu_dust/T_eddy).
					  # The default value is false. It is valid with "Diffusion_Flag = true".

Dissipation_Flag        = true            # The flag of gas energy dissipation from dust feedback. The default value is true.
					  # It is valid only if "DustFeedback_Flag = true" and "time >= time_drag" in NON_BAROTROPIC_EOS cases.

solver_id               = 0               # The id of dust Riemann solver, from 0 to 3. The default value is 0.
					  # The "solver_id = 0" is for the penetrated solver, 1 for the non-penetrated solver.
					  # 2 for the HLLE solver without the sound speed of dust, 3 for the HLLE solver with the sound speed of dust.

dust_xorder             = 2               # The spatial reconstruction order for dust. The "dust_xorder" must be equaled to or smaller than the "xorder" in <time> block, i.e., dust_xorder <= xorder.
					  # The default value is "xorder" in <time> block. The smaller spatial reconstruction order is more robust in the region where densities have strong spatial
					  # variations. The recommended value of dust_xorder is 2.

time_drag               = 0.0             # The drag term is valid when "time >= time_drag". There is no drag between gas and dust when "time < time_drag".
					  # The default value is 0.0, it means that dust-gas drags are valid from the beginning of simulations.

T_eddy                  = 1.0             # The turnover time of turbulent eddy, the default value is 1.0 .

stopping_time_n         = ......          # Constant stopping time for the nth dust species, is valid only if users do not use the "EnrollUserDustStoppingTime" in the problem file.

nu_dust_n               = ......          # Constant dust diffusivity for the nth dust species, is valid only if users do not use the "EnrollDustDiffusivity" in the problem file
					  # and "Diffusion_Flag = true" is set in the input file.

dffloor_n               = 1e-8            # The density floor for the nth dust.

drag_method             = 2nd-implicit    # drag method, "2nd-implicit","1st-implicit", "semi-implicit" and "explicit". The default option is "2nd-implicit".

Small NDUSTFLUIDS (<10):

Use 2nd-implicit/1st-implicit if stability and robustness are critical.

Consider semi-implicit or explicit for faster computations if numerical stability isn't a major concern.


Large NDUSTFLUIDS (>10):

Prioritize computational efficiency by using 1st-implicit/semi-implicit/explicit.

If you use 2nd-implicit, accepting the computational cost.


Stiff Drag Regimes (Small Stopping Time/Large Dust-Gas Ratio):

Use 2nd-implicit/1st-implicit to avoid oscillations and instabilities.


Mild Drag Regimes:

Explicit or semi-implicit methods can be used for computational speed, but ensure appropriate time-step control via EnrollUserTimeStepFunction.
```
```
The parameter of alpha disk model.
<problem>
alpha_vis                =  ......         # The alpha value in the Shakura-Sunyaev viscosity prescription, nu_gas = alpha_vis*cs^2/Omega_K.
					   # It is valid only if the name of problem file contains the string "disk".
```
