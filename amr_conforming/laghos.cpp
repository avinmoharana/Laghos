// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
//             *** THIS IS AN AUTOMATIC MESH REFINEMENT DEMO ***
//
// Sample runs:
//    TODO update this for pumi mesh adapt
//    mpirun -np 8 laghos -p 1 -m ../data/square01_quad.mesh -rs 4 -tf 0.8 -amr
//    mpirun -np 8 laghos -p 1 -m ../data/square01_quad.mesh -rs 4 -tf 0.8 -ok 3 -ot 2 -amr
//    mpirun -np 8 laghos -p 1 -m ../data/cube01_hex.mesh -rs 3 -tf 0.6 -amr
//    mpirun -np 8 laghos -p 1 -m ../data/cube01_hex.mesh -rs 4 -tf 0.6 -rt 1e-3 -amr
//
// Test problems:
//
#include "laghos_solver.hpp"
#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifdef HAVE_SIMMETRIX
#include <SimUtil.h>
#include <MeshSim.h>
#include <SimModel.h>
#include <gmi_sim.h>
#endif
#include <PCU.h>
#include <apfMDS.h>
#include <apfConvert.h>
#include <apfShape.h>
#include <apfField.h>
#include <gmi_null.h>
#include <gmi_mesh.h>
#include <spr.h>
#include <crv.h>
#include <crvBezier.h>
#include <lionPrint.h>

// === includes for safe_mkdir ===
#include <reel.h>
#include <sys/types.h> /*required for mode_t for mkdir on some systems*/
#include <sys/stat.h> /*using POSIX mkdir call for SMB "foo/" path*/
#include <errno.h> /* for checking the error from mkdir */
// ===============================

using namespace std;
using namespace mfem;
using namespace mfem::hydrodynamics;

static double PI = 3.14159265359;

void safe_mkdir(const char* path);
double getLargetsSize(
    apf::Mesh2* m,
    apf::Field* sizes);
apf::Vector3 getPointOnEllipsoid(
    apf::Vector3 center,
    apf::Vector3 abc,
    apf::Matrix3x3 rotation,
    double scaleFactor,
    double u,
    double v);
void makeEllipsoid(
    apf::Mesh2* msf,
    apf::Mesh2* m,
    apf::Field* sizes,
    apf::Field* frames,
    apf::MeshEntity* vert,
    double scaleFactor,
    int sampleSize[2]);
void visualizeSizeField(
    apf::Mesh2* m,
    const char* sizeName,
    const char* frameName,
    int sampleSize[2],
    double userScale,
    const char* outputPrefix);

// Choice for the problem setup.
int problem = 1;

void display_banner(ostream & os);

void GetZeroBCDofs(ParMesh *pmesh, ParFiniteElementSpace *pspace,
    int bdr_attr_max, Array<int> &ess_tdofs);

int FindElementWithVertex(const Mesh* mesh, const Vertex &vert);

void GetPerElementMinMax(const GridFunction &gf,
    Vector &elem_min, Vector &elem_max,
    int int_order = -1);
void setBdryAttributes(Mesh* mesh, apf::Mesh2* pumi_mesh);
void setParBdryAttributes(ParMesh* mesh, apf::Mesh2* pumi_mesh);

void FindElementsWithVertex(const Mesh* mesh, const Vertex &vert,
    const double size, Array<int> &elements);

void changePumiMesh(ParGridFunction x_gf,
    ParMesh *pmesh,
    apf::Mesh2 *pumi_mesh,
    int order);

void getXYXandFieldValuesAtXi(apf::Mesh2 *pumi_mesh, apf::Field* f, apf::Vector3 &xi,
    std::vector<apf::Vector3> &xyz, std::vector<apf::Vector3> &fv,
    std::vector<int> &clas);

void computeSizefield(const ParFiniteElementSpace &H1FESpace,
    const ParGridFunction &grad_v,
    ParMesh *pmesh,
    vector<Vector> &mval,
    vector<Vector> &mvec);
/* void tryEdgeReshape(apf::Mesh3* pumi_mesh); */

void writePumiMesh(apf::Mesh2* mesh, const char* name, int count);

void writeMfemMesh(const ParFiniteElementSpace &H1FESpace,
    const ParFiniteElementSpace &L2FESpace,
    const BlockVector &S,
    ParGridFunction rho,
    const char* name, int count, int res);

void snapCoordinateField(apf::Mesh2* mesh, apf::Field* f);

double getMinJacobian(ParMesh* pmesh,
    ParFiniteElementSpace &h1,
    ParFiniteElementSpace &l2);

int main(int argc, char *argv[])
{
  // 1. Initialize MPI (required by PUMI).
  int num_procs, myid;
  MPI_Session mpi(argc, argv);
  myid = mpi.WorldRank();

  // 2. Parse command-line options.
  //const char *pumi_mesh_file = "../data/cube.smb";
  //const char *native_model_file = "../data/cube_nat.x_t";
  //const char *simx_model_file = "../data/cube.smd";

  const char *pumi_mesh_file = "/users/mohara/Desktop/cubeC/cube.smb";
  const char *native_model_file = "/users/mohara/Desktop/cubeC/cubeC_nat.x_t";
  const char *simx_model_file = "/users/mohara/Desktop/cubeC/cubeC_nat.smd";

  int geom_order = 2;
  int num_adapt = 400;

  // Laghos command-line options
  int rs_levels = 0;
  int rp_levels = 0;
  int order_v = geom_order;
  int order_e = 0;
  int ode_solver_type = 1;//4; //used default ;
  double t_final = 1;
  double cfl = 0.5;
  double cg_tol = 1e-8;
  int cg_max_iter = 300;
  int max_tsteps = -1;
  bool p_assembly = false;
  bool visualization = false;
  int vis_steps = 5;
  bool visit = false;
  bool gfprint = false;
  const char *basename = "results/Laghos";
  bool amr = true;
  double ref_threshold = 2e-4;
  double deref_threshold = 0.75;
  const int nc_limit = 0;
  const double blast_energy = 0.25; // 0.25 default
  const double blast_position[] = {0.0, 0.0, 0.0};
  const double blast_amr_size = 1e-10;
  double adapt_ratio = 0.06;

  OptionsParser args(argc, argv);
  args.AddOption(&pumi_mesh_file, "-pm", "--pumi-mesh",
      "Mesh file to use.");
  args.AddOption(&native_model_file, "-nm", "--native-model",
      "Native model file.");
  args.AddOption(&simx_model_file, "-sm", "--simx-model",
      "Simmetrix model file.");
  args.AddOption(&rs_levels, "-rs", "--refine-serial",
      "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&rp_levels, "-rp", "--refine-parallel",
      "Number of times to refine the mesh uniformly in parallel.");

  args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
  args.AddOption(&order_v, "-ok", "--order-kinematic",
      "Order (degree) of the kinematic finite element space.");
  args.AddOption(&order_e, "-ot", "--order-thermo",
      "Order (degree) of the thermodynamic finite element space.");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
      "ODE solver: 1 - Forward Euler,\n\t"
      "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
  args.AddOption(&t_final, "-tf", "--t-final",
      "Final time; start time is 0.");

  args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
  args.AddOption(&cg_tol, "-cgt", "--cg-tol",
      "Relative CG tolerance (velocity linear solve).");
  args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
      "Maximum number of CG iterations (velocity linear solve).");
  args.AddOption(&max_tsteps, "-ms", "--max-steps",
      "Maximum number of steps (negative means no restriction).");
  args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
      "--full-assembly",
      "Activate 1D tensor-based assembly (partial assembly).");

  args.AddOption(&amr, "-amr", "--enable-amr", "-no-amr", "--disable-amr",
      "Experimental adaptive mesh refinement (problem 1 only).");
  args.AddOption(&ref_threshold, "-rt", "--ref-threshold",
      "AMR refinement threshold.");
  args.AddOption(&deref_threshold, "-dt", "--deref-threshold",
      "AMR derefinement threshold (0 = no derefinement).");

  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
      "--no-visualization",
      "Enable or disable GLVis visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
      "Visualize every n-th timestep.");
  args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
      "Enable or disable VisIt visualization.");
  args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
      "Enable or disable result output (files in mfem format).");
  args.AddOption(&basename, "-k", "--outputfilename",
      "Name of the visit dump files");
  args.AddOption(&adapt_ratio, "-ar", "--adapt_ratio",
      "adaptation factor used in MeshAdapt");

  args.Parse();
  if (!args.Good())
  {
    if (mpi.Root()) { args.PrintUsage(cout); }
    return 1;
  }

  if (amr && problem != 1)
  {
    if (mpi.Root()) { cout << "AMR only supported for problem 1." << endl; }
    return 0;
  }

  if (mpi.Root()) { args.PrintOptions(cout); }


  // 3. Read the SCOREC mesh
  PCU_Comm_Init();
  lion_set_verbosity(1);

#ifdef HAVE_SIMMETRIX
  MS_init();
  SimModel_start();
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();
#endif

  gmi_register_null();
  gmi_register_mesh();

  apf::Mesh2* pumi_mesh;

#ifdef HAVE_SIMMETRIX
  MFEM_ASSERT(simx_model_file || native_model_file, "native or simx model file required");
  const char* model_file;
  if (simx_model_file)
    model_file = simx_model_file;
  else
    model_file = native_model_file;
  pumi_mesh = apf::loadMdsMesh(model_file, pumi_mesh_file);
#endif
  pumi_mesh = apf::loadMdsMesh(".null", pumi_mesh_file);

  // 4. Increase the geometry order and refine the mesh if necessary.  Parallel
  //    uniform refinement is performed if the total number of elements is less
  //    than 100,000.
  int dim = pumi_mesh->getDimension();
  int nEle = pumi_mesh->count(dim);
  int ref_levels = 0; // (int)floor(log(100000./nEle)/log(2.)/dim);

  if (geom_order > 1)
  {
    crv::BezierCurver bc(pumi_mesh, geom_order, 0);
    bc.run();
  }

  // Perform Uniform refinement
  if (myid == 1)
  {
    std::cout << " ref level : " <<     ref_levels << std::endl;
  }

  if (ref_levels > 1) {
    ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, ref_levels);
    if ( geom_order > 1)
      crv::adapt(uniInput);
    else
      ma::adapt(uniInput);
  }

  pumi_mesh->verify();
  pumi_mesh->acceptChanges();

  // 5. Set Boundary attributes

  ParMesh *pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);

  setParBdryAttributes(pmesh, pumi_mesh);

  for (int lev = 0; lev < rp_levels; lev++)
  {
    pmesh->UniformRefinement();
  }

  int nzones = pmesh->GetNE(), nzones_min, nzones_max;
  MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
  MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
  if (myid == 0)
  { cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl; }

  int amr_max_level = rs_levels + rp_levels;

  // Define the parallel finite element spaces. We use:
  // - H1 (Gauss-Lobatto, continuous) for position and velocity.
  // - L2 (Bernstein, discontinuous) for specific internal energy.
  L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
  H1_FECollection H1FEC(order_v, dim);
  ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
  ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
  // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
  // that the boundaries are straight.
  Array<int> ess_tdofs;
  int bdr_attr_max = pmesh->bdr_attributes.Max();
  GetZeroBCDofs(pmesh, &H1FESpace, bdr_attr_max, ess_tdofs);

  // Define the explicit ODE solver used for time integration.
  ODESolver *ode_solver = NULL;
  switch (ode_solver_type)
  {
    case 1: ode_solver = new ForwardEulerSolver; break;
    case 2: ode_solver = new RK2Solver(0.5); break;
    case 3: ode_solver = new RK3SSPSolver; break;
    case 4: ode_solver = new RK4Solver; break;
    case 6: ode_solver = new RK6Solver; break;
    default:
	    if (myid == 0)
	    {
	      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
	    }
	    delete pmesh;
	    MPI_Finalize();
	    return 3;
  }

  HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
  HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();

  if (mpi.Root())
  {
    cout << "Number of kinematic (position, velocity) dofs: "
      << glob_size_h1 << endl;
    cout << "Number of specific internal energy dofs: "
      << glob_size_l2 << endl;
  }

  int Vsize_l2 = L2FESpace.GetVSize();
  int Vsize_h1 = H1FESpace.GetVSize();

  // The monolithic BlockVector stores unknown fields as:
  // - 0 -> position
  // - 1 -> velocity
  // - 2 -> specific internal energy

  Array<int> true_offset(4);
  true_offset[0] = 0;
  true_offset[1] = true_offset[0] + Vsize_h1;
  true_offset[2] = true_offset[1] + Vsize_h1;
  true_offset[3] = true_offset[2] + Vsize_l2;
  BlockVector S(true_offset);

  // Define GridFunction objects for the position, velocity and specific
  // internal energy.  There is no function for the density, as we can always
  // compute the density values given the current mesh position, using the
  // property of pointwise mass conservation.
  ParGridFunction x_gf, v_gf, e_gf;
  x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
  v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
  e_gf.MakeRef(&L2FESpace, S, true_offset[2]);
  ParGridFunction d_gf(&H1FESpace);
  d_gf = 0.;

  // Initialize x_gf using the starting mesh coordinates. This also links the
  // mesh positions to the values in x_gf.
  pmesh->SetNodalGridFunction(&x_gf);

  // Initialize the velocity.
  VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
  v_gf.ProjectCoefficient(v_coeff);

  // Initialize density and specific internal energy values. We interpolate in
  // a non-positive basis to get the correct values at the dofs.  Then we do an
  // L2 projection to the positive basis in which we actually compute. The goal
  // is to get a high-order representation of the initial condition. Note that
  // this density is a temporary function and it will not be updated during the
  // time evolution.
  ParGridFunction rho0_gf(&L2FESpace);
  {
    FunctionCoefficient rho_coeff(hydrodynamics::rho0);
    L2_FECollection l2_fec(order_e, pmesh->Dimension());
    ParFiniteElementSpace l2_fes(pmesh, &l2_fec);

    ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
    l2_rho.ProjectCoefficient(rho_coeff);
    rho0_gf.ProjectGridFunction(l2_rho);

    if (problem == 1)
    {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1], blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
    }
    else
    {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
    }
    e_gf.ProjectGridFunction(l2_e);
  }

  // Space-dependent ideal gas coefficient over the Lagrangian mesh.
  Coefficient *material_pcf = new FunctionCoefficient(hydrodynamics::gamma);

  // Additional details, depending on the problem.
  int source = 0; bool visc = false;
  switch (problem)
  {
    case 0: if (pmesh->Dimension() == 2) { source = 1; }
	      visc = false; break;
    case 1: visc = true; break;
    case 2: visc = true; break;
    case 3: visc = true; break;
    default: MFEM_ABORT("Wrong problem specification!");
  }
  //visc = false; // added; since adapt through pumi

  LagrangianHydroOperator oper(S.Size(), H1FESpace, L2FESpace,
      ess_tdofs, rho0_gf, source, cfl, material_pcf,
      visc, p_assembly, cg_tol, cg_max_iter);

  if (amr)
  {
    // set a base for h0, this will be further divided in UpdateQuadratureData
    // TODO: for AMR, the treatment of h0 needs more work
    double elem_size = 0.5; //0.5 set default; coarse element size (TODO calculate)
    double h0 = elem_size / order_v;
    oper.SetH0(h0);
  }

  socketstream vis_rho, vis_v, vis_e, vis_u;
  char vishost[] = "localhost";
  int  visport   = 19916;

  ParGridFunction rho_gf;
  if (visualization || visit)
  {
    oper.ComputeDensity(rho_gf);
  }

  if (visualization)
  {
    // Make sure all MPI ranks have sent their 'v' solution before initiating
    // another set of GLVis connections (one from each rank):
    MPI_Barrier(pmesh->GetComm());

    int Wx = 0, Wy = 0; // window position
    const int Ww = 500, Wh = 500; // window size
    int offx = Ww+10; // window

    VisualizeField(vis_rho, vishost, visport, rho_gf,
	"Density", Wx, Wy, Ww, Wh);
    Wx += offx;
    VisualizeField(vis_v, vishost, visport, v_gf,
	"Velocity", Wx, Wy, Ww, Wh);
    Wx += offx;
    VisualizeField(vis_e, vishost, visport, e_gf,
	"Specific Internal Energy", Wx, Wy, Ww, Wh);

    Wx += offx;
    VisualizeField(vis_u, vishost, visport, d_gf,
	"Displacement", Wx, Wy, Ww, Wh);
  }

  // Save data for VisIt visualization.
  VisItDataCollection visit_dc(basename, pmesh);
  if (visit)
  {
    visit_dc.RegisterField("Density",  &rho_gf);
    visit_dc.RegisterField("Velocity", &v_gf);
    visit_dc.RegisterField("Specific Internal Energy", &e_gf);
    visit_dc.SetCycle(0);
    visit_dc.SetTime(0.0);
    visit_dc.Save();
  }

  // Perform time-integration (looping over the time iterations, ti, with a
  // time-step dt). The object oper is of type LagrangianHydroOperator that
  // defines the Mult() method that used by the time integrators.
  ode_solver->Init(oper);
  oper.ResetTimeStepEstimate();
  double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old;
  bool last_step = false;
  int steps = 0;
  BlockVector S_old(S);
  ParGridFunction dgf_old(d_gf);

  int meshwritten = 0; // Testing

  ParGridFunction x0_gf(&H1FESpace);
  x0_gf = x_gf;
  double minJdet;

  for (int ti = 1; !last_step; ti++)
  {
    if (t + dt >= t_final)
    {
      dt = t_final - t;
      last_step = true;
    }
    if (steps == max_tsteps) { last_step = true; }

    //Update displacement d = d + X_(n+1) - X_n
    //First step : d = d - X_n
    //dgf_old = d_gf;
    //d_gf -= x_gf;

    S_old = S;
    t_old = t;
    oper.ResetTimeStepEstimate();

    // S is the vector of dofs, t is the current time, and dt is the time step
    // to advance.
    cout<<" computing new soultion with dt "<<dt<<endl;
    ode_solver->Step(S, t, dt);
    //cout<<" computed new solution "<<endl;
    //steps++;

    // Adaptive time step control.
    const double dt_est = oper.GetTimeStepEstimate(S);
    if (dt_est < dt)
    {
      // Repeat (solve again) with a decreased time step - decrease of the
      // time estimate suggests appearance of oscillations.
      dt *= 0.85;
      if (dt < numeric_limits<double>::epsilon())
      { MFEM_ABORT("The time step crashed!"); }
      t = t_old;
      S = S_old;
      //d_gf = dgf_old;
      oper.ResetQuadratureData();
      if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
      ti--; continue;
    }
    else if (dt_est > 1.25 * dt) { dt *= 1.02; }

    // Make sure that the mesh corresponds to the new solution state.
    pmesh->NewNodes(x_gf, false);
    x0_gf.Update();

    if (last_step || (ti % vis_steps) == 0)
    {
      double loc_norm = e_gf * e_gf, tot_norm;
      MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
	  pmesh->GetComm());
      if (mpi.Root())
      {
	cout << fixed;
	cout << "step " << setw(5) << ti
	  << ",\tt = " << setw(5) << setprecision(4) << t
	  << ",\tdt = " << setw(5) << setprecision(6) << dt
	  << ",\t|e| = " << setprecision(10)
	  << sqrt(tot_norm) << endl;
      }

      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      if (visualization || visit || gfprint)
      {
	oper.ComputeDensity(rho_gf);
      }
      if (visualization)
      {
	int Wx = 0, Wy = 0; // window position
	int Ww = 500, Wh = 500; // window size
	int offx = Ww+10; // window offsets

	VisualizeField(vis_rho, vishost, visport, rho_gf,
	    "Density", Wx, Wy, Ww, Wh);
	Wx += offx;
	VisualizeField(vis_v, vishost, visport,
	    v_gf, "Velocity", Wx, Wy, Ww, Wh);
	Wx += offx;
	VisualizeField(vis_e, vishost, visport, e_gf,
	    "Specific Internal Energy", Wx, Wy, Ww,Wh);
	Wx += offx;

	VisualizeField(vis_u, vishost, visport, d_gf,
	    "Displacement", Wx, Wy, Ww,Wh);
	Wx += offx;
      }

      if (visit)
      {
	visit_dc.SetCycle(ti);
	visit_dc.SetTime(t);
	visit_dc.Save();
      }

      if (gfprint)
      {
	ostringstream mesh_name, rho_name, v_name, e_name;
	mesh_name << basename << "_" << ti
	  << "_mesh." << setfill('0') << setw(6) << myid;
	rho_name  << basename << "_" << ti
	  << "_rho." << setfill('0') << setw(6) << myid;
	v_name << basename << "_" << ti
	  << "_v." << setfill('0') << setw(6) << myid;
	e_name << basename << "_" << ti
	  << "_e." << setfill('0') << setw(6) << myid;

	ofstream mesh_ofs(mesh_name.str().c_str());
	mesh_ofs.precision(8);
	pmesh->Print(mesh_ofs);
	mesh_ofs.close();

	ofstream rho_ofs(rho_name.str().c_str());
	rho_ofs.precision(8);
	rho_gf.Save(rho_ofs);
	rho_ofs.close();

	ofstream v_ofs(v_name.str().c_str());
	v_ofs.precision(8);
	v_gf.Save(v_ofs);
	v_ofs.close();

	ofstream e_ofs(e_name.str().c_str());
	e_ofs.precision(8);
	e_gf.Save(e_ofs);
	e_ofs.close();
      }
    }

    oper.ComputeDensity(rho_gf);
    setParBdryAttributes(pmesh, pumi_mesh);
    oper.UpdateEssentialTrueDofs();

    ofstream mesh_ofs1("meshBeforeAdapt.mesh");
    pmesh->Print(mesh_ofs1);

    minJdet = getMinJacobian(pmesh, H1FESpace, L2FESpace);
    std::cout<<" min J det in the mesh "<<minJdet<<std::endl;

    //Vector &error_est = oper.GetZoneMaxVisc();
    //bool adapt_mesh = false;

    //for (int i = 0; i < pmesh->GetNE(); i++) {
    //  if (error_est(i) > ref_threshold )
    //  	adapt_mesh = true;
    //}
    //if (last_step) {
    if (steps % num_adapt == 0 && steps != 0) {
    //if (adapt_mesh) {
    //if (minJdet < 5*10e-3) {

      changePumiMesh(x_gf, pmesh, pumi_mesh, geom_order);
      pumi_mesh->verify();
      // Hessian of velocity computation
      ParFiniteElementSpace *fespace_scalar = new ParFiniteElementSpace(pmesh, &H1FEC, 1, Ordering::byVDIM);
      ParGridFunction v_mag(fespace_scalar);

      Array<int> dofs(3);

      for (int i = 0; i < v_mag.Size(); i++) {
	dofs.SetSize(1);
	dofs[0] = i;
	H1FESpace.DofsToVDofs(dofs);
	double v_x = v_gf(dofs[0]);
	double v_y = v_gf(dofs[1]);
	double v_z = v_gf(dofs[2]);
	(v_mag)(i) = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);
      }

      VectorCoefficient* grad_v_coeff = new GradientGridFunctionCoefficient(&v_mag);
      ParGridFunction grad_v(&H1FESpace);
      grad_v.ProjectCoefficient(*grad_v_coeff);

      // Sizefield computation
      vector<Vector> mval, mvec;
      computeSizefield(H1FESpace, grad_v, pmesh, mval, mvec);

      delete grad_v_coeff;

      writePumiMesh(pumi_mesh, "beforeAdapt", steps);

      writeMfemMesh(H1FESpace, L2FESpace, S, rho_gf,
	  "before_adapt", steps, order_v);

      // Fields on PUMI,
      // currently the mesh coordinate field after adapt is used
      // to define the coordinate gridfunction in MFEM

      apf::Field *Coords_ref = apf::createField(
	  pumi_mesh, "CoordField", apf::VECTOR, crv::getBezier(order_v));
      apf::Field *Coords_mag_field = apf::createField(
	  pumi_mesh, "Coords_mag", apf::SCALAR, crv::getBezier(order_v));

      apf::Field *vel_field = apf::createField(
	  pumi_mesh, "vel", apf::VECTOR, crv::getBezier(order_v));
      apf::Field *vel_mag_field = apf::createField(
	  pumi_mesh, "vel_mag", apf::SCALAR, crv::getBezier(order_v));

      apf::Field *e_field = apf::createField(
	  pumi_mesh, "energy", apf::SCALAR, apf::getConstant(3));
      apf::Field *e_mag_field = apf::createField(
	  pumi_mesh, "energy-mag", apf::SCALAR, apf::getConstant(3));

      apf::Field *initialCoords = apf::createField(
	  pumi_mesh, "InitialCoordinate", apf::VECTOR, crv::getBezier(order_v));
      apf::Field *initial_mag_field = apf::createField(
	  pumi_mesh, "Initial_mag", apf::SCALAR, crv::getBezier(order_v));

      apf::Field *rho_field = apf::createField(
	  pumi_mesh, "DensityField", apf::SCALAR, apf::getConstant(3));
      apf::Field *rho_mag_field = apf::createField(
	  pumi_mesh, "DensityField-mag", apf::SCALAR, apf::getConstant(3));

      cout<<pumi_mesh->countFields()<<" Fields on pumi initiated"<<endl;

      ofstream mesh_ofs2("meshInsideAdapt.mesh");
      pmesh->Print(mesh_ofs2);

      ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
      std::cout<<"parmesh casted to parpumimesh"<<std::endl;

      // populate PUMI fields from MFEM

      pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x0_gf, initialCoords, initial_mag_field);
      pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x_gf, Coords_ref, Coords_mag_field);
      pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &v_gf, vel_field, vel_mag_field);
      pPPmesh->FieldMFEMtoPUMI(pumi_mesh, &e_gf, e_field, e_mag_field);
      pPPmesh->FieldMFEMtoPUMI(pumi_mesh, &rho_gf, rho_field, rho_mag_field);

      // convert fields to Bezier

      crv::convertInterpolatingFieldToBezier(pumi_mesh, Coords_ref);
      crv::convertInterpolatingFieldToBezier(pumi_mesh, vel_field);
      crv::convertInterpolatingFieldToBezier(pumi_mesh, initialCoords);

      cout<<pumi_mesh->countFields()<<" Fields on pumi defined"<<endl;
      // define sizes and frames for mesh adapt

      apf::Field* sizes = apf::createField(pumi_mesh, "sizes", apf::VECTOR,
	  apf::getLagrange(1));
      apf::Field* frames = apf::createField(pumi_mesh, "frames", apf::MATRIX,
	  apf::getLagrange(1));

      apf::MeshEntity* ent;
      apf::MeshIterator* it = pumi_mesh->begin(0);

      int nvt = 0;
      while ((ent = pumi_mesh->iterate(it))) {
	ma::Vector s;
	ma::Matrix r;

	for (int k = 0; k < mval[nvt].Size(); k++) {
	  s[k] = mval[nvt](k);
	  for (int kk = 0; kk < mval[nvt].Size(); kk++) {
	    r[k][kk] = mvec[nvt](3*k + kk);
	  }
	}

	apf::setMatrix(frames, ent, 0, apf::transpose(r));
	apf::setVector(sizes, ent, 0, s);
	nvt++;
      }
      pumi_mesh->end(it);

      // energy values before adapt
      it = pumi_mesh->begin(3);
      double sum = 0.;
      while ((ent = pumi_mesh->iterate(it))) {
	double val = apf::getScalar(e_field, ent, 0);
	sum += val*apf::measure(pumi_mesh, ent);
	//sum += val;
      }
      pumi_mesh->end(it);
      std::cout<<" total energy values before adapt"<< sum<<std::endl;

      std::cout<<"sizes and frames set"<<std::endl;
      int sampleSize[2] = {20, 20};
      safe_mkdir("size_field");
      std::cout<<"safe mk dir"<<std::endl;
      //visualizeSizeField(pumi_mesh, "sizes", "frames", sampleSize, 0.2, "size_field");

      // 19. Perform MeshAdapt
      // setup adapt for sizes and frames
/*
      apf::Vector3 xiMid = apf::Vector3(0.5, 0, 0);
      std::vector<apf::Vector3> xyzV, fvalue;
      std::vector<int> clas;

      getXYXandFieldValuesAtXi(pumi_mesh, Coords_ref, xiMid, xyzV, fvalue, clas);
      for (size_t ij = 0; ij < xyzV.size(); ij++)
	std::cout<< xyzV[ij]-fvalue[ij]<<"  "<<clas[ij]<<std::endl;
*/
      ma::Input* erinput = ma::configure(pumi_mesh, sizes, frames, 0, true);
      //ma::Input* erinput = ma::configureIdentity(pumi_mesh, 0, 0);
      //ma::Input* erinput = ma::configureUniformRefine(pumi_mesh, 1, 0);
      erinput->shouldSnap = false;
      erinput->shouldTransferParametric = false;
      erinput->goodQuality = 0.027;
      erinput->shouldFixShape = true;
      erinput->maximumIterations = 1;
      erinput->shouldCoarsen = true;
      std::cout<<"adapt start"<<std::endl;

      if ( geom_order > 1)
	crv::adapt(erinput);
      else
	ma::adapt(erinput);

      std::cout<<"adapt end"<<std::endl;

      apf::Field* Coords = pumi_mesh->getCoordinateField();
      apf::FieldShape *fsTest = apf::getShape(Coords);
      std::string name = fsTest->getName();
      std::cout<<" name of coord field "<< name <<std::endl;

      snapCoordinateField(pumi_mesh, Coords_ref);

      // compare coordinate field values after adapt
      apf::Vector3 xiMid = apf::Vector3(0.5, 0, 0);
      std::vector<apf::Vector3> xyzV, fvalue;
      std::vector<int> clas;

      getXYXandFieldValuesAtXi(pumi_mesh, Coords_ref, xiMid, xyzV, fvalue, clas);
      for (size_t ij = 0; ij < xyzV.size(); ij++)
	std::cout<< xyzV[ij]-fvalue[ij]<<"  "<<clas[ij]<<std::endl;

      safe_mkdir("size_field_after");
      sampleSize[0] = 10;
      sampleSize[1] = 10;
      //visualizeSizeField(pumi_mesh, "sizes", "frames", sampleSize, 0.2, "size_field_after");
      //AFTER ADAPT - write vtk file

      //energy values after adapt
      it = pumi_mesh->begin(3);
      sum = 0.;
      while ((ent = pumi_mesh->iterate(it))) {
	double val = apf::getScalar(e_field, ent, 0);
	sum += val*apf::measure(pumi_mesh, ent);
	//sum += val;
      }
      pumi_mesh->end(it);
      cout<<" total energy values after adapt"<< sum<<endl;

      // MFEM ParMesh from PUMI mesh

      ParMesh* Adapmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
      std::cout<<"Reached HERE-------------00"<<std::endl;
      pPPmesh->UpdateMesh(Adapmesh);
      std::cout<<"Reached HERE-------------01"<<std::endl;

      cout<<" number of fields after adapt "<<pumi_mesh->countFields()<<endl;

      setParBdryAttributes(pmesh, pumi_mesh);
      cout<<"Reached HERE-------------01.5"<<endl;

      ofstream mesh_ofs3("meshAfterAdapt.mesh");
      pmesh->Print(mesh_ofs3);

      // 20. Update the FiniteElementSpace, Gridfunction, and bilinear form
      H1FESpace.Update();
      L2FESpace.Update();
      cout<<"Reached HERE-------------01.75"<<endl;
      //Update Essential true dofs
      //oper.UpdateEssentialTrueDofs();

      cout<<"Reached HERE-------------01.80"<<endl;

      Vsize_l2 = L2FESpace.GetVSize();
      Vsize_h1 = H1FESpace.GetVSize();
      std::cout<<"Reached HERE-------------02"<<std::endl;
      std::cout<<" H1 space size "<<Vsize_h1<<" L2 space size "<<
      	Vsize_l2<<std::endl;
      Array<int> updated_offset(4);
      updated_offset[0] = 0;
      updated_offset[1] = updated_offset[0] + Vsize_h1;
      updated_offset[2] = updated_offset[1] + Vsize_h1;
      updated_offset[3] = updated_offset[2] + Vsize_l2;
      S.Update(updated_offset);
      std::cout<<"Reached HERE-------------03"<<std::endl;
      S_old.Update(updated_offset);
      x_gf.Update();
      v_gf.Update();
      e_gf.Update();

      x0_gf.Update();
      rho_gf.Update();

      // transfer fields back to MFEM
      pPPmesh->FieldPUMItoMFEM(pumi_mesh, initialCoords, &x0_gf);
      cout<<"Reached HERE----------------03.10"<<endl;
      pPPmesh->FieldPUMItoMFEM(pumi_mesh, Coords, &x_gf);
      cout<<"Reached HERE----------------03.25"<<endl;
      pPPmesh->FieldPUMItoMFEM(pumi_mesh, vel_field, &v_gf);
      cout<<"Reached HERE----------------03.5"<<endl;
      pPPmesh->FieldPUMItoMFEM(pumi_mesh, e_field, &e_gf);
      std::cout<<"Reached HERE-------------04"<<std::endl;
      //pPPmesh->FieldPUMItoMFEM(pumi_mesh, rho_field, &rho_gf);
      std::cout<<"Reached HERE-------------05"<<std::endl;
      //Update Essential true dofs
      oper.UpdateEssentialTrueDofs();
      // Set vx, vy, vz = 0 on respective faces
      GetZeroBCDofs(pmesh, &H1FESpace, bdr_attr_max, ess_tdofs);

      // if we want to change the mesh to a user defined coordinate
      // field after adapt
      //changePumiMesh(x_gf, pmesh, pumi_mesh, geom_order);

      // update S blockvector with the new field values
      Vector x_data(x_gf.GetData(), x_gf.Size());
      Vector &block0 = S.GetBlock(0);
      block0 = x_data;

      Vector v_data(v_gf.GetData(), v_gf.Size());
      Vector &block1 = S.GetBlock(1);
      block1 = v_data;

      Vector e_data(e_gf.GetData(), e_gf.Size());
      Vector &block2 = S.GetBlock(2);
      block2 = e_data;

      oper.MeshAdaptUpdate(S, x0_gf, x_gf);

      std::cout<<"Reached HERE-------------05.5"<<std::endl;

      std::cout<<"Reached HERE-------------05.75"<<std::endl;
      //S_old.Update(updated_offset);
      std::cout<<"Reached HERE-------------06"<<std::endl;

      // Set vx, vy, vz = 0 on respective faces
      //GetZeroBCDofs(pmesh, &H1FESpace, bdr_attr_max, ess_tdofs);
      ode_solver->Init(oper);

      std::cout<<"Reached HERE-------------07"<<std::endl;

      minJdet = getMinJacobian(pmesh, H1FESpace, L2FESpace);

      std::cout<<" min J det in the mesh "<<minJdet<<std::endl;

      //Destroy fields and mesh
      delete Adapmesh;
      apf::destroyField(sizes);
      apf::destroyField(frames);
      apf::destroyField(Coords_ref);
      apf::destroyField(Coords_mag_field);
      apf::destroyField(vel_field);
      apf::destroyField(vel_mag_field);
      apf::destroyField(e_field);
      apf::destroyField(e_mag_field);

      apf::destroyField(initialCoords);
      apf::destroyField(initial_mag_field);

      apf::destroyField(rho_field);
      apf::destroyField(rho_mag_field);

      block0.Destroy();
      block1.Destroy();
      block2.Destroy();

      writePumiMesh(pumi_mesh, "afterAdapt", steps);
      writeMfemMesh(H1FESpace, L2FESpace, S, rho_gf,
	  "after_adapt", steps, order_v);
    }

    steps++;

  } // end ADAPT LOOP


  switch(ode_solver_type)
  {
    case 2: steps *= 2; break;
    case 3: steps *= 3; break;
    case 4: steps *= 4; break;
    case 6: steps *= 6;
  }
  oper.PrintTimingData(mpi.Root(), steps);

  if (visualization)
  {
    vis_v.close();
    vis_e.close();
  }


  // Free the used memory.
  delete ode_solver;
  delete pmesh;
  delete material_pcf; 

  pumi_mesh->destroyNative();
  apf::destroyMesh(pumi_mesh);

#ifdef HAVE_SIMMETRIX
  gmi_sim_stop();
  Sim_unregisterAllKeys();
  SimModel_stop();
  MS_exit();
#endif

  return 0;
}

void writePumiMesh(apf::Mesh2* mesh, const char* name, int count)
{
  std::stringstream ss;
  ss << name << count;
  crv::writeCurvedVtuFiles(mesh, apf::Mesh::TRIANGLE,
      8, ss.str().c_str());
  crv::writeCurvedWireFrame(mesh, 8, ss.str().c_str());
}

void writeMfemMesh(const ParFiniteElementSpace &H1FESpace,
    const ParFiniteElementSpace &L2FESpace,
    const BlockVector &S,
    ParGridFunction rho,
    const char* name, int count, int res)
{
  ParMesh* pmesh = H1FESpace.GetParMesh();
  ParFiniteElementSpace* h1 = (ParFiniteElementSpace*) &H1FESpace;
  ParFiniteElementSpace* l2 = (ParFiniteElementSpace*) &L2FESpace;
  BlockVector* Sptr = (BlockVector*) &S;
  ParGridFunction x, v, e;
  x.MakeRef(h1, *Sptr, 0);
  v.MakeRef(h1, *Sptr, h1->GetVSize());
  e.MakeRef(l2, *Sptr, 2 * (h1->GetVSize()));

  std::stringstream ss;
  ss << name << count<<".vtk";

  ofstream mesh_vtk_ofs(ss.str().c_str());
  pmesh->PrintVTK(mesh_vtk_ofs, res);
  x.SaveVTK(mesh_vtk_ofs, "coordinate_field", res);
  v.SaveVTK(mesh_vtk_ofs, "velocity_field", res);
  e.SaveVTK(mesh_vtk_ofs, "energy_field", res);
  rho.SaveVTK(mesh_vtk_ofs, "density_field", res);
  e.SaveVTK(mesh_vtk_ofs, "energy_order1", 1);
  rho.SaveVTK(mesh_vtk_ofs, "density_order1", 1);
}

double getMinJacobian(ParMesh* pmesh,
    ParFiniteElementSpace &h1,
    ParFiniteElementSpace &l2)
{
  IntegrationRule ir = IntRules.Get(pmesh->GetElementBaseGeometry(0),
  	3*h1.GetOrder(0) + l2.GetOrder(0) - 1);
  int nqp = ir.GetNPoints();
  double jmin = 10000;

  for( int i = 0; i < pmesh->GetNE(); i++) {
    ElementTransformation *T = h1.GetElementTransformation(i);
    for( int j = 0; j < nqp; j++) {
      IntegrationPoint &ip = ir.IntPoint(j);
      T->SetIntPoint(&ip);
      DenseMatrix J(T->Jacobian());
      double jDet = J.Det();
      if (jDet < jmin) jmin = jDet;
    }
  }
  return jmin;
}

void snapCoordinateField(apf::Mesh2* mesh, apf::Field* f)
{
  apf::FieldShape* fs = apf::getShape(f);
  apf::MeshEntity* ent;
  apf::Vector3 xi, x;

  for (int d = 1; d <= 3; d++) {
    if (!fs->hasNodesIn(d)) continue;
    apf::MeshIterator* it = mesh->begin(d);
    while ( (ent =  mesh->iterate(it)) )
    {
      if ( (mesh->getModelType(mesh->toModel(ent)) != 3) ) {

        apf::MeshElement* me = apf::createMeshElement(mesh, ent);
        int type = mesh->getType(ent);
        int non = fs->countNodesOn(type);
        for (int i = 0; i < non; i++) {
          fs->getNodeXi(type, i, xi);
          apf::mapLocalToGlobal(me, xi, x);
          apf::setVector(f, ent, i, x);
        }
        int n = fs->getEntityShape(type)->countNodes();
        apf::NewArray<double> c;
        int order = fs->getOrder();
        int td = apf::Mesh::typeDimension[type];

        crv::getBezierTransformationCoefficients(order,
            apf::Mesh::simplexTypes[td], c);
        crv::convertInterpolationFieldPoints(ent,
            f, n, non, c);
      }
    }
    mesh->end(it);
  }
}

void computeSizefield(const ParFiniteElementSpace &H1FESpace,
    const ParGridFunction &grad_v,
    ParMesh *pmesh,
    vector<Vector> &mval,
    vector<Vector> &mvec)
{
  int dim = pmesh->Dimension();
  int nv = pmesh->GetNV();
  Table *vert_elem = pmesh->GetVertexToElementTable();

  DenseMatrix Mt(dim), Mt_t(dim);
  Vector ev_s(dim), evec_s(dim*dim);

  for (int i = 0; i < nv; i++) {
    Mt_t=(0.);
    const int *el = vert_elem->GetRow(i);
    int rs = vert_elem->RowSize(i);

    DenseMatrix H(dim);
    H=(0.);
    Vector eigenVal(dim), eigenVec(dim*dim);
    DenseMatrix R(dim);
    R=(0.);
    Vector eLength(6);
    double hmin= 0., hmax = 0.;
    double err = 0.1;//0.1 used initially
    Vector ev_m(dim), ev_m_log(dim);

    for (size_t j = 0; j < rs; j++) {
      Mt=(0.);
      const FiniteElement *fe = H1FESpace.GetFE(el[j]);
      IntegrationRule intRul = fe->GetNodes();

      ElementTransformation *elemTrans = pmesh->GetElementTransformation(el[j]);
      Element *elem = pmesh->GetElement(el[j]);
      int *vertArray = elem->GetVertices();

      int ipIndex;

      for(int k = 0; k < 4; k++) {
	if (vertArray[k] == i) {
	  ipIndex = k;
	  break;
	}
      }

      IntegrationPoint ip = intRul.IntPoint(ipIndex);
      elemTrans->SetIntPoint(&ip);
      grad_v.GetVectorGradient(*elemTrans, H);

      H.Symmetrize();
      H.CalcEigenvalues(eigenVal, eigenVec);
      for (int k = 0; k < dim; k++) { // column k
	for (int kk = 0; kk < dim; kk++) {
	  R(kk, k) = eigenVec(3*k + kk);
	}
      }

      hmax = 0.5;//0.5 initially used
      hmin = hmax/25600;// initially used;

      for (int k = 0; k < 3; k++) {
	ev_m(k) = std::min(std::max(abs(eigenVal(k))/err, 1./(hmax*hmax)),
	    1./(hmin*hmin));
	ev_m_log(k) = log(ev_m(k));
      }

      MultADAt(R, ev_m_log, Mt);    // construct log(Metric) matrix for each element

      for (int k = 0; k < Mt.Width(); k++) {
	for (int kk = 0; kk < Mt.Height(); kk++) {
	  Mt_t(k, kk) = Mt_t(k, kk) + Mt(k, kk)/rs;
	}
      }

    }
    Mt_t.Symmetrize();
    Mt_t.CalcEigenvalues(ev_s, evec_s);

    for (int k = 0; k < ev_s.Size(); k++)
      ev_s(k) = 1./sqrt(exp(ev_s(k)));	// size computation

    mval.push_back(ev_s);
    mvec.push_back(evec_s);

  }
}

void getXYXandFieldValuesAtXi(apf::Mesh2 *pumi_mesh, apf::Field* f, apf::Vector3 &xi,
    std::vector<apf::Vector3> &xyz, std::vector<apf::Vector3> &fv,
    std::vector<int> &clas)
{
  int iel = 0;
  apf::MeshEntity* et;
  apf::MeshIterator* itet = pumi_mesh->begin(1);

  xi[0] = 2.*xi[0] - 1.;

  while ((et = pumi_mesh->iterate(itet))) {
    apf::Vector3 xv;
    apf::Vector3 vfv;

    apf::MeshElement* vmfe = apf::createMeshElement(pumi_mesh, et);
    apf::Element* vfe = apf::createElement(f, vmfe);

    //apf::getVector(vfe, xi, vfv);
    apf::getVector(f, et, 0, vfv);
    pumi_mesh->getPoint(et, 0, xv);
    //mapLocalToGlobal(vmfe, xi, xv);

    clas.push_back(pumi_mesh->getModelType(pumi_mesh->toModel(et)) );
    xyz.push_back(xv);
    fv.push_back(vfv);

    apf::destroyMeshElement(vmfe);
    apf::destroyElement(vfe);
    iel++;

  }
  pumi_mesh->end(itet);
}

void GetZeroBCDofs(ParMesh *pmesh, ParFiniteElementSpace *pspace,
    int bdr_attr_max, Array<int> &ess_tdofs)
{
  ess_tdofs.SetSize(0);
  Array<int> ess_bdr(bdr_attr_max), tdofs1d;
  for (int d = 0; d < pmesh->Dimension(); d++)
  {
    // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
    // enforce v_x/y/z = 0 for the velocity components.
    ess_bdr = 0; ess_bdr[d] = 1;
    pspace->GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
    ess_tdofs.Append(tdofs1d);
  }
}

// This function updates the finite element spaces based on the new number of 
// dofs after AMR. The gridfunctions associated with the solution variables
// are not updated yet.

int FindElementWithVertex(const Mesh* mesh, const Vertex &vert)
{
  Array<int> v;
  const double eps = 1e-10;

  for (int i = 0; i < mesh->GetNE(); i++)
  {
    mesh->GetElementVertices(i, v);
    for (int j = 0; j < v.Size(); j++)
    {
      double dist = 0.0;
      for (int l = 0; l < mesh->SpaceDimension(); l++)
      {
	double d = vert(l) - mesh->GetVertex(v[j])[l];
	dist += d*d;
      }
      if (dist <= eps*eps) { return i; }
    }
  }
  return -1;
}

void Pow(Vector &vec, double p)
{
  for (int i = 0; i < vec.Size(); i++)
  {
    vec(i) = std::pow(vec(i), p);
  }
}

void GetPerElementMinMax(const GridFunction &gf,
    Vector &elem_min, Vector &elem_max,
    int int_order)
{
  const FiniteElementSpace *space = gf.FESpace();
  int ne = space->GetNE();

  if (int_order < 0) { int_order = space->GetOrder(0) + 1; }

  elem_min.SetSize(ne);
  elem_max.SetSize(ne);

  Vector vals, tmp;
  for (int i = 0; i < ne; i++)
  {
    int geom = space->GetFE(i)->GetGeomType();
    const IntegrationRule &ir = IntRules.Get(geom, int_order);

    gf.GetValues(i, ir, vals);

    if (space->GetVDim() > 1)
    {
      Pow(vals, 2.0);
      for (int vd = 1; vd < space->GetVDim(); vd++)
      {
	gf.GetValues(i, ir, tmp, vd+1);
	Pow(tmp, 2.0);
	vals += tmp;
      }
      Pow(vals, 0.5);
    }

    elem_min(i) = vals.Min();
    elem_max(i) = vals.Max();
  }
}

namespace mfem
{

  namespace hydrodynamics
  {

    double rho0(const Vector &x)
    {
      switch (problem)
      {
	case 0: return 1.0;
	case 1: return 1.0;
	case 2: if (x(0) < 0.5) { return 1.0; }
		  else { return 0.1; }
	case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.0; }
		  else { return 0.125; }
	default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
      }
    }

    double gamma(const Vector &x)
    {
      switch (problem)
      {
	case 0: return 5./3.;
	case 1: return 1.4;
	case 2: return 1.4;
	case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.4; }
		  else { return 1.5; }
	default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
      }
    }

    void v0(const Vector &x, Vector &v)
    {
      switch (problem)
      {
	case 0:
	  v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
	  v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
	  if (x.Size() == 3)
	  {
	    v(0) *= cos(M_PI*x(2));
	    v(1) *= cos(M_PI*x(2));
	    v(2) = 0.0;
	  }
	  break;
	case 1: v = 0.0; break;
	case 2: v = 0.0; break;
	case 3: v = 0.0; break;
	default: MFEM_ABORT("Bad number given for problem id!");
      }
    }

    double e0(const Vector &x)
    {
      switch (problem)
      {
	case 0:
	  {
	    const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
	    double val;
	    if (x.Size() == 2)
	    {
	      val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
	    }
	    else
	    {
	      val = 
		100.0 + ((cos(2*M_PI*x(2)) + 2) *
		    (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
	    }
	    return val/denom;
	  }
	case 1: return 0.0; // This case is initialized in main().
	case 2: if (x(0) < 0.5) { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
		  else { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
	case 3: if (x(0) > 1.0) { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
		  else { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
	default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
      }
    }

  } // namespace hydrodynamics

} // namespace mfem

void display_banner(ostream & os)
{
  os << endl
    << "       __                __                 " << endl
    << "      / /   ____  ____  / /_  ____  _____   " << endl
    << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
    << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
    << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
    << "               /____/                       " << endl << endl;
}

// Set Boundary Attributes
void setBdryAttributes(Mesh* mesh, apf::Mesh2* pumi_mesh)
{
  int dim = mesh->Dimension();
  apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
  apf::MeshEntity* ent ;
  int ent_cnt = 0;
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::ModelEntity *me = pumi_mesh->toModel(ent);
    if (pumi_mesh->getModelType(me) == (dim-1))
    {
      //Get tag from model by  reverse classification
      int tag = pumi_mesh->getModelTag(me);
      if (tag == 78 || tag == 82) tag = 1;
      else if (tag == 80 || tag == 76) tag = 2;
      else if (tag == 42 || tag == 24) tag = 3;
      (mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);

      ent_cnt++;
    }
  }
  pumi_mesh->end(itr);
  //Volume faces
  itr = pumi_mesh->begin(dim);
  ent_cnt = 0;
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::ModelEntity *me = pumi_mesh->toModel(ent);
    int tag = pumi_mesh->getModelTag(me);
    mesh->SetAttribute(ent_cnt, tag);
    ent_cnt++;
  }
  pumi_mesh->end(itr);
  //Apply the attributes
  mesh->SetAttributes();
}

void setParBdryAttributes(ParMesh* mesh, apf::Mesh2* pumi_mesh)
{
  int dim = mesh->Dimension();
  apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
  apf::MeshEntity* ent ;
  int ent_cnt = 0;
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::ModelEntity *me = pumi_mesh->toModel(ent);
    if (pumi_mesh->getModelType(me) == (dim-1))
    {
      //Get tag from model by  reverse classification
      int tag = pumi_mesh->getModelTag(me);
      if (tag == 78 || tag == 82) tag = 1;
      else if (tag == 80 || tag == 76) tag = 2;
      else if (tag == 42 || tag == 24) tag = 3;  
      (mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);

      ent_cnt++;
    }
  }
  pumi_mesh->end(itr);
  //Volume 
  itr = pumi_mesh->begin(dim);
  ent_cnt = 0;
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::ModelEntity *me = pumi_mesh->toModel(ent);
    int tag = pumi_mesh->getModelTag(me);
    mesh->SetAttribute(ent_cnt, tag);
    ent_cnt++;
  }
  pumi_mesh->end(itr);
  //Apply the attributes
  mesh->SetAttributes();
}

void FindElementsWithVertex(const Mesh* mesh, const Vertex &vert,
    const double size, Array<int> &elements)
{
  Array<int> v;

  for (int i = 0; i < mesh->GetNE(); i++)
  {
    mesh->GetElementVertices(i, v);
    for (int j = 0; j < v.Size(); j++)
    {
      double dist = 0.0;
      for (int l = 0; l < mesh->SpaceDimension(); l++)
      {
	double d = vert(l) - mesh->GetVertex(v[j])[l];
	dist += d*d;
      }
      if (dist <= size*size) { elements.Append(i); break; }
    }
  }
}

/*
 * This method finds the locations (parametric xi coordinates) of the nodes (dofs)
 * on a reference element based on the order of the Finite Element Space used. 
 *
 */

void changePumiMesh(ParGridFunction x_gf,
    ParMesh *pmesh,
    apf::Mesh2 *pumi_mesh,
    int order)
{
  apf::Field *tmp_coords = apf::createField(pumi_mesh, "coord", apf::VECTOR, crv::getBezier(order));
  apf::Field *tmp_coords_mag = apf::createField(pumi_mesh, "coord_mag", apf::SCALAR, crv::getBezier(order));

  ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
  pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x_gf, tmp_coords, tmp_coords_mag);

  crv::convertInterpolatingFieldToBezier(pumi_mesh, tmp_coords);

  apf::FieldShape* fs = tmp_coords->getShape();
  int dim = pumi_mesh->getDimension();

  for(int d = 0; d <= dim; d++) {
    if (!fs->hasNodesIn(d)) continue;
    apf::MeshEntity* ent;
    apf::MeshIterator* it = pumi_mesh->begin(d);
    while( (ent = pumi_mesh->iterate(it)) ) {
      int type = pumi_mesh->getType(ent);
      int non = fs->countNodesOn(type);
      for(int n=0; n<non; n++) {
	apf::Vector3 coords;
	apf::getVector(tmp_coords, ent, n, coords);
	pumi_mesh->setPoint(ent, n, coords);
      }
    }
    pumi_mesh->end(it);
  }

  apf::destroyField(tmp_coords);
  apf::destroyField(tmp_coords_mag);

}

void safe_mkdir(const char* path)
{
  mode_t const mode = S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH;
  int err;
  errno = 0;
  err = mkdir(path, mode);
  if (err != 0 && errno != EEXIST)
  {
    reel_fail("Err: could not create directory \"%s\"\n", path);
  }
}

double getLargetsSize(
    apf::Mesh2* m,
    apf::Field* sizes)
{
  double maxSize = 0.0;
  apf::MeshEntity* vert;
  apf::MeshIterator* it = m->begin(0);
  while ( (vert = m->iterate(it)) ) {
    apf::Vector3 scales;
    apf::getVector(sizes, vert, 0, scales);
    if (scales[0] > maxSize)
      maxSize = scales[0];
    if (scales[1] > maxSize)
      maxSize = scales[1];
    if (scales[2] > maxSize)
      maxSize = scales[2];
  }
  m->end(it);
  PCU_Max_Doubles(&maxSize, 1);
  return maxSize;
}

apf::Vector3 getPointOnEllipsoid(
    apf::Vector3 center,
    apf::Vector3 abc,
    apf::Matrix3x3 rotation,
    double scaleFactor,
    double u,
    double v)
{
  apf::Vector3 result;
  result[0] = abc[0] * cos(u) * cos(v);
  result[1] = abc[1] * cos(u) * sin(v);
  result[2] = abc[2] * sin(u);

  result = result * scaleFactor;

  result = rotation * result + center;
  return result;
}

void makeEllipsoid(
    apf::Mesh2* msf,
    apf::Mesh2* mesh,
    apf::Field* sizes,
    apf::Field* frames,
    apf::MeshEntity* vert,
    double scaleFactor,
    int sampleSize[2])
{

  apf::Vector3 center;
  mesh->getPoint(vert, 0, center);

  apf::Vector3 abc;
  apf::getVector(sizes, vert, 0, abc);

  apf::Matrix3x3 rotation;
  apf::getMatrix(frames, vert, 0, rotation);


  double U0 = 0.0;
  double U1 = 2 * PI;
  double V0 = -PI/2.;
  double V1 =  PI/2.;
  int n = sampleSize[0];
  int m = sampleSize[1];
  double dU = (U1 - U0) / (n-1);
  double dV = (V1 - V0) / (m-1);

  // make the array of vertex coordinates in the physical space
  std::vector<ma::Vector> ps;
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i++) {
      double u = U0 + i * dU;
      double v = V0 + j * dV;
      apf::Vector3 pt = getPointOnEllipsoid(center, abc, rotation, scaleFactor, u, v);
      ps.push_back(pt);
    }
  }
  // make the vertexes and set the coordinates using the array
  std::vector<apf::MeshEntity*> vs;
  for (size_t i = 0; i < ps.size(); i++) {
    apf::MeshEntity* newVert = msf->createVert(0);
    msf->setPoint(newVert, 0, ps[i]);
    vs.push_back(newVert);
  }

  PCU_ALWAYS_ASSERT(vs.size() == ps.size());

  apf::MeshEntity* v[3];
  // make the lower/upper t elems
  for (int i = 0; i < n-1; i++) {
    for (int j = 0; j < m-1; j++) {
      // upper triangle
      v[0] = vs[(i + 0) + n * (j + 0)];
      v[1] = vs[(i + 0) + n * (j + 1)];
      v[2] = vs[(i + 1) + n * (j + 0)];
      apf::buildElement(msf, 0, apf::Mesh::TRIANGLE, v);
      // upper triangle
      v[0] = vs[(i + 0) + n * (j + 1)];
      v[1] = vs[(i + 1) + n * (j + 1)];
      v[2] = vs[(i + 1) + n * (j + 0)];
      apf::buildElement(msf, 0, apf::Mesh::TRIANGLE, v);
    }
  }
}


void visualizeSizeField(
    apf::Mesh2* m,
    const char* sizeName,
    const char* frameName,
    int sampleSize[2],
    double userScale,
    const char* outputPrefix)
{
  apf::Field* sizes;
  apf::Field* frames;

  char message[512];
  // first find the sizes field
  sizes  = m->findField(sizeName);
  sprintf(message, "Couldn't find a field with name %s in mesh!", sizeName);
  PCU_ALWAYS_ASSERT_VERBOSE(sizes, message);

  // then find the frames field if they exist
  frames = m->findField(frameName);
  sprintf(message, "Couldn't find a field with name %s in mesh!", frameName);
  PCU_ALWAYS_ASSERT_VERBOSE(frames, message);

  /* // remove every field except for sizes and frames */
  /* int index = 0; */
  /* while (m->countFields() > 2) { */
  /*   apf::Field* f = m->getField(index); */
  /*   if (f == sizes || f == frames) { */
  /*     index++; */
  /*     continue; */
  /*   } */
  /*   m->removeField(f); */
  /*   apf::destroyField(f); */
  /* } */

  /* m->verify(); */

  // create the size-field visualization mesh
  apf::Mesh2* msf = apf::makeEmptyMdsMesh(gmi_load(".null"), 2, false);

  apf::MeshEntity* vert;
  apf::MeshIterator* it = m->begin(0);
  int count = 0;
  while ( (vert = m->iterate(it)) ) {
    //printf("at vertex %d\n", count);
    /* if (m->isOwned(vert)) { */
    makeEllipsoid(msf, m, sizes, frames, vert, userScale , sampleSize);
    apf::Vector3 h;
    apf::Matrix3x3 r;
    apf::getVector(sizes, vert, 0, h);
    apf::getMatrix(frames, vert, 0, r);
    //std::cout << h << std::endl;
    //std::cout << r << std::endl;
    /* } */
    count++;
  }
  m->end(it);

  std::stringstream ss;
  ss << outputPrefix << "/size_field_vis";
  apf::writeVtkFiles(ss.str().c_str(), msf);
  ss.str("");

  //std::cout << "Checkpoint" << std::endl;

  msf->destroyNative();
  apf::destroyMesh(msf);
  /* m->destroyNative(); */
  /* apf::destroyMesh(m); */
}

/* void tryEdgeReshape(apf::Mesh2* pumi_mesh) */
/* { */
/*      apf::MeshEntity* ent; */ 
/*      apf::MeshIterator* it; */

/*      it = pumi_mesh->begin(3); */
/*      apf::MeshEntity* edges[6]; */

/*      int ns = 0; */ 
/*      int nf = 0; */ 

/*      while (( ent = pumi_mesh->iterate(it))) { */
/*        int ne = pumi_mesh->getDownward(ent, 1, edges); */

/*        for (int i = 0; i < ne; i++) { */
/*          if (pumi_mesh->getModelType(pumi_mesh->toModel(edges[i])) == 2) { */ 
/*            crv::CrvModelEdgeOptim *opME = new crv::CrvModelEdgeOptim(pumi_mesh, edges[i], ent); */
/*            opME->setMaxIter(100); */
/*            opME->setTol(1e-8); */

/*            if (opME->run()) ns++; */
/*            else nf++; */
/*          } */
/*          else if (pumi_mesh->getModelType(pumi_mesh->toModel(edges[i])) == 3) { */ 
/*            crv::CrvEdgeOptim *opE = new crv::CrvEdgeOptim(pumi_mesh, edges[i], ent); */
/*            opE->setMaxIter(100); */
/*            opE->setTol(1e-8); */

/*            if (opE->run()) ns++; */
/*            else nf++; */
/*          } */
/*        } */
/*      } */    
/*      pumi_mesh->end(it); */


/*   crv::writeCurvedVtuFiles(pumi_mesh, apf::Mesh::TRIANGLE, 10, "mesh_afterReshape"); */
/*   crv::writeCurvedWireFrame(pumi_mesh, 10, "mesh_afterReshape"); */
/* } */
