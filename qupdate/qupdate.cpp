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

#include "qupdate.hpp"
#include "../laghos_solver.hpp"

double kVectorMin(const size_t, const double*);

namespace mfem {

namespace hydrodynamics {


// *****************************************************************************
// * Smooth transition between 0 and 1 for x in [-eps, eps].
// *****************************************************************************
__host__ __device__
inline double smooth_step_01(const double x, const double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

// *****************************************************************************
// * qkernel
// *****************************************************************************
template<const int dim> static
void qkernel(const int nzones,
             const int nqp,
             const int nqp1D,
             const double gamma,
             const bool use_viscosity,
             const double h0,
             const double h1order,
             const double cfl,
             const double infinity,                           
             const double *weights,
             const double *Jacobians,
             const double *rho0DetJ0w,
             const double *e_quads,
             const double *grad_v_ext,
             const double *Jac0inv,
             double *dt_est,
             double *stressJinvT){
   GET_CONST_ADRS(weights);
   GET_CONST_ADRS(Jacobians);
   GET_CONST_ADRS(rho0DetJ0w);
   GET_CONST_ADRS(e_quads);
   GET_CONST_ADRS(grad_v_ext);
   GET_CONST_ADRS(Jac0inv);
   GET_ADRS(dt_est);
   GET_ADRS(stressJinvT);
   
   MFEM_FORALL(z, nzones,
   {
      double min_detJ = infinity;
      double Jinv[dim*dim];
      double stress[dim*dim];
      double sgrad_v[dim*dim];
      double eig_val_data[3];
      double eig_vec_data[9];
      double compr_dir[dim];
      double Jpi[dim*dim];
      double ph_dir[dim];
      double stressJiT[dim*dim];
      // ********************************************************************
      for (int q = 0; q < nqp; q++) { // this for-loop should be kernel'd too
         const int zdx = z * nqp + q;
         const double weight =  d_weights[q];
         const double inv_weight = 1. / weight;
         const double *J = d_Jacobians + zdx*dim*dim;
         const double detJ = det(dim,J);
         min_detJ = fmin(min_detJ,detJ);
         calcInverse2D(dim,J,Jinv);
         // *****************************************************************
         const double rho = inv_weight * d_rho0DetJ0w[zdx] / detJ;
         const double e   = fmax(0.0, d_e_quads[zdx]);
         const double p  = (gamma - 1.0) * rho * e;
         const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
         // *****************************************************************
         for (int k = 0; k < dim*dim;k+=1) stress[k] = 0.0;
         for (int d = 0; d < dim; d++) stress[d*dim+d] = -p;
         // *****************************************************************
         double visc_coeff = 0.0;
         if (use_viscosity)
         {
            // Compression-based length scale at the point. The first
            // eigenvector of the symmetric velocity gradient gives the
            // direction of maximal compression. This is used to define the
            // relative change of the initial length scale.
            const double *dV = d_grad_v_ext + zdx*dim*dim;
            mult(dim,dim,dim, dV, Jinv, sgrad_v);
            symmetrize(dim,sgrad_v);
            if (dim==1) {
               eig_val_data[0] = sgrad_v[0];
               eig_vec_data[0] = 1.;
            }
            else {
               calcEigenvalues(dim, sgrad_v, eig_val_data, eig_vec_data);
            }
            for(int k=0;k<dim;k+=1) compr_dir[k]=eig_vec_data[k];
            // Computes the initial->physical transformation Jacobian.
            mult(dim,dim,dim, J, d_Jac0inv+zdx*dim*dim, Jpi);
            multV(dim, dim, Jpi, compr_dir, ph_dir);
            // Change of the initial mesh size in the compression direction.
            const double h = h0 * norml2(dim,ph_dir) / norml2(dim,compr_dir);
            // Measure of maximal compression.
            const double mu = eig_val_data[0];
            visc_coeff = 2.0 * rho * h * h * fabs(mu);
            // The following represents a "smooth" version of the statement
            // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
            // eps must be scaled appropriately if a different unit system is
            // being used.
            const double eps = 1e-12;
            visc_coeff += 0.5 * rho * h * sound_speed *
                          (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
            //if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
            add(dim, dim, visc_coeff, sgrad_v, stress);
         }
         // Time step estimate at the point. Here the more relevant length
         // scale is related to the actual mesh deformation; we use the min
         // singular value of the ref->physical Jacobian. In addition, the
         // time step estimate should be aware of the presence of shocks.
         const double sv = calcSingularvalue(dim, dim-1, J);
         const double h_min = sv / h1order;
         const double inv_h_min = 1. / h_min;
         const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
         const double inv_dt = sound_speed * inv_h_min
            + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
         if (min_detJ < 0.0) {
            // This will force repetition of the step with smaller dt.
            d_dt_est[z] = 0.0;
         } else {
            const double cfl_inv_dt = cfl / inv_dt;
            d_dt_est[z] = fmin(d_dt_est[z], cfl_inv_dt);
         }
         // Quadrature data for partial assembly of the force operator.
         multABt(dim, dim, dim, stress, Jinv, stressJiT);
         for(int k=0;k<dim*dim;k+=1) stressJiT[k] *= weight * detJ;
         for (int vd = 0 ; vd < dim; vd++) {
            for (int gd = 0; gd < dim; gd++) {
               const size_t offset = zdx + nqp*nzones*(gd+vd*dim);
               d_stressJinvT[offset] = stressJiT[vd+gd*dim];
            }
         }
      }
   });
}

// *****************************************************************************
QUpdate::~QUpdate(){ }

// *****************************************************************************
QUpdate::QUpdate(const int _dim,
                 const int _nzones,
                 const int _l2dofs_cnt,
                 const int _h1dofs_cnt,
                 const bool _use_viscosity,
                 const bool _p_assembly,
                 const double _cfl,
                 const double _gamma,
                 TimingData *_timer,
                 Coefficient *_material_pcf,
                 const IntegrationRule &_ir,
                 ParFiniteElementSpace &_H1FESpace,
                 ParFiniteElementSpace &_L2FESpace):
   dim(_dim),
   nzones(_nzones),
   l2dofs_cnt(_l2dofs_cnt),
   h1dofs_cnt(_h1dofs_cnt),
   use_viscosity(_use_viscosity),
   p_assembly(_p_assembly),
   cfl(_cfl),
   gamma(_gamma),
   timer(_timer),
   material_pcf(_material_pcf),
   ir(_ir),
   H1FESpace(_H1FESpace),
   L2FESpace(_L2FESpace),
   h1_maps(mfem::kDofQuadMaps::Get(H1FESpace,ir)),
   l2_maps(mfem::kDofQuadMaps::Get(L2FESpace,ir)),
   h1_kfes(new kFiniteElementSpace(static_cast<FiniteElementSpace*>(&H1FESpace))),
   l2_kfes(new kFiniteElementSpace(static_cast<FiniteElementSpace*>(&L2FESpace))),
   d_e_quads_data(NULL),
   d_grad_x_data(NULL),
   d_grad_v_data(NULL),
   nqp(ir.GetNPoints())
{
   assert(dim==2);
   assert(p_assembly);
   assert(material_pcf);
}
   
// *****************************************************************************
// * QUpdate UpdateQuadratureData kernel
// *****************************************************************************
void QUpdate::UpdateQuadratureData(const Vector &S,
                                   bool &quad_data_is_current,
                                   QuadratureData &quad_data)
{
   // **************************************************************************
   if (quad_data_is_current) { return; }
   
   // **************************************************************************
   timer->sw_qdata.Start();
   Vector* S_p = (Vector*) &S;

   // **************************************************************************
   //const mfem::FiniteElement& fe = *H1FESpace.GetFE(0);
   //const int numDofs  = fe.GetDof();
   //const int nqp = ir.GetNPoints();
   //dbg("numDofs=%d, nqp=%d, nzones=%d",numDofs,nqp,nzones);
   const size_t H1_size = H1FESpace.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();

   // Energy dof => quads ******************************************************
   dbg("Energy dof => quads (L2FESpace)");
   ParGridFunction d_e;
   d_e.MakeRef(&L2FESpace, *S_p, 2*H1_size);
   Dof2QuadScalar(l2_kfes, L2FESpace, l2_maps, ir, d_e, &d_e_quads_data);
   
   // Coords to Jacobians ******************************************************
   dbg("Refresh Geom J, invJ & detJ");
   ParGridFunction d_x;
   d_x.MakeRef(&H1FESpace,*S_p, 0);
   Dof2QuadGrad(h1_kfes, H1FESpace, h1_maps, ir, d_x, &d_grad_x_data);
      
   // Velocity *****************************************************************
   dbg("Velocity H1_size=%d",H1_size);
   ParGridFunction d_v;
   d_v.MakeRef(&H1FESpace,*S_p, H1_size);
   Dof2QuadGrad(h1_kfes, H1FESpace, h1_maps, ir, d_v, &d_grad_v_data);

   // **************************************************************************
   const double h1order = (double) H1FESpace.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();

   // **************************************************************************
   dbg("d_dt_est");
   const size_t dt_est_sz = nzones;
   static double *d_dt_est = NULL;
   if (!d_dt_est){
      d_dt_est = (double*)mm::malloc<double>(dt_est_sz);
   }
   Vector d_dt(d_dt_est, dt_est_sz);
   d_dt = quad_data.dt_est;
   //dbg("d_dt:"); d_dt.Print(); fflush(0); //assert(false);
   
   // **************************************************************************
   dbg("qkernel");
   qkernel<2>(nzones,
              nqp,
              nqp1D,
              gamma,
              use_viscosity,
              quad_data.h0,
              h1order,
              cfl,
              infinity,
              h1_maps->quadWeights,
              d_grad_x_data,
              quad_data.rho0DetJ0w,
              d_e_quads_data,
              d_grad_v_data,
              quad_data.Jac0inv.Data(),
              d_dt_est,
              quad_data.stressJinvT.Data());
   
   // **************************************************************************
   quad_data.dt_est = kVectorMin(dt_est_sz, d_dt_est);
   dbg("dt_est=%.16e",quad_data.dt_est);
   //fflush(0); assert(false);
   
   quad_data_is_current = true;
   timer->sw_qdata.Stop();
   timer->quad_tstep += nzones;
}

} // namespace hydrodynamics

} // namespace mfem