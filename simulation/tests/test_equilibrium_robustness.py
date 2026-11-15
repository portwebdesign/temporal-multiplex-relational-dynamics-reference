"""Closed-form branch tests for the equilibrium continuation machinery."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from equilibrium_robustness import continue_curve, newton_correct, fold_diagnostic


class EmbeddedNormalForm:
    """A known scalar fold/cusp with six independent stable coordinates."""
    nominal=1.
    def __init__(self,power):
        self.power=power
    def fun(self,y):
        return np.r_[(y[0]-.5)**self.power+y[-1]-1, -(y[1:7]-.4)]
    def state_jacobian(self,y):
        return np.diag(np.r_[self.power*(y[0]-.5)**(self.power-1),-np.ones(6)])
    def parameter_derivative(self,y):
        return np.r_[1.,np.zeros(6)]


class EquilibriumContinuationTests(unittest.TestCase):
    def settings(self):
        return {'initial_arclength_step':.02,'minimum_arclength_step':1e-6,
                'maximum_arclength_step':.03,'maximum_steps_per_sweep':500,
                'residual_tolerance':1e-11,'newton_max_iterations':30}

    def test_linear_branch_matches_closed_form(self):
        fun=lambda y:np.array([y[0]-2*y[1]])
        jac=lambda y:np.array([[1.,-2.]])
        points,record=continue_curve(fun,jac,np.array([2.,1.]),self.settings(),[.9,1.1],1)
        self.assertEqual(record['termination'],'factor_boundary_crossed')
        self.assertGreater(len(points),3)
        self.assertLess(max(abs(y[0]-2*y[1]) for y,_ in points),1e-10)

    def test_normal_form_crosses_fold_without_parameter_monotonicity(self):
        # Equilibrium y^2+lambda=0 folds at (0,0).
        fun=lambda z:np.array([z[0]**2+z[1]])
        jac=lambda z:np.array([[2*z[0],1.]])
        points,record=continue_curve(fun,jac,np.array([.5,-.25]),self.settings(),[-.5,.1],1)
        self.assertEqual(record['termination'],'factor_boundary_crossed')
        self.assertTrue(any(z[0]<-.3 for z,_ in points))
        self.assertTrue(any(a[1][-1]*b[1][-1]<0 for a,b in zip(points[:-1],points[1:])))
        self.assertLess(max(abs(z[0]**2+z[1]) for z,_ in points),1e-10)

    def test_rejects_nonphysical_trial_instead_of_clipping(self):
        fun=lambda z:np.array([z[0]-2,z[1]-1])
        jac=lambda z:np.eye(2)
        corrected,record=newton_correct(fun,jac,np.array([.5,1.]),physical_dimension=1)
        self.assertIsNone(corrected)
        self.assertFalse(record['success'])

    def fold_settings(self):
        return {'residual_tolerance':1e-9,'null_eigenvalue_tolerance':1e-7,
                'nondegeneracy_minimum_absolute':1e-7}

    def test_augmented_fold_solver_recovers_known_limit_point(self):
        family=EmbeddedNormalForm(2)
        left=np.r_[.502,np.full(6,.4),1-.002**2]
        right=np.r_[.499,np.full(6,.4),1-.001**2]
        result=fold_diagnostic(family,left,right,self.fold_settings(),[.9,1.1],.01)
        self.assertEqual(result['status'],'nondegenerate_local_fold_numerical')
        self.assertAlmostEqual(result['factor'],1,places=9)
        self.assertAlmostEqual(result['s'],.5,places=9)

    def test_degenerate_cubic_is_not_classified_as_a_simple_fold(self):
        family=EmbeddedNormalForm(3)
        left=np.r_[.501,np.full(6,.4),1-.001**3]
        right=np.r_[.499,np.full(6,.4),1+.001**3]
        result=fold_diagnostic(family,left,right,self.fold_settings(),[.9,1.1],.01)
        self.assertEqual(result['status'],'unresolved_fold_candidate')


if __name__=='__main__':
    unittest.main()
