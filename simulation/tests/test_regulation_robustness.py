"""Independent analytical and numerical controls for the robustness runner."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import NONNEGATIVE_PARAMETERS
from network import aggregate_network
from referee_experiments import raw_scenarios
from regulation_robustness import (ROOT, read_json, rank_groups, bracketed_crossings,
                                   event_solutions, program_configuration)


class RegulationRobustnessControls(unittest.TestCase):
    def test_ranking_retains_ties_without_transitive_chaining(self):
        result=rank_groups({'a':1.,'b':1.-.75e-8,'c':1.-1.5e-8,'d':.5},1e-8)
        self.assertEqual(result,[['a','b'],['c'],['d']])
        self.assertEqual(rank_groups({'a':1.,'b':.3},descending=False),[['b'],['a']])

    def test_crossings_known_polynomial_and_ties(self):
        f=lambda time:(time-12.25)*(time-17.75)
        times=[10.+i*.5 for i in range(21)]
        crossings,ties=bracketed_crossings(times,[f(t) for t in times],f)
        self.assertEqual(len(crossings),2)
        self.assertAlmostEqual(crossings[0]['time'],12.25,places=7)
        self.assertAlmostEqual(crossings[1]['time'],17.75,places=7)
        exact=lambda time:time-12.
        crossings,ties=bracketed_crossings(times,[exact(t) for t in times],exact)
        self.assertEqual(ties,[12.])
        self.assertEqual(len(crossings),1)
        self.assertAlmostEqual(crossings[0]['time'],12.,places=7)

    def test_tangent_and_initial_endpoint_not_called_crossing(self):
        times=[10.+i*.5 for i in range(11)]
        tangent=lambda time:(time-12.)**2
        crossings,ties=bracketed_crossings(times,[tangent(t) for t in times],tangent)
        self.assertEqual(crossings,[])
        self.assertEqual(ties,[12.])
        endpoint=lambda time:time-10.
        crossings,ties=bracketed_crossings(times,[endpoint(t) for t in times],endpoint)
        self.assertEqual(crossings,[])
        self.assertEqual(ties,[10.])

    def test_event_integrator_matches_piecewise_exact_load(self):
        protocol=read_json(ROOT/'configs/regulation_robustness_protocol.json')
        source=read_json(ROOT/protocol['source_protocol'])
        cfg=raw_scenarios(source)['opposed_tradeoffs']
        cfg['initial_state']=[.5,.5,.5,.45,.5,.5,.35]
        for name in NONNEGATIVE_PARAMETERS:
            cfg['parameters'][name]=0.
        cfg['parameters'].update({'psi_max':1.,'lambda_d':.2,'h_w2':.7})
        cfg=program_configuration(cfg,('delay',),protocol)
        evaluate,_=event_solutions(cfg,protocol,{'rtol':1e-11,'atol':1e-13})
        w=.35
        for index,(left,right) in enumerate([(0.,5.),(5.,10.),(10.,60.)]):
            signals=aggregate_network(cfg['network']['schedule'][0]['network'] if index==1
                                      else cfg['network']['baseline'])
            delay=.6 if index==1 else cfg['inputs']['baseline']['delay']
            gain=.2*signals.external_demand
            rate=gain+.7*delay
            equilibrium=gain/rate
            for time in [left,(left+right)/2,right]:
                exact=equilibrium+(w-equilibrium)*math.exp(-rate*(time-left))
                self.assertAlmostEqual(float(evaluate(time)[6]),exact,places=10)
                self.assertAlmostEqual(float(evaluate(time)[2]),.5,places=12)
            w=equilibrium+(w-equilibrium)*math.exp(-rate*(right-left))


if __name__=='__main__':
    unittest.main()
