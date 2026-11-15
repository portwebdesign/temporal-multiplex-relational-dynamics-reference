"""Independent equal-dose and exact linear-response checks for new histories."""
from copy import deepcopy
import json
from math import exp
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import simulate
from network import aggregate_network
from recovery_research import (DEFAULT_PROTOCOL, ROOT, base_configuration,
                               exposure_audit, history_configuration, scalar_value,
                               validate_histories)


class RecoveryResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text())
        cls.existing = json.loads((ROOT/cls.protocol['existing_protocol']).read_text())

    def config(self,direction='aligned',history='four_blocks'):
        base = base_configuration(self.protocol,self.existing,direction)
        base['initial_state'] = self.existing['baseline_construction']['target'][:]
        return history_configuration(base,self.protocol,history)

    def test_equal_high_duration_peak_and_both_doses(self):
        self.assertAlmostEqual(validate_histories(self.protocol),20)
        for direction in self.protocol['directions']:
            audits = [exposure_audit(self.config(direction,name)) for name in self.protocol['histories']]
            for audit in audits:
                self.assertAlmostEqual(audit['integral_total_demand'],30*.36+20*2.4)
                self.assertAlmostEqual(audit['integral_external_demand'],30*(.36/1.36)+20*(2.4/3.4))
                self.assertAlmostEqual(audit['last_high_end'],45)
                self.assertAlmostEqual(audit['peak_total_demand'],2.4)
            for key in audits[0]:
                self.assertTrue(all(abs(row[key]-audits[0][key])<1e-11 for row in audits))

    def test_mismatched_dose_is_rejected(self):
        protocol = deepcopy(self.protocol)
        protocol['histories']['one_block'][0][0] -= 1
        with self.assertRaises(ValueError):
            validate_histories(protocol)

    def test_overlapping_histories_are_rejected(self):
        protocol = deepcopy(self.protocol)
        protocol['histories']['two_blocks'] = [[5,30],[25,45]]
        with self.assertRaises(ValueError):
            validate_histories(protocol)

    def test_scalar_exact_single_step_and_common_terminal(self):
        cfg = self.config(history='one_block')
        low = .36/1.36
        high = 2.4/3.4
        for tau in (1,5,20):
            self.assertAlmostEqual(scalar_value(cfg,tau,0),low)
            self.assertAlmostEqual(scalar_value(cfg,tau,25),low)
            at_end = high+(low-high)*exp(-20/tau)
            self.assertAlmostEqual(scalar_value(cfg,tau,45),at_end)
            self.assertAlmostEqual(scalar_value(cfg,tau,50),low+(at_end-low)*exp(-5/tau))
            self.assertAlmostEqual(scalar_value(cfg,tau,2000),low)
        with self.assertRaises(ValueError):
            scalar_value(cfg,0,10)

    def test_off_grid_core_load_matches_independent_exact_solution(self):
        cfg = self.config()
        cfg['active_coordinates'] = ['r','w2']
        cfg['solver']['dt'] = .02
        rows = simulate(cfg)
        par = cfg['parameters']
        s,e,_r,c,p,_w1,w2 = cfg['initial_state']
        u = cfg['inputs']['baseline']
        loss = par['d_w2']*s*p*u['boundary']+par['h_w2']*u['delay']
        events = sorted({0,50,*[point for item in cfg['network']['schedule'] for point in (item['start'],item['end'])]})
        for left,right in zip(events,events[1:]):
            midpoint = (left+right)/2
            net = cfg['network']['baseline']
            for item in cfg['network']['schedule']:
                if item['start'] <= midpoint < item['end']:
                    net = item['network']
                    break
            dext = aggregate_network(net).external_demand
            gain = par['lambda_d']*dext+par['a_w2']*c*(1-par['q_h']*u['boundary'])+par['b_w2']*e
            reference = gain/(gain+loss)
            w2 = reference+(w2-reference)*exp(-(gain+loss)*(right-left))
        self.assertLess(abs(rows[-1]['w2']-w2),1e-10)
        for name in ('s','e','c','p','w1'):
            self.assertTrue(all(row[name] == cfg['initial_state'][('s','e','r','c','p','w1','w2').index(name)] for row in rows))


if __name__ == '__main__':
    unittest.main()
