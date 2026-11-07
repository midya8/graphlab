import copy
import unittest
from graphlab.core import Network
from graphlab.edge_audit import edge_sensitivity, edge_demo

X=[[1.,.2],[.3,1.],[-.4,.7],[.8,-.2],[.5,.4],[-.7,.6]]
E=[[0,1],[1,2],[2,3],[3,4],[4,5]]


class EdgeTests(unittest.TestCase):
    def test_full_refits_every_deletion(self):
        n=Network(2,3)
        a=edge_sensitivity(n,X,E,0,1,0)
        for i,row in enumerate(a['edges']):
            logits=n.forward(X,E[:i]+E[i+1:])[1][-1][0]
            self.assertEqual(row['deleted_margin'],logits[1]-logits[0])
        self.assertEqual(a['forwards'],6)

    def test_normalization_boundary_counterexample(self):
        r=edge_demo();a=r['audit']
        self.assertEqual(r['node_3_input_gradient'],[0.,0.])
        self.assertNotEqual(a['edges'][2]['margin_change'],0.)
        self.assertEqual(a['edges'][3]['margin_change'],0.)
        self.assertEqual(a['edges'][4]['margin_change'],0.)

    def test_budget_is_preflight_and_actual_calls(self):
        class Count(Network):
            def __init__(self):super().__init__(2);self.calls=0
            def forward(self,*a):self.calls+=1;return super().forward(*a)
        n=Count()
        with self.assertRaises(ValueError):edge_sensitivity(n,X,E,0,1,0,5)
        self.assertEqual(n.calls,0)
        r=edge_sensitivity(n,X,E,0,1,0,6)
        self.assertEqual(n.calls,r['forwards'])

    def test_no_edges_one_forward(self):
        r=edge_sensitivity(Network(2),X,[],0,1,0,1)
        self.assertEqual(r['forwards'],1);self.assertEqual(r['edges'],[])
        self.assertEqual(r['max_absolute_margin_change'],0.)

    def test_mlp_no_edge_effect(self):
        r=edge_sensitivity(Network(2,mode='mlp'),X,E,0,1,0)
        self.assertEqual(r['max_absolute_margin_change'],0.)

    def test_reversal_and_edge_orientation(self):
        n=Network(2)
        a=edge_sensitivity(n,X,E,0,1,0)
        b=edge_sensitivity(n,X,[list(reversed(e)) for e in E],0,0,1)
        for x,y in zip(a['edges'],b['edges']):
            self.assertEqual(x['edge'],y['edge'])
            self.assertAlmostEqual(x['margin_change'],-y['margin_change'])

    def test_no_mutation(self):
        n=Network(2);old=list(n.params);x,e=copy.deepcopy(X),copy.deepcopy(E)
        edge_sensitivity(n,x,e,0,1,0)
        self.assertEqual(n.params,old);self.assertEqual((x,e),(X,E))

    def test_invalid_budget_and_target(self):
        for budget in [0,True,8194]:
            with self.assertRaises(ValueError):edge_sensitivity(Network(2),X,E,0,1,0,budget)
        with self.assertRaises(ValueError):edge_sensitivity(Network(2),X,E,0,0,0)
