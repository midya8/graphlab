import copy
import unittest
from graphlab.core import Network
from graphlab.attribution import margin_attribution, attribution_demo

X=[[1.,.2],[.3,1.],[-.4,.7],[.8,-.2],[.5,.4],[-.7,.6]]
E=[[0,1],[1,2],[2,3],[3,4],[4,5]]


class AttributionTests(unittest.TestCase):
    def test_input_finite_differences_both_modes(self):
        for mode in ['gcn','mlp']:
            n=Network(2,3,mode=mode,seed=5)
            audit=margin_attribution(n,X,E,0,1,0)
            for i in range(6):
                for j in range(2):
                    a,b=copy.deepcopy(X),copy.deepcopy(X)
                    a[i][j]+=1e-6;b[i][j]-=1e-6
                    la=n.forward(a,E)[1][-1][0];lb=n.forward(b,E)[1][-1][0]
                    fd=((la[1]-la[0])-(lb[1]-lb[0]))/2e-6
                    self.assertAlmostEqual(audit['input_gradient'][i][j],fd,places=8)

    def test_zero_baseline_sum_identity(self):
        for mode in ['gcn','mlp']:
            n=Network(2,4,mode=mode)
            for target in range(6):
                a=margin_attribution(n,X,E,target,1,0)
                self.assertAlmostEqual(a['contribution_sum'],a['margin_logit'],places=12)
                self.assertAlmostEqual(a['sum_residual'],0.,places=12)
                self.assertEqual(n.forward([[0.,0.]]*6,E)[1][-1][target],[0.,0.])

    def test_two_hop_support_exact(self):
        a=margin_attribution(Network(2),X,E,0,1,0)
        self.assertEqual(a['propagation_hops_up_to_two'],[0,1,2,None,None,None])
        self.assertEqual(a['receptive_field_nodes'],[0,1,2])
        self.assertEqual(a['input_gradient'][3:],[[0.,0.]]*3)
        x=copy.deepcopy(X);x[5]=[-100.,100.]
        b=margin_attribution(Network(2),x,E,0,1,0)
        self.assertEqual(a['margin_logit'],b['margin_logit'])

    def test_mlp_support_only_target(self):
        a=margin_attribution(Network(2,mode='mlp'),X,E,2,1,0)
        self.assertEqual(a['receptive_field_nodes'],[2])
        self.assertTrue(all(row==[0.,0.] for i,row in enumerate(a['input_gradient']) if i!=2))

    def test_permutation_equivariance(self):
        n=Network(2);perm=[3,0,5,2,1,4];xp=[None]*6
        for old,new in enumerate(perm):xp[new]=X[old]
        ep=[[perm[i],perm[j]] for i,j in E]
        a=margin_attribution(n,X,E,0,1,0);b=margin_attribution(n,xp,ep,perm[0],1,0)
        for old,new in enumerate(perm):
            for u,v in zip(a['input_gradient'][old],b['input_gradient'][new]):self.assertAlmostEqual(u,v,places=12)
        self.assertAlmostEqual(a['margin_logit'],b['margin_logit'],places=12)

    def test_margin_reversal(self):
        n=Network(2);a=margin_attribution(n,X,E,0,1,0);b=margin_attribution(n,X,E,0,0,1)
        self.assertEqual(a['margin_logit'],-b['margin_logit'])
        for x,y in zip(a['input_gradient'],b['input_gradient']):
            for u,v in zip(x,y):self.assertAlmostEqual(u,-v)

    def test_zero_gradient_does_not_mean_no_large_change(self):
        c=attribution_demo()['zero_gradient_counterexample']
        self.assertEqual(c['before_margin'],0.)
        self.assertEqual(c['before_gradient'],[[0.]])
        self.assertEqual(c['after_replacing_minus1_with_plus1_margin'],1.)

    def test_no_mutation(self):
        n=Network(2);before=list(n.params);x,e=copy.deepcopy(X),copy.deepcopy(E)
        margin_attribution(n,x,e,0,1,0)
        self.assertEqual(n.params,before);self.assertEqual((x,e),(X,E))

    def test_invalid_targets(self):
        n=Network(2)
        for node,pos,neg in [(-1,1,0),(6,1,0),(True,1,0),(0,1,1),(0,2,0),(0,1,False)]:
            with self.assertRaises(ValueError):margin_attribution(n,X,E,node,pos,neg)

    def test_zero_features_kink(self):
        a=margin_attribution(Network(2),[[0.,0.]]*6,E,0,1,0)
        self.assertEqual(a['margin_logit'],0.)
        self.assertEqual(a['sum_residual'],0.)

    def test_disconnected_support(self):
        a=margin_attribution(Network(2),X,[[0,1],[2,3],[3,4],[4,5]],0,1,0)
        self.assertEqual(a['receptive_field_nodes'],[0,1])
        self.assertEqual(a['input_gradient'][2:],[[0.,0.]]*4)
