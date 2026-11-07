"""Fixed-model node margin derivatives, not causal or edge attributions."""
import math
from .core import integer, matrix, propagate, Network


def margin_attribution(net, features, edges, node, positive, negative):
    x=matrix(features)
    integer(node,0,len(x)-1,'target node')
    integer(positive,0,net.classes-1,'positive class')
    integer(negative,0,net.classes-1,'negative class')
    if positive==negative:raise ValueError('margin classes must differ')
    _,(op,sx,pre,sh,logits)=net.forward(x,edges)
    n,h,d=len(x),net.hidden,net.inputs
    split=d*h
    dsh=[[0.]*h for _ in x]
    dsh[node]=[net.params[split+j*net.classes+positive]-net.params[split+j*net.classes+negative] for j in range(h)]
    dh=propagate(op,dsh)
    dp=[[v if pre[i][j]>0 else 0. for j,v in enumerate(row)] for i,row in enumerate(dh)]
    dsx=[[sum(dp[i][j]*net.params[k*h+j] for j in range(h)) for k in range(d)] for i in range(n)]
    gradient=propagate(op,dsx)
    contributions=[[a*b for a,b in zip(row,g)] for row,g in zip(x,gradient)]
    score=logits[node][positive]-logits[node][negative]
    total=sum(sum(row) for row in contributions)
    if not all(math.isfinite(v) for rows in [gradient,contributions] for row in rows for v in row) or not math.isfinite(score) or not math.isfinite(total):
        raise ValueError('nonfinite margin attribution')
    # Distances use the actual propagation operator: identity in MLP mode.
    distances=[None]*n;distances[node]=0;frontier={node}
    for depth in [1,2]:
        following={j for i in frontier for j,w in op[i] if distances[j] is None}
        for j in following:distances[j]=depth
        frontier=following
    return {'node':node,'positive_class':positive,'negative_class':negative,
            'margin_logit':score,'input_gradient':gradient,'gradient_times_input':contributions,
            'node_contributions':[sum(row) for row in contributions],
            'contribution_sum':total,'zero_baseline_margin':0.,'sum_residual':score-total,
            'propagation_hops_up_to_two':distances,
            'receptive_field_nodes':[i for i,v in enumerate(distances) if v is not None],
            'scope':'fixed graph and weights; local class-logit margin derivative, not probability or causal effect'}


def attribution_demo():
    x=[[1.,.2],[.3,1.],[-.4,.7],[.8,-.2],[.5,.4],[-.7,.6]]
    edges=[[0,1],[1,2],[2,3],[3,4],[4,5]]
    net=Network(2,2)
    net.params=[1.,-.5,.3,.8,.5,1.2,-.7,.2]
    audit=margin_attribution(net,x,edges,0,1,0)
    moved=[list(row) for row in x];moved[5]=[10.,-10.]
    remote=margin_attribution(net,moved,edges,0,1,0)
    dead=Network(1,1,mode='mlp');dead.params=[1.,0.,1.]
    before=margin_attribution(dead,[[-1.]],[],0,1,0)
    after=margin_attribution(dead,[[1.]],[],0,1,0)
    assert audit['margin_logit']==remote['margin_logit']
    assert max(abs(v) for row in audit['input_gradient'][3:] for v in row)==0.
    assert abs(audit['sum_residual']) < 1e-12
    return {'audit':audit,'features':x,'edges':edges,'parameters':net.params,
            'remote_node_5_replacement_margin':remote['margin_logit'],
            'zero_gradient_counterexample':{'before_margin':before['margin_logit'],
              'before_gradient':before['input_gradient'],'after_replacing_minus1_with_plus1_margin':after['margin_logit']}}
