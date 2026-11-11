"""Complete finite original-edge deletion census; floating arithmetic, not proof."""
import itertools
import math
from .core import integer, matrix, number, operator, Network, artifact


def joint_sensitivity(net, features, edges, node, positive, negative,
                      max_deletions=2, max_forwards=4096):
    x=matrix(features)
    if len(x[0])!=net.inputs:raise ValueError('feature width differs from model')
    integer(node,0,len(x)-1,'target node')
    integer(positive,0,net.classes-1,'positive class')
    integer(negative,0,net.classes-1,'negative class')
    if positive==negative:raise ValueError('margin classes must differ')
    operator(len(x),edges,net.mode)
    integer(max_deletions,0,8,'max_deletions')
    integer(max_forwards,1,65536,'max_forwards')
    m=len(edges);k=min(m,max_deletions)
    required=sum(math.comb(m,j) for j in range(k+1))
    if required>max_forwards:
        raise ValueError(f'joint audit needs {required} forwards; budget is {max_forwards}')
    rows=[];single={};base=None;base_class=None
    for size in range(k+1):
        for ids in itertools.combinations(range(m),size):
            removed=set(ids)
            kept=[list(pair) for i,pair in enumerate(edges) if i not in removed]
            _,cache=net.forward(x,kept);logits=cache[-1][node]
            margin=number(logits[positive]-logits[negative],'margin')
            predicted=max(range(net.classes),key=lambda c:logits[c])
            if base is None:base,base_class=margin,predicted
            delta=number(margin-base,'margin change')
            if size==1:single[ids[0]]=delta
            additive=number(sum(single[i] for i in ids),'single effect sum')
            rows.append({'deleted_indices':list(ids),
                         'deleted_edges':[sorted(edges[i]) for i in ids],
                         'target_logits':[number(v,'logit') for v in logits],
                         'margin':margin,'margin_change':delta,
                         'singleton_sum':additive,
                         'interaction_residual':number(delta-additive,'interaction'),
                         'predicted_class':predicted,
                         'prediction_changed':predicted!=base_class})
    def arg(ids,key):return min(ids,key=lambda i:(key(rows[i]),i))
    summary=[]
    for b in range(k+1):
        ids=[i for i,r in enumerate(rows) if len(r['deleted_indices'])<=b]
        changes=[i for i in ids if rows[i]['prediction_changed']]
        summary.append({'budget':b,'sets':len(ids),
            'minimum_margin_row':arg(ids,lambda r:r['margin']),
            'maximum_margin_row':arg(ids,lambda r:-r['margin']),
            'maximum_interaction_row':arg(ids,lambda r:-abs(r['interaction_residual'])),
            'changed_predictions':len(changes),
            'first_changed_row':changes[0] if changes else None})
    return {'schema':'graphlab-joint/1','node':node,'positive_class':positive,
            'negative_class':negative,'requested_max_deletions':max_deletions,
            'effective_max_deletions':k,'edge_count':m,
            'baseline_margin':base,'baseline_class':base_class,
            'forwards':len(rows),'required_forwards':required,'max_forwards':max_forwards,
            'rows':rows,'budgets':summary,
            'scope':'all original-edge subsets up to budget; weights/features fixed; degrees recomputed; binary64 census, not causal or formal robustness proof'}


def joint_demo():
    # Discovered by enumerating tiny integer controls, not an accuracy experiment.
    n=Network(1,2);n.params=[1.,-1.,0.,1.,0.,-1.]
    x=[[-4.],[-4.],[6.],[6.]]
    e=[[0,1],[0,2],[0,3],[1,2],[1,3]]
    return {'model':artifact(n),'features':x,'edges':e,
            'audit':joint_sensitivity(n,x,e,0,1,0,2,16)}
