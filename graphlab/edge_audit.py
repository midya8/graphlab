"""Finite edge-deletion effects with full normalization, not causal effects."""
from .core import integer, number, matrix, operator, Network


def edge_sensitivity(net, features, edges, node, positive, negative, max_forwards=256):
    x=matrix(features)
    integer(node,0,len(x)-1,'target node')
    integer(positive,0,net.classes-1,'positive class')
    integer(negative,0,net.classes-1,'negative class')
    if positive==negative:raise ValueError('margin classes must differ')
    operator(len(x),edges,net.mode)
    integer(max_forwards,1,8193,'max_forwards')
    required=len(edges)+1
    if required>max_forwards:
        raise ValueError(f'edge audit needs {required} forwards; budget is {max_forwards}')
    calls=0
    def score(e):
        nonlocal calls
        _,cache=net.forward(x,e);calls+=1
        row=cache[-1][node]
        return number(row[positive]-row[negative],'margin')
    base=score(edges);rows=[]
    for i,pair in enumerate(edges):
        changed=score(edges[:i]+edges[i+1:])
        rows.append({'input_edge_index':i,'edge':sorted(pair),
                     'deleted_margin':changed,'margin_change':changed-base})
    return {'node':node,'positive_class':positive,'negative_class':negative,'baseline_margin':base,
            'edges':rows,'forwards':calls,'required_forwards':required,'max_forwards':max_forwards,
            'max_absolute_margin_change':max((abs(row['margin_change']) for row in rows),default=0.),
            'scope':'single original edge deletions, weights/features fixed, all degrees recomputed; no joint-deletion or causal guarantee'}


def edge_demo():
    from .attribution import attribution_demo
    fixture=attribution_demo();net=Network(2,2);net.params=list(fixture['parameters'])
    audit=edge_sensitivity(net,fixture['features'],fixture['edges'],0,1,0)
    assert audit['edges'][2]['margin_change']!=0.
    assert audit['edges'][3]['margin_change']==0.
    return {'audit':audit,'node_3_input_gradient':fixture['audit']['input_gradient'][3],
            'reason':'edge 2-3 deletion changes node-2 degree normalization even though node-3 features are outside the two-hop input field'}
