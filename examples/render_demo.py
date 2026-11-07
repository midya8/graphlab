"""Real fixture edges and predictions; positions are class-based schematic, not embedding."""
import json
import math
import sys
from pathlib import Path


def render(r):
    ink,gray,teal,red='#251f21','#585254','#73a89a','#ed313e'
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="670" viewBox="0 0 980 670" role="img" aria-label="GraphLab synthetic context and expressivity audit">',
       '<rect width="980" height="670" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="34" font-size="25">GraphLab | context is part of the model</text>',
       '<text x="24" y="60" font-size="13">96 synthetic nodes, 483 undirected edges. 16 gradient labels + 16 validation labels; 64 test labels withheld.</text>']
    positions=[]
    for i in range(96):
        cls,k=i//48,i%48
        radius=32+9*(k%8)
        theta=2*math.pi*(k//8)/6+.07*(k%8)
        positions.append((140+235*cls+radius*math.cos(theta),213+radius*math.sin(theta)))
    for i,j in r['graph']['edges']:
        x,y=positions[i];a,b=positions[j]
        p.append(f'<path d="M{x:.2f},{y:.2f} L{a:.2f},{b:.2f}" stroke="#c0bebf" opacity=".22"/>')
    train=set(r['fixture']['train']);test=set(r['fixture']['test']);pred=r['arms']['gcn']['prediction']['classes']
    for i,(x,y) in enumerate(positions):
        color=teal if i>=48 else gray
        wrong=i in test and pred[i]!=r['graph']['labels'][i]
        p.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" stroke="{ink if i in train else color}" stroke-width="{2 if i in train else 0}"/>')
        if wrong:p.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="none" stroke="{red}" stroke-width="1.5"/>')
    p += ['<text x="24" y="343" font-size="12">Fill = true class. Black outline = gradient label. Red ring = GCN test error.</text>',
          '<text x="24" y="364" font-size="12">Positions use known classes for a schematic, not a learned embedding.</text>',
          '<text x="544" y="108" font-size="17" font-weight="600">Selected-checkpoint test results</text>',
          '<text x="544" y="136" font-size="12">Model / selected update</text><text x="810" y="136" font-size="12">Accuracy</text><text x="890" y="136" font-size="12">CE</text>']
    for j,(name,title) in enumerate([('gcn','GCN'),('feature_mlp','Feature MLP'),('scrambled_gcn','Scrambled GCN')]):
        a=r['arms'][name];y=169+j*48
        p += [f'<text x="544" y="{y}" font-size="14">{title} / {a["training"]["selected_update"]}</text>',
              f'<text x="816" y="{y}" font-size="14">{100*a["test"]["accuracy"]:.2f}%</text>',
              f'<text x="892" y="{y}" font-size="14">{a["test"]["cross_entropy"]:.3f}</text>']
    p += [f'<text x="544" y="314" font-size="13">Train-only centroid: {100*r["centroid_control"]["test_accuracy"]:.2f}% accuracy</text>',
          '<text x="544" y="340" font-size="12">All neural arms: 72 parameters, 300-update budgets.</text>',
          '<text x="544" y="361" font-size="12">Validation CE selects independently; final checkpoints also reported.</text>',
          '<path d="M24,391 H956" stroke="#eae9ea"/>',
          '<text x="24" y="424" font-size="18">An exact limitation: a cycle is not two triangles</text>']
    for offset,edges,title in [(100,[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]],'6-cycle'),(340,[[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]],'Two triangles')]:
        xy=[(offset+48*math.cos(k*math.pi/3),497+48*math.sin(k*math.pi/3)) for k in range(6)]
        for i,j in edges:
            x,y=xy[i];a,b=xy[j];p.append(f'<path d="M{x:.2f},{y:.2f} L{a:.2f},{b:.2f}" stroke="{gray}" stroke-width="2"/>')
        for x,y in xy:p.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{gray}"/>')
        p.append(f'<text x="{offset-42}" y="571" font-size="13">{title}</text>')
    p += ['<text x="480" y="470" font-size="14">Same constant features + degree-2 regularity</text>',
          '<text x="480" y="496" font-size="14">produce identical node probabilities in this GCN.</text>',
          '<text x="480" y="522" font-size="14">Maximum difference: 0.0 on the exact fixture.</text>',
          '<text x="480" y="548" font-size="13">Permutation equivariance is not graph distinguishability.</text>',
          '<text x="24" y="620" font-size="13">Transductive development fixture: every node feature and edge is visible. This is not unseen-graph evaluation.</text>',
          '<text x="24" y="645" font-size="13">Edges were generated from class communities. No causal-component, real-data or general model-ranking claim.</text>',
          '</g></svg>']
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    svg=render(json.loads(Path(sys.argv[1]).read_text()))
    dest=Path('assets/graph-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==svg,'stale graph-audit.svg'
    else:dest.write_text(svg)
