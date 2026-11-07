"""Render actual derivative audit from a disclosed hand-set tiny network."""
import json
import sys
from pathlib import Path


def render(r):
    a=r['audit'];c=r['zero_gradient_counterexample']
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="420" viewBox="0 0 920 420" role="img" aria-label="GraphLab fixed-model input gradient audit">',
       '<rect width="920" height="420" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="34" font-size="24">GraphLab | which input coordinates reach node 0?</text>',
       '<text x="24" y="60" font-size="13">Hand-set 2 x 2 x 2 GCN, fixed graph. Class-1 minus class-0 logit, not a probability or causal effect.</text>']
    for i in range(5):p.append(f'<path d="M{80+145*i},117h145" stroke="#c0bebf" stroke-width="2"/>')
    for i,value in enumerate(a['node_contributions']):
        x=80+145*i;color='#73a89a' if i<3 else '#c0bebf'
        p += [f'<circle cx="{x}" cy="117" r="17" fill="{color}"/>',
              f'<text x="{x}" y="122" text-anchor="middle" font-size="14">{i}</text>',
              f'<text x="{x}" y="157" text-anchor="middle" font-size="12">{"target" if i==0 else str(i)+(" hop" if i==1 else " hops")}</text>',
              f'<rect x="{x-26}" y="{245-value*200:.3f}" width="52" height="{value*200:.3f}" fill="{color}"/>',
              f'<text x="{x}" y="269" text-anchor="middle" font-size="13">{value:.6f}</text>']
    p += ['<path d="M24,245H896" stroke="#eae9ea"/>',
          f'<text x="24" y="301" font-size="14">Signed coordinate contributions sum to margin {a["margin_logit"]:.12f}; residual {a["sum_residual"]:.1f}.</text>',
          '<text x="24" y="325" font-size="13">Node 2 includes a negative coordinate contribution; full JSON retains every sign. Nodes 3-5 have zero gradient.</text>',
          '<path d="M24,345H896" stroke="#eae9ea"/>',
          '<text x="24" y="372" font-size="14">Zero local gradient is not robustness: ReLU(-1) has gradient 0, but replacing -1 with +1 changes margin 0 to 1.</text>',
          '<text x="24" y="398" font-size="13">The sum identity follows this bias-free ReLU architecture and zero reference. It does not validate a causal explanation.</text>',
          '</g></svg>']
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    svg=render(json.loads(Path(sys.argv[1]).read_text()))
    dest=Path('assets/input-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==svg,'stale input-audit.svg'
    else:dest.write_text(svg)
