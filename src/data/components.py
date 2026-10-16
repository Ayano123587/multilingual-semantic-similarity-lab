"""Label-free sentence-connected groups; reuse is not itself a copied target."""
import hashlib
from collections import Counter,defaultdict
from src.data.make_dataset import normalized


def audit_components(splits):
    parent={}
    def find(key):
        parent.setdefault(key,key)
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    edges=[]
    for split,rows in splits.items():
        for row in rows:
            a,b=((row['language'],normalized(row[k])) for k in ('sentence1','sentence2'))
            ra,rb=find(a),find(b)
            if ra!=rb:parent[rb]=ra
            edges.append((split,a,b))
    members=defaultdict(list)
    for node in parent:members[find(node)].append(node)
    identifiers={root:root[0]+'-'+hashlib.sha256(min(n[1] for n in nodes).encode()).hexdigest() for root,nodes in members.items()}
    groups=defaultdict(list);counts=defaultdict(Counter);languages=defaultdict(list)
    for split,a,b in edges:
        group=identifiers[find(a)];groups[split].append(group);counts[group][split]+=1
    for root,nodes in members.items():languages[root[0]].append((identifiers[root],len(nodes)))
    report={}
    for language,items in languages.items():
        group_ids={group for group,_ in items}
        crossing=[g for g in group_ids if len(counts[g])>1]
        report[language]={'components':len(items),'sentence_nodes':sum(n for _,n in items),
                          'largest_component_sentence_nodes':max(n for _,n in items),
                          'cross_split_components':len(crossing),
                          'rows_in_cross_split_components':dict(sum((counts[g] for g in crossing),Counter())),
                          'split_component_counts':{s:len(set(g for g in groups[s] if g in group_ids)) for s in splits}}
    return dict(groups),{'languages':report,'label_access':False,
                        'interpretation':'Sentence reuse connects rows but does not prove copied pair-label leakage; official split is not sentence-disjoint.'}
