from graphviz import Digraph

dot = Digraph(format='png')
dot.attr(rankdir='LR', size='12,3')

dot.node('X', 'Input\nX', style='filled', fillcolor='lightgray')
dot.node('P', 'Projection', style='filled', fillcolor='lightblue')
dot.node('W', 'WaveNet', style='filled', fillcolor='lightyellow')
dot.node('G', 'GAT', style='filled', fillcolor='lightgreen')
dot.node('B', 'BiLSTM', style='filled', fillcolor='orange')
dot.node('T', 'Temporal\nAttention', style='filled', fillcolor='pink')
dot.node('H', 'Output Heads\n(ŷ, ẑ)', style='filled', fillcolor='lightcoral')

dot.edge('X', 'P')
dot.edge('P', 'W')
dot.edge('W', 'G')
dot.edge('G', 'B')
dot.edge('B', 'T')
dot.edge('T', 'H')

dot.render('greenyeyes_architecture', view=False)

print("Diagram saved as greeneyes_architecture.png")