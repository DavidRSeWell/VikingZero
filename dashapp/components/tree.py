import igraph
import plotly.graph_objects as go

from igraph import Graph,EdgeSeq

#from vikingzero.search import MCTS

test_tree = {
    '0':[1,2],
    '1':[3,4],
    '2':[],
    '3':[],
    '4':[]
}

class TreeUI:

    def __init__(self,tree):

        assert type(tree) == dict

        (self._g,self._Xn,self._Yn,
         self._Xe,self._Ye,self._labels) = self.load_tree(tree)

    def create_edges(self,tree):

        edges = []
        for parent,children in tree.items():
            for child in children:
                edges.append((int(parent),child))

        return edges

    def load_tree(self,tree) -> tuple:

        n_vertices = len(tree.keys())

        v_label = list(map(str, range(n_vertices)))

        g = Graph(directed=True)

        g.add_vertices(n_vertices)

        g.add_edges(self.create_edges(tree))

        lay = g.layout('rt', root=(0, 0))

        position = {k: lay[k] for k in range(n_vertices)}

        Y = [lay[k][1] for k in range(n_vertices)]
        M = max(Y)

        es = EdgeSeq(g)  # sequence of edges
        E = [e.tuple for e in g.es]  # list of edges

        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2 * M - position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe += [position[edge[0]][0], position[edge[1]][0], None]
            Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

        labels = v_label

        return (g,Xn,Yn,Xe,Ye,labels)

    def display_tree(self):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self._Xe,
                                 y=self._Ye,
                                 mode='lines',
                                 line=dict(color='rgb(210,210,210)', width=1),
                                 hoverinfo='none'
                                 ))
        fig.add_trace(go.Scatter(x=self._Xn,
                                 y=self._Yn,
                                 mode='markers',
                                 name='bla',
                                 marker=dict(symbol='circle-dot',
                                             size=18,
                                             color='#6175c1',  # '#DB4551',
                                             line=dict(color='rgb(50,50,50)', width=1)
                                             ),
                                 text=self._labels,
                                 hoverinfo='text',
                                 opacity=0.8
                                 ))

        fig.show()


    def n_vertices(self) -> int:
        return len(self._g.vs)



if __name__ == '__main__':

    tree_ui = TreeUI(test_tree)

    tree_ui.display_tree()
