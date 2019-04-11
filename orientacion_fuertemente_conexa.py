#programa que crea una orientacion fuertemente conexa
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np




def es_puente(arista:tuple, G:nx.Graph()): # modificar
    """
    funcion que remueve arista de grafo G y mira si lo que queda es conexo
    """
    H = G.copy()
    H.remove_edge(arista[0],arista[1])
    return not nx.is_connected(H)
    #revisar la estructura de dato Graph para ver si modifica el metodo

def es_orientable(G:nx.Graph()): #modificar por nx.has_bridges(G) y luego verificar
    """
    funcion que revisa si cada arista de G es puente o no, si alguna es puente, G no es orientable
    """
    no_hay_puentes = True
    for arista in list(G.edges()):
        if es_puente(arista, G):
            no_hay_puentes = False
            break
    return no_hay_puentes



def matriz_distancia(G):
    n = len(G.nodes)
    nodes = list(G.nodes)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = nx.shortest_path_length(G,nodes[i], nodes[j])

    return D


def maxima_distancia(G):
    A = list(matriz_distancia(G))
    a = max(A)
    return a

def min_max_distancia(lista_digrafos):  # no me quiere funcinar
    """
    :param lista_digrafos: tiene las diferentes orientaciones (DiGraphs) entre las cuales se quiere la mejor
    :return: best DiGraph
    """
    distancias_maximas = {}
    for digrafo in lista_digrafos:
        distancias_maximas[digrafo] = maxima_distancia(digrafo)
    d_min = min(distancias_maximas.values())
    for digrafo in distancias_maximas.keys():
        if distancias_maximas[digrafo] == d_min:
            mejor_orientacion = digrafo


    return mejor_orientacion





###############################
#####  ideas para practicar ###
###############################

#M = nx.grid_graph(dim=[3,5]) # crea grafo de calles y carreras urbanas
#mapping = dict(zip(M.nodes(),range(1,16))) #crea diccionario keys->pares ordenados, values->enteros
#la idea en el futuro es en lugar de range(1,6) poner una lista permutada
#H = nx.relabel_nodes(M, mapping)# crea grafo renombrando nodos usando values de mapping
#aristas = nx.dfs_edges(H) # crea una lista de aristas usando el algoritmo de busqueda profunda dfs
#la idea a futuro es completar un metodo para agregar aristas dirigidas desde valores de nodos mayores hasta los menores
# crear un DiGraph usando aristas, el cual es la orientacion fuertemente conexa
# crear matriz de distancias o longitudes de caminos mas cortos
# definir sobre dicha matriz parametros, como distancia promedio o mayor distancia
# iterar todo el proceso anterior para optimizar sobre las orientaciones fuertemente conexas


if __name__ == "__main__":

    #primero veamos si k tiene un puente
    ciclos_juntos = nx.DiGraph()
    cinco_ciclos = [(1,2),(2,7),(7,8),(8,3),(3,4),(4,9),(9,10),(10,15),(15,14),(14,13),(13,12),(12,11),
    (11,6),(6,1),(3,2),(10,5),(5,4),(14,9),(13,8),(12,7),(7,6),(9,8)]
    #todo con ciclos
    todo_ciclos = nx.DiGraph()
    ocho_ciclos = [(1,10),(2,1),(9,2),(10,9),(2,3),(3,8),(8,9),(8,7),(7,4),(4,3),(4,5),
                   (5,6),(6,7),(11,10),(9,12),(12,11),(12,13),(13,8),(7,14),(14,13),(14,15),(15,6)]
    ciclos_juntos.add_edges_from(cinco_ciclos)
    todo_ciclos.add_edges_from(ocho_ciclos)
    k = nx.krackhardt_kite_graph()
    Kn = nx.complete_graph(5)
    #ahora voy a practicar orientando el grid_graph
    M = nx.grid_graph(dim=[3, 5])
    #mapping = dict(zip(M.nodes(), range(1, 16)))
    #mapping = dict(zip(M.nodes(), np.random.permutation(15))) #no me funciona el np.random.permutation, porque suelta orientaciones no fuertemente conexas
    mapping = dict(zip(M.nodes(), "qwertyuiopasdfg")) #resulta una mejor orientacion con dicho relabeling
    #mapping = dict(zip(M.nodes(), "mnbvcxzlkjhgfds")) #interesante para mostrar una region en la que hay mayores maximas distancias
    #mapping = dict(zip(M.nodes(), "mpwqerituyalskd")) #ejemplo en el que falla orientacion fuertemente conexa, debe ser que al usar codigo ASCCI falla la orientacion
    H = nx.relabel_nodes(M, mapping)
    aristas = nx.dfs_edges(H)
    arcos = list(aristas)
    orientacion = nx.DiGraph()
    orientacion.add_edges_from(arcos)
    for nodo_ini in orientacion.nodes():
        for nodo_end in orientacion.nodes():
            if (nodo_ini, nodo_end) in H.edges():
                if not ((nodo_ini, nodo_end) in arcos):
                    if not ((nodo_end, nodo_ini) in arcos):
                        orientacion.add_edge(max(nodo_ini, nodo_end), min(nodo_ini,nodo_end))

    # a continuacion pretendo mostrar varios graficos y sus diferencias


    #nx.draw_networkx(orientacion)
    #nx.draw_networkx(ciclos_juntos)
    #nx.draw_networkx(ciclos_juntos)


    #mostrar todos los graficos
    plt.subplot(221)
    nx.draw_networkx(orientacion)

    plt.subplot(222)
    nx.draw_networkx(todo_ciclos)

    plt.subplot(223)
    nx.draw_networkx(ciclos_juntos)

    plt.subplot(224)
    nx.draw_networkx(orientacion)

    plt.show()



    #print(es_puente((7,8),k))
    #print(es_orientable(Kn))

    #nx.draw_networkx(k)
    #plt.show()
    A = matriz_distancia(ciclos_juntos)
    print(A)
    print("____________________")
    B = matriz_distancia(orientacion)
    print(B)
    #h = nx.dfs_edges(k)

    #digra = nx.DiGraph()
    #for arista in h:
    #    digra.add_edge(arista[0],arista[1])

    #print(is_orientable(digra.to_undirected()))
    #print(is_orientable(nx.complete_graph(5)))


    #nx.draw_networkx(digra)
    #plt.show()




# para usar mas adelante cuando presente las centralidades
"""
import matplotlib.pyplot as plt
import networkx as nx

G = nx.krackhardt_kite_graph()

print("Betweenness")
b = nx.betweenness_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, b[v]))

print("Degree centrality")
d = nx.degree_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, d[v]))

print("Closeness centrality")
c = nx.closeness_centrality(G)
for v in G.nodes():
    print("%0.2d %5.3f" % (v, c[v]))

nx.draw(G)
plt.show()

"""


# para presentar varios graficos de las diferentes orientaciones fuertemente conexas
"""
import matplotlib.pyplot as plt
import networkx as nx

G = nx.grid_2d_graph(4, 4)  # 4x4 grid

pos = nx.spring_layout(G, iterations=100)

plt.subplot(221)
nx.draw(G, pos, font_size=8)

plt.subplot(222)
nx.draw(G, pos, node_color='k', node_size=0, with_labels=False)

plt.subplot(223)
nx.draw(G, pos, node_color='g', node_size=250, with_labels=False, width=6)

plt.subplot(224)
H = G.to_directed()
nx.draw(H, pos, node_color='b', node_size=20, with_labels=False)

plt.show()

"""
