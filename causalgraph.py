def genCausalGraph(collisions,num):
    edgelist = []
    for p in range(num):
        collsp = []

        for i in range(len(collisions)):
            if collisions[i][0] == p or collisions[i][1] == p:
                collsp.append(i)

        if len(collsp) > 0:
            for i in range(len(collsp)-1):
                edgelist.append([collsp[i], collsp[i+1]])

    return edgelist