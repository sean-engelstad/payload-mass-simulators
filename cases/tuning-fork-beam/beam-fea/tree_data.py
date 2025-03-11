import numpy as np, matplotlib.pyplot as plt

class TreeData:
    def __init__(
            self, 
            tree_start_nodes:list, 
            tree_directions:list,
            nelem_per_comp:int=5
        ):
        assert len(tree_start_nodes) == len(tree_directions)
        self.tree_start_nodes = tree_start_nodes
        self.tree_directions = tree_directions
        self.nelem_per_comp = nelem_per_comp

        self.tree_xpts = None
        self.elem_conn = None
        self.elem_comp = None

        self.get_mesh_symbolic()

    @property
    def ncomp(self) -> int:
        """number of components of the tree structure"""
        return len(self.tree_start_nodes)
    
    @property
    def nelems(self) -> int:
        return self.nelem_per_comp * self.ncomp
    
    @property
    def nnodes(self) -> int:
        return self.nelems + 1

    def get_mesh_symbolic(self):
        """get the element connectivity of the tree all symbolic data"""
        elem_conn = []
        elem_comp = []
        npc = self.nelem_per_comp

        for icomp in range(self.ncomp):
            # tree nodes are just main points of tree, mesh nodes include extra nodes
            # in each branch of tree
            start_tree_node = self.tree_start_nodes[icomp]

            # loop over all mesh nodes between the two main tree nodes here
            for ielem in range(self.nelem_per_comp):
                elem_comp += [icomp]
                if ielem == 0:
                    elem_conn += [[start_tree_node*npc, 
                                   icomp*npc + ielem+1]]
                else:
                    elem_conn += [[icomp*npc + ielem,
                                   icomp*npc + ielem+1]]
                    
        self.elem_conn = elem_conn
        self.elem_comp = elem_comp

        # print(f"{self.elem_conn=} {self.elem_comp=}")

        return elem_conn, elem_comp

    def get_xpts(self, lengths, origin=None):
        origin = [0.0]*3 if origin is None else origin

        # these are the main points in the tree
        self.tree_xpts = [origin]
        npc = self.nelem_per_comp
        
        # this is the mesh we return to the user
        xpts = origin # first point at origin

        for icomp in range(self.ncomp):
            start_tree_node = self.tree_start_nodes[icomp]
            start_xpt = np.array(self.tree_xpts[start_tree_node])
            _direc = self.tree_directions[icomp]
            direc = _direc // 2 # 0-5 to 0,1,2
            sign = (_direc % 2) * 2 - 1

            # next node on the tree
            dxpt = np.zeros((3,))
            dxpt[direc] = sign * lengths[icomp]
            end_xpt = start_xpt + dxpt
            self.tree_xpts += [end_xpt]

            # loop over all mesh nodes between the two main tree nodes here
            for ielem in range(npc):
                frac = (ielem+1) * 1.0 / npc
                xpts += list(start_xpt + dxpt * frac)
        return np.array(xpts)
    
    def plot_mesh(self, xpts, filename="mesh.png"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for ielem in range(len(self.elem_conn)):
            nodes = self.elem_conn[ielem]
            xpt1 = xpts[3*nodes[0]:3*nodes[0]+3]
            xpt2 = xpts[3*nodes[1]:3*nodes[1]+3]
            xv = [xpt1[0], xpt2[0]]
            yv = [xpt1[1], xpt2[1]]
            zv = [xpt1[2], xpt2[2]]
            plt.plot(xv, yv, zv, linewidth=2)
        # plt.show()
        plt.savefig("_modal/" + filename, dpi=400)
        # exit()

    def plot_mesh_compare(self, xpts1, xpts2, filename="mesh-compare.png"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for ielem in range(len(self.elem_conn)):
            nodes = self.elem_conn[ielem]
            for i, xpts in enumerate([xpts1, xpts2]):
                xpt1 = xpts[3*nodes[0]:3*nodes[0]+3]
                xpt2 = xpts[3*nodes[1]:3*nodes[1]+3]
                xv = [xpt1[0], xpt2[0]]
                yv = [xpt1[1], xpt2[1]]
                zv = [xpt1[2], xpt2[2]]
            plt.plot(xv, yv, zv, linewidth=2)
        # plt.show()
        plt.savefig("_modal/" + filename, dpi=400)
        # exit()