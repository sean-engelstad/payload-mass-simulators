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
        origin = np.array([0.0]*3 if origin is None else origin)
        #origin = [0.0]*3 if origin is None else origin

        # these are the main points in the tree
        self.tree_xpts = [origin]
        npc = self.nelem_per_comp
        # print(f"{self.tree_xpts}")
        # this is the mesh we return to the user
        #xpts = [0.0]*3 if origin is None else origin # first point at origin
        xpts = list(origin.copy())

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
            # print(f"{self.tree_xpts}")
            # loop over all mesh nodes between the two main tree nodes here
            for ielem in range(npc):
                frac = (ielem+1) * 1.0 / npc
                xpts += list(start_xpt + dxpt * frac)
        return np.array(xpts)
    
    def get_centroid(self, x, origin=None):
        #rho = self.material.rho
        rho = 1
        lengths = np.array([x[3*icomp] for icomp in range(self.ncomp)])
        Varray = np.array([x[3*icomp+2]*x[3*icomp+1]*x[3*icomp] for icomp in range(self.ncomp)])

        lengths_0 = lengths.copy()
        self.get_xpts(lengths_0)
        nodal_xpts_0 = self.tree_xpts.copy()
        #print(f"nodal xpts 0: {nodal_xpts_0}")

        sum_Mi = 0
        sum_MiCij = np.zeros(shape = (3,))
        for icomp in range(self.ncomp):
            sum_Mi += rho*Varray[icomp]

            start_tree_node = self.tree_start_nodes[icomp]
            start_xpt_0 = np.array(nodal_xpts_0[start_tree_node])
            # print(f"{nodal_xpts_0=}")
            # print(f"{start_xpt_0.shape=}")
            direction = self.tree_directions[icomp]%3
            Cij = start_xpt_0.copy()
            #print(f"Cij: {Cij}")
            # print(f"{Cij.shape=}")
            Cij[direction] += lengths[icomp]/2
            #print(f"Cij: {Cij}")
            sum_MiCij += (rho*Varray[icomp]*Cij)
        Xj = (sum_MiCij)/sum_Mi
        return Xj

    def get_centroid_gradient(self, x, origin=None):
        h = 1e-5
        #rho = self.material.rho
        rho = 1
        lengths = np.array([x[3*icomp] for icomp in range(self.ncomp)])
        t1s = np.array([x[3*icomp+1] for icomp in range(self.ncomp)])
        t2s = np.array([x[3*icomp+2] for icomp in range(self.ncomp)])
        Varray = lengths * t1s * t2s
        # print(f"{Varray=}")
        # print(f"{lengths=}")

        lengths_0 = lengths.copy()
        self.get_xpts(lengths_0)
        nodal_xpts_0 = self.tree_xpts.copy()
        
        dXj_gradient = np.zeros((3, 3*self.ncomp)) 
        for kcomp in range(self.ncomp):
            for dim in range(3):  # 0 = length, 1 = t1, 2 = t2
                #Initialize summations
                sum_Mi = 0
                sum_dv = np.zeros(shape = (3,))
                sum_v = np.zeros(shape = (3,))
                sum_du = 0
                for icomp in range(self.ncomp):
                    start_tree_node = self.tree_start_nodes[icomp]
                    start_xpt_0 = np.array(nodal_xpts_0[start_tree_node])
                    direction = self.tree_directions[icomp]%3

                    L = x[3*icomp]  
                    t1 = x[3*icomp+1]
                    t2 = x[3*icomp+2]
                    V = L * t1 * t2
                    Mi = rho * V

                    if icomp == kcomp:
                            if dim == 0: #d/dL
                                dMi_dlk = rho*t1*t2
                                dCij_dlk = np.zeros(shape = (3,))
                                dCij_dlk[direction] = 0.5
                                # print(f"{icomp=}, {kcomp=}, {dCij_dlk=}")
                            elif dim == 1: #d/dt1
                                dMi_dlk = rho*L*t2
                                dCij_dlk = np.zeros(shape = (3,))
                            elif dim == 2: #d/dt2
                                dMi_dlk = rho*L*t1
                                dCij_dlk = np.zeros(shape = (3,))
                    else:
                        if dim == 0:
                            dMi_dlk = 0
                            p_vec = np.zeros((self.ncomp,))
                            p_vec[kcomp] = 1
                            self.get_xpts(lengths_0+p_vec*h)
                            nodal_xpts_p = self.tree_xpts.copy()
                            start_xpt_p = np.array(nodal_xpts_p[start_tree_node])
                            dCij_dlk = (start_xpt_p-start_xpt_0)/h
                        else:
                            dMi_dlk = 0
                            dCij_dlk = np.zeros(shape = (3,))

                    Cij = start_xpt_0.copy()
                    Cij[direction] += L/2 
                    sum_Mi += Mi
                    sum_dv += dMi_dlk*Cij+dCij_dlk*Mi
                    sum_v += Mi*Cij
                    sum_du += dMi_dlk
                # if dim == 1:
                    # print(f"{sum_Mi = }, {sum_dv = }, {sum_v = }, {sum_du = }")
                # numerator_1 = sum_Mi*sum_dv
                # numerator_2 = sum_v*sum_du
                # print(f"{numerator_1=}, {numerator_2=}")
                dXj_gradient[:, 3*kcomp+dim] = (sum_Mi*sum_dv-sum_v*sum_du)/(sum_Mi**2) #/dl
                # dXj_gradient[:, 3*kcomp+1] = 0                                      #/t1
                # dXj_gradient[:, 3*kcomp+2] = 0                                      #/t2
        # print(f"{dXj_gradient=}")

            # Derivative of length and local centroid in same componen
            # dcdx_grad[:, 3*icomp] = (1/2*x[3*icomp+1]*x[3*icomp+2]*Varray[icomp] - centroid_0*Varray[icomp]*x[3*icomp+1]*x[3*icomp+2])/(Varray[icomp])**2 # dC/dl
            # dcdx_grad[:, 3*icomp+1] = (centroid_0*x[3*icomp]*x[3*icomp+2]*Varray[icomp] - centroid_0*Varray[icomp]*x[3*icomp]*x[3*icomp+2])/(Varray[icomp])**2 # dC/dt1
            # dcdx_grad[:, 3*icomp+2] = (centroid_0*x[3*icomp]*x[3*icomp+2]*Varray[icomp] - centroid_0*Varray[icomp]*x[3*icomp]*x[3*icomp+2])/(Varray[icomp])**2 # dC/dt2
            # FD_val = (centroid_p - centroid_0) / h
            # HC_val = np.dot(p_vec, centroid_0)
            # print(f"Centroid gradient FD test: {FD_val=} {HC_val=}")
        return dXj_gradient

    def centroid_FD_test(self, x, h=1e-3):
        p_vec = np.random.rand(x.shape[0])
        # p_vec = np.array([1] + [0]*(x.shape[0]-1))
        # p_vec = np.array([0, 1, 0, 0, 0, 0])
        centroid_0 = self.get_centroid(x)
        centroid_grad = self.get_centroid_gradient(x)
        centroid_1 = self.get_centroid(x + p_vec * h)
        FD_val = (centroid_1 - centroid_0) / h
        HC_val = np.dot(centroid_grad, p_vec)
        print(f"Centroid FD test: {FD_val=} {HC_val=}")
        return

    # def centroid_CS_test(self, x, h=1e-30):
    #TODO: Add functionality to get_centroid to support complex numbers
    #     #p_vec = np.random.rand(x.shape[0])
    #     p_vec = np.array([1] + [0]*(x.shape[0]-1))
    #     # p_vec = np.array([0, 0, 0, 1, 0, 0])
    #     centroid_0 = self.get_centroid(x)
    #     centroid_grad = self.get_centroid_gradient(x)
    #     centroid_1 = self.get_centroid(x.astype(complex) + 1j*p_vec * h)
    #     CS_val = np.imag(centroid_1) / h
    #     CS_val = CS_val.imag
    #     HC_val = np.dot(centroid_grad, p_vec)
    #     print(f"Centroid CS test: {CS_val=} {HC_val=}")
    #     return

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