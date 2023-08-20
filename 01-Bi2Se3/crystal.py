import numpy as np
from progress import ProgressBar
import matplotlib.pyplot as plt

class crystal():
    def __init__(self):
        # Bi2Se3
        # 1. Geometry
        # (1) lattice
        self.n_orbital = 4
        self.n_atom = 5
        self.n_dim = 2 * self.n_atom * self.n_orbital
        self.a = 4.114  # lattice parameter (in A)
        self.h1 = np.sqrt(3.030 ** 2 - self.a **2 / 3)
        self.h2 = np.sqrt(2.926 ** 2 - self.a **2 / 3)
        self.h3 = np.sqrt(3.354 ** 2 - self.a **2 / 3)
        self.a1 = np.array([self.a / 2, self.a * np.sqrt(3) / 6, 2 * (self.h1 + self.h2) + self.h3])
        self.a2 = np.array([-self.a / 2, self.a * np.sqrt(3) / 6, 2 * (self.h1 + self.h2) + self.h3])
        self.a3 = np.array([0, -self.a * np.sqrt(3) / 3, 2 * (self.h1 + self.h2) + self.h3])
        self.v = np.dot(self.a1,np.cross(self.a2, self.a3)) # volume
        self.b1 = 2 * np.pi * np.cross(self.a2, self.a3) / self.v
        self.b2 = 2 * np.pi * np.cross(self.a3, self.a1) / self.v
        self.b3 = 2 * np.pi * np.cross(self.a1, self.a2) / self.v
        self.n_layer = 2

        # (2) neighbor
        self.neighbour11 = self.a*np.array([[1, 0, 0],[1/2, np.sqrt(3)/2, 0],[-1/2, np.sqrt(3)/2, 0],[-1, 0, 0],[-1/2, -np.sqrt(3)/2, 0],[1/2, -np.sqrt(3)/2, 0]])
        self.neighbour15 = np.array([ [0, -np.sqrt(3)* self.a/3, -self.h1],[ self.a/2, np.sqrt(3)*self.a/6, -self.h1],[ -self.a/2, np.sqrt(3)*self.a/6, -self.h1 ]])
        self.neighbour13 = np.array([[ 0,np.sqrt(3)*self.a/3, self.h2],[ -self.a/2, -np.sqrt(3)*self.a/6, self.h2],[ self.a/2, -np.sqrt(3)*self.a/6 ,self.h2 ]])
        self.neighbour34 = np.array([[ 0,np.sqrt(3)*self.a/3, self.h3],[ -self.a/2, -np.sqrt(3)*self.a/6, self.h3],[ self.a/2, -np.sqrt(3)*self.a/6, self.h3 ]])
        self.neighbour35 = np.array([[ 0,np.sqrt(3)*self.a/3, -self.h1-self.h2],[ -self.a/2, -np.sqrt(3)*self.a/6 ,-self.h1-self.h2],[ self.a/2, -np.sqrt(3)*self.a/6 ,-self.h1-self.h2 ]])
        self.neighbour14 = np.array([[ 0, -np.sqrt(3)*self.a/3 ,self.h3+self.h2],[ -self.a/2, np.sqrt(3)*self.a/6, self.h3+self.h2],[ self.a/2,np.sqrt(3)*self.a/6, self.h3+self.h2 ]])
        self.neighbour12 = np.array([[ 0,np.sqrt(3)*self.a/3, -2*self.h1],[ -self.a/2, -np.sqrt(3)*self.a/6, -2*self.h1],[ self.a/2, -np.sqrt(3)*self.a/6 ,-2*self.h1 ]])

        # 2. K-path
        self.Z = np.array([0.5,0.5,0.5])
        self.F = np.array([0.5,0.5,0.0])
        self.G = np.array([0.0,0.0,0.0])
        self.L = np.array([0.5,0.0,0.0])

        # self.kpath = np.array([self.G,self.Z,self.F,self.G,self.L]) @ np.array([self.b1, self.b2, self.b3])
        self.kpath = np.array([ self.F, self.G, self.L]) @ np.array([self.b1, self.b2, self.b3])
        self.n_kpath = self.kpath.shape[0]
        self.dk = 300 # this is number of points between two high symmetry points
        self.k = np.zeros(((self.n_kpath - 1)* self.dk, 3))
        self.x = np.zeros((self.n_kpath-1) * self.dk)
        self.k_label = np.zeros(self.n_kpath)
        self.n_kpoints = self.k.shape[0]

        for i in range(self.n_kpath - 1):
            self.k[i*self.dk:(i+1)*self.dk,:] = np.linspace(self.kpath[i,:], self.kpath[i+1,:] - (self.kpath[i+1,:] - self.kpath[i,:])/self.dk, self.dk)
            # self.k[i * self.dk:(i + 1) * self.dk, :] = np.linspace(self.kpath[i, :], self.kpath[i + 1, :] , self.dk)
            self.k_label[i + 1] =  self.k_label[i] + np.linalg.norm(self.kpath[i+1] - self.kpath[i])
            # self.x[i*self.dk:(i+1)*self.dk] = np.linspace(self.k_label[i], self.k_label[i+1], self.dk)
            self.x[i * self.dk:(i + 1) * self.dk] = np.linspace(self.k_label[i], self.k_label[i + 1] - (self.k_label[i + 1] - self.k_label[i])/self.dk, self.dk)

        # 3. Tight Binding model Parameter from DFT
        parameter_path = './parameter/'
        self.delta_so = np.loadtxt(parameter_path + 'delta_so.dat')
        self.delta = self.delta_so / 3
        self.es = np.loadtxt(parameter_path +'es.dat')
        self.ep = np.loadtxt(parameter_path +'ep.dat')
        self.Vss_sigma = np.loadtxt(parameter_path + 'Vss_sigma.dat')
        self.Vsp_sigma = np.loadtxt(parameter_path + 'Vsp_sigma.dat')
        self.Vps_sigma = np.loadtxt(parameter_path + 'Vps_sigma.dat')
        self.Vpp_sigma = np.loadtxt(parameter_path + 'Vpp_sigma.dat')
        self.Vpp_pi = np.loadtxt(parameter_path + 'Vpp_pi.dat')
        print('all parameters have been loaded')

class Hamiltonian(crystal):
    def __init__(self):
        super(Hamiltonian,self).__init__() # get crystal parameter from crystal()

        # Total Hamiltonian Frame (to be diagonalized) n_layer(*n_dim) * n_layer(*n_dim)
        self.H_QL10 = np.eye(self.n_layer, k=-1)
        self.H_QL01 = np.eye(self.n_layer, k=1)
        self.H_QL00 = np.eye(self.n_layer)

        # Bulk-Surface interaction
        # self.intraction = 0.
        # self.H_QL_perturb_upper = np.eye(self.n_layer, k=self.n_layer//2)
        # self.H_QL_perturb_lower = np.eye(self.n_layer, k=-self.n_layer//2)

        # SOC Hamiltonian 40*40
        self.Hsoc_uu = np.array([[0, 0, 0,  0],
                                 [0, 0,-1j, 0],
                                 [0, 1j, 0, 0],
                                 [0,  0, 0, 0]])
        self.Hsoc_dd = np.array([[0, 0, 0,  0],
                                 [0, 0, 1j, 0],
                                 [0,-1j, 0, 0],
                                 [0,  0, 0, 0]])
        self.Hsoc_ud = np.array([[0, 0, 0,  0],
                                 [0, 0, 0,  1],
                                 [0, 0, 0,-1j],
                                 [0,-1,1j, 0]])
        self.Hsoc_du = np.array([[0, 0, 0,  0],
                                 [0, 0, 0, -1],
                                 [0, 0, 0,-1j],
                                 [0, 1,1j, 0]])
        self.Hsoc = np.zeros((self.n_atom * self.n_orbital * 2, self.n_atom * self.n_orbital * 2), dtype=complex)
        self.Hsoc[:self.n_dim//2, :self.n_dim//2] = np.kron(np.diag(self.delta), self.Hsoc_uu)
        self.Hsoc[:self.n_dim//2, self.n_dim//2:] = np.kron(np.diag(self.delta), self.Hsoc_ud)
        self.Hsoc[self.n_dim//2:, :self.n_dim//2] = np.kron(np.diag(self.delta), self.Hsoc_du)
        self.Hsoc[self.n_dim//2:, self.n_dim//2:] = np.kron(np.diag(self.delta), self.Hsoc_dd)

        # Build Tight Binding Modelb (Bulk and Slab)
        # Bulk
        self.eigenvalue_bulk = np.zeros((self.n_kpoints, self.n_dim),dtype=complex)
        self.H_bulk = np.zeros((self.n_dim,self.n_dim,self.n_kpoints),dtype=complex)

        # Slab
        self.eigenvalue_slab = np.zeros((self.n_kpoints, self.n_layer * self.n_dim),dtype=complex)
        self.H_slab = np.zeros((self.n_layer*self.n_dim, self.n_layer*self.n_dim, self.n_kpoints),dtype=complex)

        # self.H00 = np.zeros(self.n_dim,self.n_dim,self.n_kpoints)
        # self.H01 = np.zeros(self.n_dim, self.n_dim, self.n_kpoints)
        # self.H10 = np.zeros(self.n_dim, self.n_dim, self.n_kpoints)
        zeroCell = np.zeros((self.n_orbital,self.n_orbital))

        progress = ProgressBar( self.n_kpoints ,fmt=ProgressBar.FULL)
        for ik in range(self.n_kpoints):
            # if ik % (self.n_kpoints // 10) == 0:
            #     print('[k-point diagonalized:]',ik / self.n_kpoints * 100,'%')

            # progress report
            progress.current += 1
            progress()
            #

            kpoint = self.k[ik,:]
            H11 = self.build_ham(1, 1,  self.neighbour11, kpoint)
            H33 = self.build_ham(3, 3,  self.neighbour11, kpoint)
            H55 = self.build_ham(5, 5,  self.neighbour11, kpoint)
            H12 = self.build_ham(1, 2,  self.neighbour12, kpoint)
            H13 = self.build_ham(1, 3,  self.neighbour13, kpoint)
            H14 = self.build_ham(1, 4,  self.neighbour14, kpoint)
            H15 = self.build_ham(1, 5,  self.neighbour15, kpoint)
            H23 = self.build_ham(2, 3, -self.neighbour14, kpoint)
            H24 = self.build_ham(2, 4, -self.neighbour13, kpoint)
            H25 = self.build_ham(2, 5, -self.neighbour15, kpoint)
            H34 = self.build_ham(3, 4,  self.neighbour34, kpoint)
            H35 = self.build_ham(3, 5,  self.neighbour35, kpoint)
            H45 = self.build_ham(4, 5, -self.neighbour35, kpoint)

            H0 = np.vstack([np.hstack([H11,             H12,              H13,               H14,              H15]),
                           np.hstack([H12.conjugate().T,H11,              H23,               H24,              H25]),
                           np.hstack([H13.conjugate().T,H23.conjugate().T,H33,               H34,              H35]),
                           np.hstack([H14.conjugate().T,H24.conjugate().T,H34.conjugate().T, H33,              H45]),
                           np.hstack([H15.conjugate().T,H25.conjugate().T,H35.conjugate().T, H45.conjugate().T,H55])])

            H_inner = np.vstack([np.hstack([H11,              H12,              H13,         zeroCell,              H15]),
                                np.hstack([H12.conjugate().T,H11,          zeroCell,              H24,              H25]),
                                np.hstack([H13.conjugate().T,zeroCell,          H33,         zeroCell,              H35]),
                                np.hstack([zeroCell,H24.conjugate().T,     zeroCell,              H33,              H45]),
                                np.hstack([H15.conjugate().T,H25.conjugate().T,H35.conjugate().T,H45.conjugate().T,H55])])

            H_inter = np.vstack([np.hstack([zeroCell, zeroCell,      zeroCell,      H14, zeroCell]),
                                np.hstack([zeroCell, zeroCell,       zeroCell, zeroCell, zeroCell]),
                                np.hstack([zeroCell, H23.conjugate().T, zeroCell,   H34, zeroCell]),
                                np.hstack([zeroCell, zeroCell,       zeroCell, zeroCell, zeroCell]),
                                np.hstack([zeroCell, zeroCell,       zeroCell, zeroCell, zeroCell])])

            self.H_bulk[:,:, ik] = np.kron(np.eye(2, 2), H0) + self.Hsoc

            H00 = np.kron(np.eye(2, 2), H_inner) + self.Hsoc
            H01 = np.kron(np.eye(2,2),  H_inter)
            H10 = H01.conjugate().T

            self.H_slab[:,:,ik] =  np.kron(self.H_QL10,H10) + np.kron(self.H_QL01,H01) + np.kron(self.H_QL00,H00)\
                                   # + np.kron(self.H_QL_perturb_lower,np.ones_like(H00)* self.intraction) \
                                   # + np.kron(self.H_QL_perturb_upper, np.ones_like(H00) * self.intraction)

            # Diagonalize
            self.eigenvalue_bulk[ik,:],_ = np.linalg.eig(self.H_bulk[:,:, ik])
            self.eigenvalue_slab[ik,:],_ = np.linalg.eig(self.H_slab[:,:, ik])

    def plot_bulk(self):
        fig = plt.figure()
        xx = (np.ones_like(np.abs(self.eigenvalue_bulk)) * self.x[:,np.newaxis])
        # plt.scatter(xx, np.real(self.eigenvalue_bulk), s=1, color='blue')
        plt.plot(xx, np.sort(np.real(self.eigenvalue_bulk),axis=1 ), color='blue')
        plt.show()

    def plot_slab(self):
        fig = plt.figure()
        xx = (np.ones_like(np.abs(self.eigenvalue_slab)) * self.x[:,np.newaxis])
        # plt.scatter(xx, np.real(self.eigenvalue_slab), s=1, color='blue')
        plt.plot(xx, np.sort(np.real(self.eigenvalue_slab), axis=1),color='blue')
        plt.ylim((-1.5,2))
        plt.xlim((xx.min(),xx.max()))
        plt.show()

    def build_ham(self,i,j,neighbour,kpoint):
        """
        For different system, this build_ham need to be modifeid
        :param i: atom row index
        :param j: atom column index
        :param neighbour: neighbour
        :return: 4 * 4 hamiltonian in orbital subspace
        """
        i = i-1
        j = j-1
        ss_sigma = self.Vss_sigma[i,j]
        sp_sigma = self.Vsp_sigma[i,j]
        ps_sigma = self.Vps_sigma[i,j]
        pp_sigma = self.Vpp_sigma[i,j]
        pp_pi = self.Vpp_pi[i,j]
        n_neighbour = neighbour.shape[0]
        H_orb_sub = np.zeros((self.n_orbital, self.n_orbital))

        for n in range(n_neighbour):
            d = neighbour[n,:]
            dl = d[0]/np.linalg.norm(d)
            dm = d[1]/np.linalg.norm(d)
            dn = d[2]/np.linalg.norm(d)
            para1 = np.array([[dl**2,dl*dm,dl*dn],
                             [dl*dm,dm**2,dm*dn],
                             [dl*dn,dm*dn,dn**2]])
            para2 = np.eye(3,3) - para1
            pp = pp_sigma*para1 + pp_pi*para2
            V = np.zeros_like(H_orb_sub)
            V[0,:] = np.array([ss_sigma, dl*sp_sigma, dm*sp_sigma, dn*sp_sigma])
            V[1:,0] = np.array([dl*ps_sigma,dm*ps_sigma,dn*ps_sigma])
            V[1:,1:] = pp

            H_orb_sub = H_orb_sub + np.exp(1j*np.dot(kpoint,d)) * V

        if i == j: # add on-site term
            H_orb_sub = H_orb_sub + np.diag(np.array([self.es[i],self.ep[i],self.ep[i],self.ep[i]]))
        return H_orb_sub

if __name__ == "__main__":
    h = Hamiltonian()

