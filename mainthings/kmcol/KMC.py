import matplotlib.pyplot as plt
import itertools, os, copy, random, sys, time, pickle, glob
import numpy as np
from random import uniform, random, choice, randrange, shuffle
from math import sqrt
from matplotlib.cbook import flatten
from matplotlib import pylab
from pylab import *
import random
import pandas as pd
from itertools import accumulate
from matplotlib import colors as mcolors
from scipy.stats.distributions import norm
import pickle
from joblib import dump, load
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from matplotlib import cycler
#import h5py
from scipy import spatial
import gc, yaml


def SEI(br, OUTPUTDIR):
    '''
    Kinetic Monte Carlo model fro SEI growth python script + ploy
    This script includes all required packages, and functions.
    '''
    # ==============================================================================================
    # REQUIRED PYTHON PACKAGES

    # ==============================================================================================
    '''Classes, functions definition'''
    # Boltzman anf Planck constants
    K_B = 8.617333262145e-5  # EVK-1
    h = 4.135667696e-15  # EV.S
    '''function to calculate rates form barriers'''

    def bar_rate(e, T):
        return ((K_B * T) / h) * exp(-e / (K_B * T))

    # ==============================================================================================
    """
    species type:

    S =  for solvent (at the beginning all other sites) - White
    P = first electron reduction product (which can move)
    F = second electron reduction product which cannot move (Li2CO3)
    A = absorbent layer (on top of the box)
    B = Electrode (bottom layer), black can have multiple species depending
        on catalytic activitiy B1, B2, ....
    O = SEI organic ingredient(LiEDC)
    OO= 2 LiEDC agglomeration
    G = gas
    C = SEI cluster
    EC = Active solvent (EC-Li+)
    IS = Inert solvent
    """

    # ==============================================================================================
    # ---------------------------------SITE CLASS-----------------------------------
    # every site on lattice is an object of this class with 3 attributes for coordinate
    # type of species, and species itself
    class site:

        def __init__(self, coordinate, species, bonds, nbr, status):
            self.coordinate = coordinate
            self.species = species
            self.nbr = nbr
            self.bonds = bonds
            self.status = status

        def setN(self, coordinate):
            self.coordinate = coordinate

        def setS(self, species):
            self.species = species

        def setnb(self, nbr):
            self.nbr = nbr

        def setb(self, bonds):
            self.bonds = bonds

        def setss(self, status):
            self.status = status

        def getC(self, coordinate):
            return self.coordinate

        def getS(self, species):
            return self.species

        def getnb(self, nbr):
            return self.nbr

        def getb(self, bonds):
            return self.bonds

        def getss(self, status):
            return self.status

    # to species the color in visualization
    def colors(s):
        if Full(s)[0] == "A":
            return 'yellow'

        elif Full(s)[0] == "E":
            return 'black'

        elif Full(s)[0] == "P":
            return 'green'

        elif Full(s)[0] == "S":
            return 'w'

        elif Full(s)[0] == "O":
            return 'orange'

        elif Full(s)[0] == "F":
            return 'red'

        elif Full(s)[0] == "OO":
            return 'blue'

        elif Full(s)[0] == "C":
            return 'm'

    # ==============================================================================================
    # --------------------------------EVENT CLASS-----------------------------------
    # events/reaction are objects of this class with four attributes
    class event:
        # Generic attributes
        def __init__(self, reactant, product, barrier, rate):
            self.reactant = reactant
            self.rate = rate
            self.barrier = barrier
            self.product = product

        def set_reactant(self, reactant):
            self.reactant = reactant

        def set_rate(self, rate):
            self.rate = rate

        def set_product(self, product):
            self.product = product

        def set_statt(self, barrier):
            self.barrier = barrier

        def get_reactant(self, reactant):
            return self.reactant

        def get_rate(self, rate):
            return self.rate

        def get_product(self, product):
            return self.product

        def get_stat(self, barrier):
            return self.barrier

    # -----------------------------NEW EVENT SUBCLASS-------------------------------
    """
    this subclass helps in collecting events
    """

    class new_event(event):
        def __init__(self, reactant, product, barrier, rate, new_coord):
            super().__init__(reactant, product, barrier, rate)
            self.new_coord = new_coord

        def set_newC(self, new_coord):
            self.new_coord = new_coord

        def get_newC(self, new_coord):
            return self.new_coord

    # ==============================================================================================
    # ----------------------------------- lattice-----------------------------------
    """
    this function provides a NxN lattice with all grid points as site object which
    were defined before.
    """

    def Lattice(xdim, ydim):
        # lattice points as a list of 2-elements sublists as [x,y]
        l = [[i, j] for i, j in itertools.product(range(1, xdim + 1), range(1, ydim + 1))]
        # initialize lattice as a list of site objects// coordinate,species,bonds,nbr,status
        sites = [site(i, [], [], [[i[0] + 1, i[1]], [i[0] - 1, i[1]], [i[0], i[1] + 1], [i[0], i[1] - 1]], [[], [], []])
                 for
                 i in l]
        # sites = [site(i,[],[],[[i[0]+1,i[1]],[i[0]-1,i[1]],[i[0],i[1]+1],[i[0],i[1]-1],[i[0]+1,i[1]+1],[i[0]-1,i[1]+1],[i[0]+1,i[1]-1],[i[0]-1,i[1]-1]]) for i in l]
        # predefining the neighbors
        for i in sites:
            temp = i.nbr
            temp2 = [i for i in sites if i.coordinate in temp]
            i.nbr = temp2
        return sites

    # --------------------------------Next neighbors--------------------------------
    def next_nbr(x):
        n_ = x.nbr
        nn_ = [i.nbr for i in n_]
        nn_ = list(itertools.chain(*nn_))
        nn_ = [i for i in nn_ if i != x]
        return nn_

    # ==============================================================================================
    # -------------------------Selection function for event ------------------------
    """
    this function selects the type of event after ordering them in a list of list.
    it gets the random number and looks for the interval which satisfies the selection scheme.
    5 different regions as a whole selection box. the function select one of them each time.
    """

    def ratee(i, r, R_index):
        if R_index[i - 1] < r <= R_index[i]:
            picked = R_index[i]
            # this index indicates the ith element in index list
            # e.g if this one is 2 one should look at the index list
            # to find the true index for EVENT which is 3
            picked_index = i - 1
            return [picked, picked_index]

    # ---------------------------------append---------------------------------------
    def app_list(l):
        coord_, spec_, color_ = [], [], []
        for i in l:
            coord_.append(i.coordinate)
            spec_.append(i.species)
            color_.append(site.colors(i))
        temp = [coord_, spec_, color_]
        return temp

    # -------------------------Counting species in step----------------------------
    def counter_spec(l, points, t):
        s = 0
        p = 0
        f = 0
        o = 0
        o2 = 0
        c = 0
        ############
        sample = l[:]
        ############
        for i in l:
            if Full(i) == ["S"]:
                s += 1
            elif Full(i) == ["F"]:
                f += 1
            elif Full(i) == ["P"]:
                p += 1
            elif Full(i) == ["O"]:
                o += 1
            elif Full(i) == ["OO"]:
                o2 += 1
            elif Full(i) == ["C"]:
                c += 1

        # O2 per 2
        temp = [s, f, p, o, o2, c]
        temp = [float(item / points) for item in temp]
        temp = temp + [t]
        return temp

    # ==============================================================================================
    # getting full name of attached components
    def Full(x):
        # F and C no added letter
        if x.species[0] in ["C", "F"] or len(x.bonds) == 0:
            return x.species
        else:
            if x.bonds:
                temp = x.species[0]
                for i in x.bonds:
                    if i != x:
                        temp = temp + i.species[0]
                return [temp]

    # ==============================================================================================
    def delta(y, x):
        temp = [x.coordinate[0] - y.coordinate[0], x.coordinate[1] - y.coordinate[1]]
        return temp

    # ==============================================================================================
    # rate function
    def rate_func(x, r):
        return r * np.exp(-0.1 * x)

    # ==============================================================================================
    def sort_(x):
        diff, others = [], []
        for i in x:
            if ["S"] in i.reactant and ["S"] in i.product:
                diff.append(i)
            else:
                others.append(i)
        temp = [diff, others]
        return temp

    # ==============================================================================================
    def rand_site(l):
        c = time.time()
        while True:
            '''set a waiting tile to pick a random place'''
            if time.time() > c + 1:
                X = []
                return X
            else:
                X = choice(l)
                if X.species == ["S"]:
                    # nbs = [i for i in X.nbr if i.species == ["S"]]
                    # if len(nbs) >= 1:
                    #     test = [len([j for j in i.nbr if j.species == ["S"]]) for i in X.nbr]
                    #     if 2 in test:
                    return X

    def dis(x1, x0):
        return sqrt((x1[0] - x0[0]) ** 2 + (x1[1] - x0[1]) ** 2)

    def list_splitter(list_to_split, ratio):
        elements = len(list_to_split)
        middle = int(elements * ratio)
        return [list_to_split[:middle], list_to_split[middle:]]

    def round_float_list(float_list, decimal_points):
        float_list = [round(float(item), decimal_points) for item in float_list]
        return float_list

    # ==============================================================================================
    ####################################### MAIN PART #############################################
    # ==============================================================================================
    with open('sei_args.yml') as file:
        args = yaml.full_load(file)
    xdim = args['Xdim']
    ydim = args['Ydim']
    T = args['T']
    vis_save = args['SaveStep']
    maxsteps = args['Itters']
    # saving parameters
    frac_div = 50 * vis_save
    frac_save = 50 * vis_save
    TIME = args['time']
    Xoutput = args['Xoutput']
    '''''''''
    with open('barriers.yml') as file:
        raw_list = yaml.full_load(file)
        br = list(raw_list.values())
    '''''''''
    br = br
    # To read barriers for samples generated by DOE in barries folder uncomment  lines 327 to 341 and comment lines 322 to 324.
    # With this one need to run the code with one aditional arg as sample number like: python3 sei_al.py $sample
    '''
    #to get the TASK ID
    if (len(sys.argv) < 2 or len(sys.argv) > 2):
        print("just division size")
        sys.exit()
    # to get the arguments and check whether they are in correct format or not.
    if (len(sys.argv) > 1):
        try:
            sample = int(sys.argv[1])
        except ValueError:
            print("sample number is the TASK ID on server")
            sys.exit()
    with open('barriers/list_'+str(sample),'rb') as f:
        br = list(pickle.load(f))
    '''
    ###############################################################################
    """
    creating a lattice
    """
    lattice = Lattice(xdim, ydim)
    points = len(lattice)
    os.getcwd()

    # hashh = str(hash(time.time()))
    '''change the directory to have dump files in a new folder outside '''
    # os.chdir("/home/ws/ab5528/simstack_workspace/")
    # base = 'data_xdim' + str(xdim) + '_ydim_' + str(ydim) + '_T_' + str(T) + '_SaveStep_' + str(vis_save) + '_H'+hashh
    output_dir = OUTPUTDIR + '/output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    fraction_dir = OUTPUTDIR + '/fraction'
    if not os.path.exists(fraction_dir): os.makedirs(fraction_dir)

    '''follow up modification for DOE comment 351 to 354 and uncomment 357 to 362'''
    '''
    output_dir   = 'data_xdim'+str(xdim)+'_ydim_'+str(ydim)+'_T_'+str(T)+'_SaveStep_'+str(vis_save)+'_Sample_'+str(sample)+'/output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    fraction_dir = 'data_xdim'+str(xdim)+'_ydim_'+str(ydim)+'_T_'+str(T)+'_SaveStep_'+str(vis_save)+'_Sample_'+str(sample)+'/fraction'
    if not os.path.exists(fraction_dir): os.makedirs(fraction_dir)
    '''
    # --------------------------------------------------------------------------------
    '''
    lattice consideration: EC as the initial concentration of active EC.
    it should be taken into account as first distribution.
    First lattice is like
        up    : A
        down  : B
        middle: S
    '''
    '''for cases we need to reduce CPU hours'''
    for i in lattice:
        if i.coordinate[1] == 1:
            i.species = ["E"]
        elif i.coordinate[1] == ydim:
            i.species = ["A"]
        else:
            i.species = ["S"]

    # ###############################################################################
    '''Events'''
    e1 = event([["E"], ["S"]], [["E"], ["P"]], br[0], 0)
    e2 = event([["F"], ["S"]], [["F"], ["P"]], br[1], 0)
    e3 = event([["E"], ["P"]], [["E"], ["F"]], br[2], 0)
    e4 = event([["P"], ["P"]], [["O"], ["S"]], br[3], 0)
    e5 = event([["O"], ["O"]], [["OO"]], br[4], 0)  # 731812
    e6 = event([["O"], ["OO"]], [["C"]], br[5], 0)
    e7 = event([["F"], ["P"]], [["F"], ["F"]], br[6], 0)  # 717589
    e8 = event([["O"], ["C"]], [["C"]], br[7], 0)
    e9 = event([["OO"], ["C"]], [["C"]], br[8], 0)
    e10 = event([["OO"], ["OO"]], [["C"]], br[9], 0)
    e11 = event([["C"], ["C"]], [["C"]], br[10], 0)  # 733897
    e12 = event([["OO"], ["S"]], [["S"], ["OO"]], br[11], 0)  # 534308
    e13 = event([["O"], ["S"]], [["S"], ["O"]], br[12], 0)  # 384422
    e14 = event([["P"], ["S"]], [["S"], ["P"]], br[13], 0)  # 461731
    e15 = event([["C"], ["S"]], [["S"], ["C"]], br[14], 0)
    e16 = event([["P"], ["A"]], [["A"], ["S"]], 0.01, 0)
    e17 = event([["OO"], ["A"]], [["A"], ["S"]], 0.01, 0)
    e18 = event([["C"], ["A"]], [["A"], ["S"]], 0.01, 0)
    e19 = event([["O"], ["A"]], [["A"], ["S"]], 0.01, 0)
    #
    # list of events
    Events = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19]
    ok = [e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19]
    '''rates based on temperature'''
    for i in Events:
        temp = i.barrier
        i.rate = bar_rate(temp, T)
        #print(temp, i.rate)
    ##############################################################################
    '''helps with clusters diffusion'''
    direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    ###############################################################################
    """
    collection function
    """

    def pre_event(top_list):
        pre_list = list()
        del pre_list[:]
        for s in top_list:
            # determine the type of sites
            ss = Full(s)
            if ss != ["C"]:
                NBR = [i for i in s.nbr if i not in s.bonds]
                for nbr in NBR:
                    ns = Full(nbr)
                    C_check = [ss, ns]
                    # to divide the list into noncluster elements
                    if ["C"] not in C_check:
                        '''to all diffusions'''
                        if ss == ["E"] and ns == ["S"]:
                            ev = e1
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["F"] and ns == ["S"]:
                            ev = e2
                            '''tunneling distance is set on 4nm'''
                            test = [i for i in [nbr, s] if i.species == ["S"] and i.coordinate[1] <= 5]
                            if test:
                                pre_list.append(
                                    new_event(ev.reactant, ev.product, ev.barrier, ev.rate,
                                                [s, nbr, Events.index(ev)]))

                        elif ss == ["E"] and ns == ["P"]:
                            ev = e3
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["P"] and ns == ["P"]:
                            ev = e4
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["O"] and ns == ["O"]:
                            ev = e5
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif (ss == ["O"] and ns == ["OO"]):
                            ev = e6
                            m = [i for i in [nbr, s] if Full(i) == ["O"]]
                            n = [i for i in [nbr, s] if Full(i) == ["OO"]]
                            siteO = m[0]
                            siteO2 = n[0]
                            pre_list.append(new_event(ev.reactant, ev.product, ev.barrier, ev.rate,
                                                        [siteO2, siteO, Events.index(ev)]))

                        elif ss == ["F"] and ns == ["P"]:
                            ev = e7
                            test = [i for i in [nbr, s] if i.species == ["P"] and i.coordinate[1] <= 5]
                            if test:
                                pre_list.append(
                                    new_event(ev.reactant, ev.product, ev.barrier, ev.rate,
                                                [s, nbr, Events.index(ev)]))

                        elif ss == ["OO"] and ns == ["OO"]:
                            ev = e10
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["OO"] and ns == ["S"]:
                            ev = e12
                            new = [i for i in nbr.nbr if i.species == ["S"]]
                            if new:
                                nw_nbr = choice(new)
                                pre_list.append(new_event(ev.reactant, ev.product, ev.barrier, ev.rate,
                                                            [s, nbr, nw_nbr, Events.index(ev)]))

                        elif ss == ["O"] and ns == ["S"]:
                            ev = e13
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["P"] and ns == ["S"]:
                            ev = e14
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["P"] and ns == ["A"]:
                            ev = e16
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["OO"] and ns == ["A"]:
                            ev = e17
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                        elif ss == ["O"] and ns == ["A"]:
                            ev = e19
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [s, nbr, Events.index(ev)]))

                    # those with C and O OO
                    elif ["C"] in C_check and (["O"] in C_check or ["OO"] in C_check):
                        if ss == ["O"]:
                            ev = e8
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [nbr, s, Events.index(ev)]))
                        elif ss == ["OO"]:
                            ev = e9
                            pre_list.append(
                                new_event(ev.reactant, ev.product, ev.barrier, ev.rate, [nbr, s, Events.index(ev)]))
            elif ss == ["C"]:
                sb = s.bonds
                cb = sb + [s]
                cb_cord = [i.coordinate for i in cb]
                # all surounding sisepciestes
                all_nbr = [j for j in list(dict.fromkeys(list(itertools.chain(*([i.nbr for i in cb]))))) if
                            j not in cb]
                named_list = [i.species for i in all_nbr]
                if ["O"] in named_list:
                    n_spec = [i for i in all_nbr if i.species == ["O"]]
                    if Full(n_spec[0]) == ["O"]:
                        pre_list.append(
                            new_event(e8.reactant, e8.product, e8.barrier, e8.rate,
                                        [s, n_spec[0], Events.index(e8)]))
                    elif Full(n_spec[0]) == ["OO"]:
                        pre_list.append(
                            new_event(e9.reactant, e9.product, e9.barrier, e9.rate,
                                        [s, n_spec[0], Events.index(e9)]))
                elif ["A"] in named_list:
                    '''
                    to avoid removing the main cluster
                    if it is closed to E it should not be go away !!! not accurate
                    '''
                    if ["E"] not in named_list:
                        pre_list.append(
                            new_event(e18.reactant, e18.product, e18.barrier, e18.rate,
                                        [s, s, cb, Events.index(e18)]))

                # single diff of particles:
                # not for attached C
                elif ["F"] not in named_list:
                    # directions lists
                    if xdim not in [i.coordinate[0] for i in cb]:
                        rs = [i for i in all_nbr if [(i.coordinate[0] - 1), i.coordinate[1]] in cb_cord]
                    else:
                        rs = []

                    if 1 not in [i.coordinate[0] for i in cb]:
                        ls = [i for i in all_nbr if [(i.coordinate[0] + 1), i.coordinate[1]] in cb_cord]
                    else:
                        ls = []

                    if ydim not in [i.coordinate[1] for i in cb]:
                        us = [i for i in all_nbr if [i.coordinate[0], (i.coordinate[1] - 1)] in cb_cord]
                    else:
                        us = []

                    if 1 not in [i.coordinate[1] for i in cb]:
                        ds = [i for i in all_nbr if [i.coordinate[0], (i.coordinate[1] + 1)] in cb_cord]
                    else:
                        ds = []

                    dir_list = [rs, ls, us, ds]
                    for i in dir_list:  # 0 1 2 3
                        if i:
                            '''condition for each possible diffusion S EC [] '''
                            if len(i) == len([j for j in i if j.species == ["S"]]):
                                # not more move when reached to F
                                ev = e15
                                ind = dir_list.index(i)
                                '''diffusion of clusters depends on the size of them !!!!!!!!!!!!!!!!!!!!'''
                                # pass the direction of diffusion using the direction vector RATE DEPENDS ON SIZE
                                evra = ev.rate * exp(-(len(cb)) / 30)
                                pre_list.append(new_event(ev.reactant, ev.product, ev.barrier, evra,
                                                            [s, direction[ind], i, all_nbr, cb, Events.index(ev)]))

                            C_test = [j for j in i if j.species == ["C"]]
                            if C_test:
                                k = C_test[0]
                                ev = e11
                                cb2 = sb + [s]
                                evra = ev.rate
                                # evra = ev.rate * exp(-(len(cb)) / 30)
                                # all surounding sisepciestes
                                all_nbr2 = [j for j in
                                            list(dict.fromkeys(list(itertools.chain(*([i.nbr for i in cb2]))))) if
                                            j not in cb2]
                                pre_list.append(new_event(ev.reactant, ev.product, ev.barrier, evra,
                                                            [s, k, all_nbr, all_nbr2, Events.index(ev)]))

        return pre_list

    ###############################################################################
    """
    update function
    """

    def update(ev, lattice):
        """
        e1 = event([["E"],["EC"]],[["E"],["P"])
        """
        if ev.reactant == e1.reactant:
            if ev.new_coord[0].species == ["E"]:
                ev.new_coord[1].species = ["P"]
            else:
                ev.new_coord[0].species = ["P"]
            ev.new_coord[0].bonds = []
            ev.new_coord[1].bonds = []
            return lattice

        """
        e2 = event([["F"],["EC"]],[["F"],["P"])
        """
        if ev.reactant == e2.reactant:
            if ev.new_coord[0].species == ["F"]:
                ev.new_coord[1].species = ["P"]
            else:
                ev.new_coord[0].species = ["P"]
            ev.new_coord[0].bonds = []
            ev.new_coord[1].bonds = []
            return lattice
        """
        e3 = event([["E"],["P"]],[["E"],["F"]])

        """
        if ev.reactant == e3.reactant:
            if ev.new_coord[0].species == ["E"]:
                ev.new_coord[1].species = ["F"]

            else:
                ev.new_coord[0].species = ["F"]
            ev.new_coord[1].bonds = []
            ev.new_coord[0].bonds = []
            return lattice
        """
        e4 = event([["P"],["P"]],[["O"],["S"]])
        """
        if ev.reactant == e4.reactant:
            # pick one randomly
            pick = [ev.new_coord[0], ev.new_coord[1]]
            p = choice(pick)
            p.species = ["O"]
            g = [i for i in pick if i != p]
            g[0].species = ["S"]
            g[0].bonds = []
            p.bonds = []
            return lattice
        """
            e5 = event([["O"],["O"]],[["OO"]])
        """
        if ev.reactant == e5.reactant:
            ev.new_coord[0].bonds = []
            ev.new_coord[1].bonds = []
            ev.new_coord[0].bonds = [ev.new_coord[1]]
            ev.new_coord[1].bonds = [ev.new_coord[0]]
            return lattice
        """
        e6 = event([["O"],["OO"]],[["C"]])
        """
        if ev.reactant == e6.reactant:
            s1 = ev.new_coord[0]
            s2 = ev.new_coord[1]
            sb = s1.bonds[0]
            s1.bonds = []
            s2.bonds = []
            sb.bonds = []
            s1.species = ["C"]
            s2.species = ["C"]
            sb.species = ["C"]
            s1.bonds = [sb, s2]
            s2.bonds = [s1, sb]
            sb.bonds = [s1, s2]
            return lattice
        """
        e7 = event([["F"],["P"]],[["F"],["F"]])
        """
        if ev.reactant == e7.reactant:
            ev.new_coord[1].species = ["F"]
            ev.new_coord[0].species = ["F"]
            ev.new_coord[0].bonds = []
            ev.new_coord[1].bonds = []
            return lattice
        """
        e8= event([["O"],["C"]],[["C"]])
        """
        if ev.reactant == e8.reactant:
            # C
            s2 = ev.new_coord[0]
            # O
            s1 = ev.new_coord[1]
            s1.bonds = []
            s1.species = ["C"]
            sl = [s2] + s2.bonds
            s1.bonds = sl
            for i in sl:
                i.bonds.append(s1)
            return lattice
        """
        e9= event([["OO"],["C"]],[["C"]])
        """
        if ev.reactant == e9.reactant:
            # C
            s2 = ev.new_coord[0]
            # O2
            s1 = ev.new_coord[1]
            sb = s1.bonds[0]
            s1.bonds = []
            sb.bonds = []
            s1.species = ["C"]
            sb.species = ["C"]
            sl = [s2] + s2.bonds
            s1.bonds = [sb] + sl
            sb.bonds = [s1] + sl
            '''previous cluster'''
            for i in sl:
                i.bonds = i.bonds + [s1, sb]
            return lattice
        """
        e10= event([["OO"],["OO"]],[["C"]])
        """
        if ev.reactant == e10.reactant:
            s = ev.new_coord[0]
            d = ev.new_coord[1]
            sb = s.bonds[0]
            db = d.bonds[0]
            cb = [s, sb, d, db]
            for i in cb:
                i.bonds = []
                i.species = ["C"]
                i.bonds = [j for j in cb if j != i]
            return lattice
        """
        e11 C+C---> C
        """
        if ev.reactant == e11.reactant:
            s = ev.new_coord[0]
            n = ev.new_coord[1]
            sb = [s] + s.bonds
            nb = [n] + n.bonds
            for i in sb:
                i.bonds = []
            for i in nb:
                i.bonds = []
            for j in sb:
                temp = [i for i in sb if i != j]
                j.bonds = temp + nb
            for j in nb:
                temp = [i for i in nb if i != j]
                j.bonds = temp + sb
            return lattice
        """
            e12 = event([["OO"],["S"]],[["S"],["OO"]])
        """
        if ev.reactant == e12.reactant:
            O1 = ev.new_coord[0]  # OO
            n1 = ev.new_coord[1]  # EC/IS/S
            O2 = O1.bonds[0]
            n2 = event.new_coord[2]  # S/EC/[]
            O1.bonds = []
            O2.bonds = []
            n1.bonds = []
            n2.bonds = []
            O1.species = ["S"]
            O2.species = ["S"]
            n1.species = ["O"]
            n2.species = ["O"]
            n1.bonds = [n2]
            n2.bonds = [n1]
            return lattice
        """
        e13= event([["O"],["S"]],[["S"],["O"]])
        """
        if ev.reactant == e13.reactant:
            s1 = ev.new_coord[0]
            s2 = ev.new_coord[1]

            s1.species = ["S"]
            s2.species = ["O"]
            s1.bonds = []
            s2.bonds = []
            return lattice
        """
        e14= event([["P"],["S"]],[["S"],["P"]])
        """
        if ev.reactant == e14.reactant:
            s1 = ev.new_coord[0]
            s2 = ev.new_coord[1]
            s1.species = ["S"]
            s2.species = ["P"]
            s1.bonds = []
            s2.bonds = []
            return lattice
        """
        e15 C---> C
        """
        if ev.reactant == e15.reactant:
            s = ev.new_coord[0]
            # the direction vector
            d = ev.new_coord[1]
            sorted_cb = []
            sb = s.bonds
            cb = sb + [s]
            for i in cb:
                i.bonds = []
                i.species = ["S"]

            if d == [1, 0]:
                sort_cb = sorted([i.coordinate for i in cb], key=lambda k: [k[0], k[1]], reverse=True)
                for j in sort_cb:
                    temp = [i for i in cb if i.coordinate == j]
                    sorted_cb.append(temp[0])
                new = []
                for i in sorted_cb:
                    # i.species = ["S"]
                    jcoord = [i.coordinate[0] + d[0], i.coordinate[1] + d[1]]
                    test = [k for k in i.nbr if k.coordinate == jcoord]
                    test[0].species = ["C"]
                    test[0].bonds = []
                    new.append(test[0])
            elif d == [-1, 0]:
                sort_cb = sorted([i.coordinate for i in cb], key=lambda k: [k[0], k[1]], reverse=False)
                for j in sort_cb:
                    temp = [i for i in cb if i.coordinate == j]
                    sorted_cb.append(temp[0])
                new = []
                for i in sorted_cb:
                    # i.species = ["S"]
                    jcoord = [i.coordinate[0] + d[0], i.coordinate[1] + d[1]]
                    test = [k for k in i.nbr if k.coordinate == jcoord]
                    test[0].species = ["C"]
                    test[0].bonds = []
                    new.append(test[0])
            elif d == [0, 1]:
                sort_cb = sorted([i.coordinate for i in cb], key=lambda k: [k[1], k[0]], reverse=True)
                for j in sort_cb:
                    temp = [i for i in cb if i.coordinate == j]
                    sorted_cb.append(temp[0])
                new = []
                for i in sorted_cb:
                    # i.species = ["S"]
                    jcoord = [i.coordinate[0] + d[0], i.coordinate[1] + d[1]]
                    test = [k for k in i.nbr if k.coordinate == jcoord]
                    test[0].species = ["C"]
                    test[0].bonds = []
                    new.append(test[0])
            elif d == [0, -1]:
                sort_cb = sorted([i.coordinate for i in cb], key=lambda k: [k[1], k[0]], reverse=False)
                for j in sort_cb:
                    temp = [i for i in cb if i.coordinate == j]
                    sorted_cb.append(temp[0])
                new = []
                for i in sorted_cb:
                    # i.species = ["S"]
                    jcoord = [i.coordinate[0] + d[0], i.coordinate[1] + d[1]]
                    test = [k for k in i.nbr if k.coordinate == jcoord]
                    test[0].species = ["C"]
                    test[0].bonds = []
                    new.append(test[0])

            for j in new:
                j.bonds = [i for i in new if i != j]

            return lattice
        """
        all top layer reactions
        """
        if ev.reactant == e16.reactant or ev.reactant == e17.reactant or ev.reactant == e18.reactant or ev.reactant == e19.reactant:
            '''if we need full space with solvent we have to change this part to '''
            if ev.new_coord[0].species != ["A"]:
                ev.new_coord[0].species = ["S"]
                bd = len(ev.new_coord[0].bonds + [ev.new_coord[0]])
                if ev.new_coord[0].bonds:
                    for i in ev.new_coord[0].bonds:
                        i.species = ["S"]
                        i.bonds = []
                ev.new_coord[0].bonds = []
            # counting scaped species
            if ["P"] in ev.reactant:
                Counter[0] += 1
            elif ["O"] in ev.reactant:
                Counter[1] += 1
            elif ["OO"] in ev.reactant:
                Counter[2] += 2
            elif ev.reactant == e18.reactant:
                Counter[3] += bd
            return lattice
        ###############################################################################

    """
    initial conditions

    as the the beginning it is enough to take into account last and first to rows of species code can make a list of them to
    collect possible events
    """
    EE = [i for i in lattice if i.species == ["E"]]
    '''at beginging kmc needs to take all species could react'''
    top_list = EE
    # time
    t = 0
    # list of eventts
    pre = []
    num = 0
    # event counter and residence time
    C = [0] * len(Events)
    res_time = [0] * len(Events)
    '''
    0:p
    1:o
    2:o2
    3:c
    4: ec
    5:is
    6: consumed ec [li]
    '''
    Counter = [0] * 4
    # to save output
    frc_num = int(frac_save / frac_div)
    vis_num = int(frac_save / vis_save)
    frc_ = 0
    vis_ = 0
    fraction_list = np.zeros([frc_num], dtype=list)
    lists = np.zeros([vis_num], dtype=list)
    keep = 0
    tracking = {}
    # --------------------------------------------------------------------------------
    """
                                            KMC model
    """
    start_time = time.time()
    curtime = 0
    while curtime < TIME:

        temp = []
        del temp[:]
        # have electrode layer in
        '''get single representative of each cluster'''
        # print(iss,ecc,iss0,ecc0) - check for polymer and cluster left on lattice
        spec_in_top = [i for i in lattice if i.species in [["C"]]]
        top_list = [i for i in top_list if i not in spec_in_top]
        catch = []
        for i in spec_in_top:
            if i not in catch:
                temp = [i] + i.bonds
                catch = catch + temp
                top_list.append(i)
        top_list = list(dict.fromkeys((top_list + EE)))
        temp = pre_event(top_list)
        pre = temp + pre
        if len(pre) == 0:
            spec_in_top = [i for i in lattice if i.species not in [["S"], ["F"], ["E"]]]
            catch = []
            spec_top = []
            for i in spec_in_top:
                if i not in catch:
                    temp = [i] + i.bonds
                    catch = catch + temp
                    spec_top.append(i)
            top_list = []
            del top_list[:]
            top_list = spec_top
            pre = pre_event(top_list)
            del spec_top[:]
            del spec_in_top[:]
            if len(pre) == 0:
                coordinate_, spec_, color_, bonds_, status_, = [], [], [], [], []
                for i in lattice:
                    coordinate_.append(i.coordinate)
                    spec_.append(i.species)
                    color_.append(colors(i))
                    status_.append(i.status)
                    bonds_.append(([lattice.index(j) for j in i.bonds]))
                lists[vis_] = [coordinate_, spec_, color_, bonds_, status_] + [t]
                vis_ += 1
                tem_frac = counter_spec(lattice, points, t)
                fraction_list[frc_] = tem_frac
                frc_ += 1
                stat = [time.time() - start_time, t, num]
                try:
                    with open(output_dir + "/list_" + str(keep + frac_save), 'wb') as out:
                        pickle.dump(lists, out)
                    with open(fraction_dir + "/fraction_" + str(keep + frac_save), 'wb') as out:
                        pickle.dump(fraction_list, out)
                except:
                    print("I/O error")

                del fraction_list
                del lists
                frc_ = 0
                vis_ = 0
                tracking["Occurrence"] = C
                tracking["Residence_time"] = round_float_list(res_time, 30)
                tracking["Counter"] = Counter
                tracking["Status"] = round_float_list(stat, 30)
                tracking['barriers'] = br
                try:

                    with open(output_dir + "/tracking.yml", 'w') as out:
                        yaml.dump(tracking, out, default_flow_style=False)

                except IOError:
                    print("I/O error")

                break

        else:
            EVENTS = pre[:]
            shuffle(EVENTS)
            R1 = list(accumulate([i.rate for i in EVENTS]))
            R_index = []
            del R_index[:]
            R_index = [0]
            R_index = R_index + R1
            cum_rate = R_index[-1]

            while True:
                rho1 = random.random()
                r = rho1 * (R_index[-1])
                if (R_index[0] < r and r <= R_index[-1]):
                    break
            rate = []
            for i in range(1, len(R_index)):
                temp = ratee(i, r, R_index)
                rate.append(temp)
            # EVENT index is:
            which = [i for i in rate if i]
            rate_d = which[0][1]
            event = EVENTS[rate_d]
            del EVENTS[:]
            del rate[:]
            del R1[:]
            """
            Time increment
            """
            dt = -(np.log(random.random()) / cum_rate)
            t = t + dt
            """
            UPDATING the list of sites using the function considering the type of selected event.
            """
            lattice = update(event, lattice)
            """
            removing old events related to selected sites and nbrs
            """
            # it would be usefull later to keep C around eliminated F
            if event.new_coord[-1] in [14]:
                all_nbr = event.new_coord[3]
                main_before = [event.new_coord[0]] + event.new_coord[0].bonds + all_nbr
                if ["F"] in [i.species for i in all_nbr]:
                    pre = [i for i in pre if event.new_coord[0] not in all_nbr]
                pre = [i for i in pre if i.new_coord[0] not in main_before and i.new_coord[1] not in main_before]
            else:
                s1 = event.new_coord[0]
                s2 = event.new_coord[1]
                main_ = [s1] + s1.nbr + [s2] + s2.bonds + s2.nbr
                main_ = list(dict.fromkeys(main_))
                test = [i.bonds for i in main_ if i.species in [["C"]]]
                if test:
                    all_nbr = list(itertools.chain(*(test)))
                    main_before_ = [j for j in
                                    list(dict.fromkeys(list(itertools.chain(*([i.nbr for i in all_nbr])))))
                                    if j not in all_nbr]
                    main_before = main_before_ + main_
                    pre = [i for i in pre if
                            i.new_coord[0] not in main_before and i.new_coord[1] not in main_before]

                else:
                    main_before = main_
                    pre = [i for i in pre if
                            i.new_coord[0] not in main_before and i.new_coord[1] not in main_before]

            pre = [i for i in pre if i.new_coord[-1] not in [14, 10, 18]]  # it depends on main lsit!!!
            main_before = list(dict.fromkeys(main_before))
            pre1 = [i for i in pre if i.new_coord[-1] in [11, 25]]
            pre2 = [i for i in pre if i not in pre1]
            pre1 = [i for i in pre1 if i.new_coord[2] not in main_before]
            pre = pre1 + pre2

            counter = event.new_coord[-1]
            C[counter] += 1
            num += 1
            t22, t11 = 0,0

            del top_list[:]
            top_list = [i for i in main_before if i.species != ["S"]]
            res_time[counter] += dt

            if num % vis_save == 0:
                coordinate_, spec_, color_, bonds_, status_, = [], [], [], [], []
                for i in lattice:
                    coordinate_.append(i.coordinate)
                    spec_.append(i.species)
                    color_.append(colors(i))
                    status_.append(i.status)
                    bonds_.append(([lattice.index(j) for j in i.bonds]))
                lists[vis_] = [coordinate_, spec_, color_, bonds_, status_] + [t]
                vis_ += 1

            if num % frac_div == 0:
                tem_frac = counter_spec(lattice, points, t)
                fraction_list[frc_] = tem_frac
                frc_ += 1

            if num % frac_save == 0:
                #print(f'No. Step:   {num}', end='\r', flush=True)
                others = [i for i in lattice if i.species in [["P"], ["O"], ["OO"]]]
                stat = [time.time() - start_time, t, num]

                try:
                    outputlast = output_dir + "/list_" + str(num)
                    outputindex = num

                    with open(output_dir + "/list_LAST", 'wb') as out:
                        pickle.dump(lists, out)
                    with open(fraction_dir + "/fraction_" + str(num), 'wb') as out:
                        pickle.dump(fraction_list, out)


                except:
                    print("I/O error")

                del fraction_list
                del lists

                fraction_list = np.empty([frc_num], dtype=list)
                lists = np.zeros([vis_num], dtype=list)
                frc_ = 0
                vis_ = 0
                tracking["Occurrence"] = C
                tracking["Residence_time"] = round_float_list(res_time, 30)
                tracking["Counter"] = Counter
                tracking["Status"] = round_float_list(stat, 30)
                tracking['barriers'] = br
                try:

                    with open(output_dir + "/tracking.yml", 'w') as out:
                        yaml.dump(tracking, out, default_flow_style=False)


                except IOError:
                    print("I/O error")

                keep = num
        curtime = time.time() - start_time - (t22-t11)

    return outputlast, outputindex, curtime



def Analy(datadir, OII, File, br, brINDEX, Freq=0, Concentration=0, Porosity=0, Thickness=0, Flux=1):
    # ==============================================================================================
    ####################################### MAIN PART #############################################
    # ==============================================================================================

    #      <WaNoFile logical_filename="Parameters"  name="Parameters"> Parameters </WaNoFile>
    # #---------------------------------ARGUMENTS-----------------------------------
    with open('sei_args.yml') as file:
        wano_file2 = yaml.full_load(file)

    #KMEAN = load('kmeanRP of rdfRP-4-5-05-25-25-1')
    #dtcloaded = load('dtc ds50knew kmeanRP of rdfRP-4-5-05-25-25-1')
    #pcaloaded = load('pca rdfRP-4-5-05-25-25-1')

    xdim = wano_file2['Xdim']
    ydim = wano_file2['Ydim']
    T = wano_file2['T']
    vis_save = wano_file2['SaveStep']
    option = wano_file2['Options']
    # data_dir = '../../../' + data_dir
    # initial stuffs

    data_dir = datadir
    # data_dir = 'data_xdim' + str(xdim) + '_ydim_' + str(ydim) + '_T_' + str(T) + '_SaveStep_' + str(vis_save)
    '''second analysis path should be managed in advanced'''

    output_dir = datadir + '/analysis' + File

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    divd = 50 * vis_save
    step_cut = 50 * vis_save
    point = xdim * ydim

    '''
    prepration of figs
    '''
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['xtick.major.pad'] = '8'  # set the gap between the labels and the axis
    plt.rcParams['ytick.major.pad'] = '5'
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    plt.rcParams['savefig.facecolor'] = 'whitesmoke'
    plt.rc('axes', linewidth=1.8)
    # -------------------------------Graphic part-----------------------------------
    # openning a directory to store the pics
    img = 0

    #print('\n stage 1 done')

    def snapshots(pos, colors):
        global img
        pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
        pylab.gcf().set_size_inches(10, 10)
        pylab.axis([0, 1, 0, 1])
        pylab.title('step = ' + str(step_num) + '   t = ' + str(round(t,5) ))
        pylab.xlabel('$x$')
        pylab.ylabel('$y$')
        pylab.setp(pylab.gca(), xticks=[0, xdim + 1], yticks=[0, ydim + 1])
        for (x, y), c in zip(pos, colors):
            circle = pylab.Circle((x, y), radius=0.5, fc=c)  # Radius is r of particle in pics
            pylab.gca().add_patch(circle)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()
        pylab.savefig((confgs + '/last%d.png' % img), transparent=True)  # ,dpi=100)    img += 1
        img += 1
        pylab.close()
        plt.close()
        # --------------------

    def snapshot(pos, colors):
        pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
        pylab.gcf().set_size_inches(10, 10)
        pylab.axis([0, 1, 0, 1])
        pylab.title('step = ' + str(step_num) + '   t = ' + str(round(t,5) ) )
        pylab.xlabel('$x$')
        pylab.ylabel('$y$')
        pylab.setp(pylab.gca(), xticks=[0, xdim + 1], yticks=[0, ydim + 1])
        for (x, y), c in zip(pos, colors):
            circle = pylab.Circle((x, y), radius=0.5, fc=c)  # Radius is r of particle in pics
            pylab.gca().add_patch(circle)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()
        pylab.savefig(output_dir + '/last.png', transparent=True)
        pylab.close()
        plt.close()

    def rdf2(allpoints, typ1, typ2, dr, rmax, poi, DD):

        r1, r2, dr = 0., rmax, dr

        g = []
        rb = []

        raddi = np.arange(0, r2, dr)
        T = np.array(allpoints[2])
        allpoints = np.array(allpoints[0])

        # return allpoints, T
        points = allpoints[np.where((T == typ1) | (T == typ2)| (T == 'orange') | (T == 'blue') | (T == 'green')   )[0]]
        # return points
        tree = spatial.cKDTree(points)

        N = len(points)
        rho = N / DD
        cnt = 0

        for r in raddi:
            tmp = 0
            n = tree.query_ball_point(poi, r + dr, return_length=True) - tree.query_ball_point(poi, r,return_length=True)
            tmp += n
            tmp /= 0.5 * np.pi * ((r + dr) ** 2 - r ** 2) * rho
            g.append(tmp)
            rb.append(r)
            cnt += 1

        return rb, g
    ###############################################################################
    '''
    concentration graph
    '''
    if Concentration:

        #print('entering concentration')
        listf = os.listdir(data_dir + '/fraction/')
        it = max([int(i) for i in [s.replace("fraction_", "") for s in listf]])
        spec_list = []
        for ii in range(divd, it + divd, divd):
            # load the data
            if os.path.getsize(data_dir + '/fraction/fraction_' + str(ii)) > 0:
                with open(data_dir + '/fraction/fraction_' + str(ii), 'rb') as fp:
                    l = pickle.load(fp)
                for i in l:
                    # print(ii)
                    if type(i) != list:
                        l1 = i.tolist()
                    else:
                        l1 = i
                    spec_list.append(l1)
                del l
        t = spec_list[-1][-1]
        df = pd.DataFrame(spec_list)
        df.columns = ["S", "F", "P", "O", "O2", "C", 'Time']
        new_order1 = [-1, 0, 1, 2, 3, 4, 5]
        df_spec1 = df[df.columns[new_order1]]
        # df_spec1 = df_spec1.replace(0, np.nan) # df_spec1 = df_spec1.expanding(10).mean()
        plt.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.97, wspace=0, hspace=0)
        ax = df_spec1.set_index('Time').plot(logx=True, logy=True, color=['k', 'red', 'green', 'orange', 'blue', 'm'],
                                             marker='x', markersize=7)
        returnorange = [df_spec1['O'].iloc[-2:].tolist(), df_spec1['O'].max()]
        df_spec1.to_json(datadir + f'/ConcA{brINDEX}.json')
        plt.xlabel("Time(s)")
        plt.ylabel("Concentration")
        ax.tick_params(which='major', direction='in', length=8, width=1.5)
        ax.tick_params(which='minor', direction='in', length=5, width=1.5)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        # plt.title("Concentration evolution with time")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
        plt.savefig(output_dir + '/concentration.png')  # , dpi=600)
        plt.close()
    else:
        returnorange = 'null'

    # ###############################################################################
    """
    occurrence freq and residence time
    """
    if Freq:
        #print('concentration done \n entering Freq')
        with open(data_dir + '/output/tracking.yml') as f:
            dataset = yaml.full_load(f)

        list_s = dataset["Occurrence"]
        s = sum(list_s)
        main_list = [i / s for i in list_s[:]]
        main_list1 = dataset["Residence_time"]

        df1 = pd.DataFrame(main_list, index=list(range(len(main_list))))
        df2 = pd.DataFrame(main_list1, index=list(range(len(main_list1))))
        fig, (ax1, ax2) = plt.subplots(2, 1)
        df1.plot(ax=ax1, kind='bar', legend=True, figsize=(8, 5), fontsize=8)
        ax1.set_yscale('log')
        ax1.set_ylabel("Occurrence frequency", fontsize=10)
        ax1.set_xticks([])
        ax1.legend().set_visible(False)
        df2.plot(ax=ax2, kind='bar', legend=True, figsize=(8, 5), fontsize=8)
        ax2.set_xlabel("Reactions", fontsize=10)
        ax2.set_yscale('log')
        ax2.set_ylabel("Residence time", fontsize=10)
        ax2.legend().set_visible(False)
        plt.xticks(fontsize=6)
        plt.savefig(output_dir + '/map.png')  # , dpi=300)
        plt.close()
        # visualization
        #print('Freq done')

    listf = os.listdir(data_dir + '/output/')
    listf.remove('tracking.yml')
    it = OII
    # max([int(i) for i in [s.replace("list_", "") for s in listf]])
    spec_list = []
    col = []
    time_ = []
    spec_list = []
    #print('listf done')
    if option == 2:
        with open(data_dir + '/output/' + File, 'rb') as fp:
            l = pickle.load(fp)
        step_num = it
        q = -1
        col = [k for k in l[q][2]]
        t = l[q][-1]
        outputtime = t
        position = l[q][0]
        returnposition = position
        returncolor = col

        # ---------------------------- FLUX ---------------------------
        '''flux is sum over purple sites averages on y direction'''
        if Flux:
            #print('entering Flux')
            confg = list(zip(position, col))
            # y=1 is E and it should look from y=2
            counted_sei = []
            counted_organic = []
            counted_inorganic = []
            for i in range(2, ydim + 1):
                # going through layers
                current_layer = [j for j in confg if j[0][1] == i]
                # collecting SEI ingredients
                counted_sei_per_layer = [j for j in current_layer if j[1] in ['m', 'red']]
                counted_inorganic_layer = [j for j in current_layer if j[1] == 'red']
                counted_organic_layer = [j for j in current_layer if j[1] == 'm']
                counted_sei.append(len(counted_sei_per_layer))
                counted_inorganic.append(len(counted_inorganic_layer))
                counted_organic.append(len(counted_organic_layer))
            total_sei_number = sum(counted_sei)
            fraction = [i / xdim for i in counted_sei]
            total_organic_sei_number = sum(counted_organic)
            organic_fraction = [i / xdim for i in counted_organic]
            total_inorganic_sei_number = sum(counted_inorganic)
            inorganic_fraction = [i / xdim for i in counted_inorganic]
            '''plot flux'''
            plt.figure(figsize=(12, 8))
            plt.scatter(list(range(2, ydim + 1)), fraction, marker="x", c='r', s=50)
            plt.plot(list(range(2, ydim + 1)), fraction, '--')
            plt.ylabel('SEI fraction')
            plt.xlabel('y')
            plt.savefig(output_dir + '/flux.png')
            plt.close()
            # Flux1 is a sum over SEI fractions over layers
            Flux = sum(fraction)
            in_flux = sum(inorganic_fraction)
            or_flux = sum(organic_fraction)
            #print('Flux done')
        #print('entering option 2')

        #print('snap')

        plt.figure(figsize=(8, 8), constrained_layout=True)

        plt.scatter([i[0] for i in l[q][0]], [i[1] for i in l[q][0]], c=col, s=25)
        title('step = ' + str(step_num) + '   t = ' + str(t) + ' Flux: ' + str(Flux))
        plt.savefig(output_dir + '/last.png')
        plt.close()
        a, gr = rdf2(l[q], 'red', 'm', .25, 25, [25, 1], 2500)
        gr=np.array(gr)
        a=np.array(a)

        tmp = pd.DataFrame(np.array(br)[:15].reshape(1, -1),
                           columns=['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11',
                                    'e12', 'e13', 'e14', 'e15'])

        plt.figure(100)
        plt.plot(a, gr, "-bD", markersize=5)
        plt.savefig(output_dir + '/rdf.png')
        plt.close()

        #print('option 2 done')

        # ----------------------------Thickness---------------------------
        if Thickness:
            #print('entering thickness')
            thickness = []
            thickness_in = []
            for i in range(1, xdim + 1):
                sei_in = [j[0][1] for j in confg if j[0][0] == i and j[1] == 'red']
                if sei_in:
                    thickness_in.append(max(sei_in) - 1)
                else:
                    thickness_in.append(0)
                sei_ = [j[0][1] for j in confg if j[0][0] == i and j[1] in ['red', 'm']]
                if sei_:
                    thickness.append(max(sei_) - 1)
                else:
                    thickness.append(0)
                thickness_or = [a_i - b_i for a_i, b_i in zip(thickness, thickness_in)]

            thick_all = (sum(thickness) / xdim)
            thick_in = (sum(thickness_in) / xdim)
            thick_or = (sum(thickness_or) / xdim)

            #print('thickness done')

        # ----------------------------Porosity----------------------------
        # creating lattice in proper size
        # lattice_ = Lattice(xdim,ydim)
        # fing the empty sites inside the SEI
        if Porosity:
            height = int(Flux) + 1
            empties = [i[0] for i in confg if i[0][1] <= height and i[1] not in ["red", "m"]]
            porosity = len(empties) / (height * xdim)

        # ------------------------------------------------------------------------
        with open(output_dir + '/parameter_table.txt', 'w+') as test:

            try:
                test.write(
                    "dtcpredict	Flux      in-flux    or_flux      sei/all    porosity   thickness   thick_in    thick_or\n")
                fmt = "%s%17s%12s%12s%12s%12s%12s%12s%12s"
                test.write(fmt % (
                    dtcpredict[0], round(Flux, 5), round(in_flux, 5), round(or_flux, 5),
                    round(total_sei_number / (xdim * ydim), 5),
                    round(porosity, 5), round(thick_all, 5), round(thick_in, 5), round(thick_or, 5)))
                test.write("\n")
            except:
                pass

        print(
            f'flux, in, or,:{Flux}- {in_flux}- {or_flux}, {porosity}, Conc. orng: {returnorange},  time:{outputtime:.5f}')
        gc.collect()

    return brINDEX, in_flux, or_flux, porosity, returnorange, outputtime, gr



