#!/usr/bin/env python

# Generate lightcone for DESI mocks
# from HOD catalogues generated by Shadab

import sys
import os
import glob
import tracemalloc
import time
import configparser
import warnings
import pandas as pd
import multiprocessing as mp

from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
from itertools import product

from astropy.table import Table, vstack
import desimodel.footprint as foot
import desimodel.io
import h5py


def bits(ask="try"):
	if ask == "LC": return 0  #(0 0 0)
	if ask == "downsample": return 1 #(0 0 1)
	if ask == "entireDESIfoot": return 2 #(0 1 0)
	if ask == "SV3foot": return 4 #(1 0 0)
	sys.exit()

def combine_shells(path_instance):
	start_time = time.time()
	
	# files = glob.glob(path_instance.begin_out_shell + "_*")
	print(path_instance.begin_out_shell.format("*", "*"))
	files = glob.glob(path_instance.begin_out_shell.format("*", "*") + "*")
		
	print(f"There are {len(files)} files")

	out_file_beg = path_instance.dir_out + path_instance.input_name[:-4]

	tot_ra = np.array([])
	tot_dec = np.array([])
	tot_zz = np.array([])
	tot_zz_rsd = np.array([])
	tot_nz = np.array([])

	shells = []
	for i, file_ in enumerate(files):
		start_time = time.time()
		shells.append(Table.read(file_))
	
	joint_table = vstack(shells)
	joint_table.write(out_file_beg + path_instance.end_combined_file)
	print("TIME: It took {} seconds to combine all shells.".format(time.time()-start_time))
		
def nz_oneperc(zz, nz_pars):
	if nz_pars['galtype'] == "lrg":
		z, nz = np.loadtxt("./nz_sv3/sm_LRG_mycosmo_ev2.1.dat", usecols=(0,1), unpack=True)
		failurerate = 0.02
	elif nz_pars['galtype'] == "elg":
		z, nz = np.loadtxt("./nz_sv3/sm_ELG_mycosmo_ev2.1.dat", usecols=(0,1), unpack=True)
		failurerate = 0.25
	elif nz_pars['galtype'] == "qso":
		z, nz = np.loadtxt("./nz_sv3/sm_QSO_mycosmo_ev2.1.dat", usecols=(0,1), unpack=True)
		failurerate = 0.37
	elif nz_pars['galtype'] == "bgs":
		z, nz = np.loadtxt("./nz_sv3/sm_BGS_mycosmo_v1.0.dat", usecols=(0,1), unpack=True)
	else:
		raise RuntimeError("Unknown galaxy type.")

	nz = (nz / ( 1 - failurerate )) * 1.0
	# nleft = int((np.min(z)-0.1) / nz_pars['zmin'])
	# zleft = np.linspace(nz_pars['zmin'], np.min(z), nleft + 1)
	# nzleft = np.zeros(nleft + 1)

	# nright = int((nz_pars['zmax'] +0.1)/ np.max(z))
	# zright = np.linspace(np.max(z), nz_pars['zmax']+0.1, nright + 1)
	# nzright = np.zeros(nright + 1)

	# nz_n = np.concatenate((nz, nzright))
	# z_n = np.concatenate((z, zright))

	nz_n = nz
	z_n = z
	print(np.min(zz), np.max(zz))
	print(np.min(z_n), np.max(z_n))
	print(nz_pars['zmin'], nz_pars['zmax'])

	# nzint = interp1d(z_n, nz_n, fill_value=(0,0))

	np.savetxt("./nz_sv3/" + nz_pars['galtype'] + "_sm_nz_mycosmo_redcor_ev2.1.txt", np.array([z_n, nz_n]).T)
	# return nzint(zz)
	return np.interp(zz, z_n, nz_n, left=0, right=0)

def downsample(file_, boxL, nz_pars, ngalbox, z_cosmo):
	""" downsample galaxies following n(z) model specified in nz_pars """

	n_mean = ngalbox/(boxL**3)
	nz         = nz_oneperc(z_cosmo, nz_pars)

	# downsample
	ran         = np.random.rand(len(z_cosmo))
	nz_selected = (ran<nz/n_mean)
	idx         = np.where(nz_selected)

	print("Selected {} out of {} galaxies.".format(len(idx[0]), len(z_cosmo)))

	bitval = bits(ask="downsample")
	
	newbits = np.zeros(len(z_cosmo), dtype=np.int32)
	newbits[idx] = bitval

	return newbits

def apply_footprint(file_, ra, dec, fullfootprint):
	""" apply desi footprint """
	
	bitval = 0
	# fullfootprint possibilities
	# 0 - Full DESI, 1 - old 1%, 2 - new 1%

	if fullfootprint == 0:
		tiles = desimodel.io.load_tiles()
		point = foot.is_point_in_desi(tiles, ra, dec)
		bitval = bits(ask="entireDESIfoot")
	elif fullfootprint == 2:
		tiles = Table.read('/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-sv3.ecsv')
		point = foot.is_point_in_desi(tiles, ra, dec)
		bitval = bits(ask="SV3foot")
	else:
		print("ERROR: Wrong footprint.")
		exit()
	
	idx   = np.where(point)
	print("Selected {} out of {} galaxies.".format(len(idx[0]), len(ra)))
	
	newbits = np.zeros(len(ra), dtype=np.int32)
	newbits[idx] = bitval

	return newbits

def generate_shell(args):
	file_, boxL, nz_pars, fullfootprint, todo = args
	print(file_)
	
	f = h5py.File(file_, 'r+')
	data = f['galaxy']
	ra = data['RA'][()]
	dec = data['DEC'][()]
	z_cosmo = data['Z_COSMO'][()]
			
	start = time.time()
	
	if todo == 0:
		out_arr = apply_footprint(file_, ra, dec, fullfootprint)
	elif todo == 1:
		out_arr = downsample(file_, boxL, nz_pars, f.attrs["NGAL"], z_cosmo)
	elif todo == 2:
		foot_bit = apply_footprint(file_, ra, dec, fullfootprint)
		down_bit = downsample(file_, boxL, nz_pars, f.attrs["NGAL"], z_cosmo)
		out_arr = np.bitwise_or(foot_bit, down_bit)

	# out_type=[('STATUS', np.int32)]
	out_arr = out_arr.astype(np.int32)
	print("TIME: It took {} seconds to get the bits.".format(time.time()-start))
	
	start = time.time()

	if "STATUS" in data.keys():
		print("STATUS EXISTS")
		status = data["STATUS"][:]

		new_arr = np.bitwise_or(status, out_arr)
		new_arr = new_arr.astype(np.int32)
		print(np.unique(new_arr))
		data["STATUS"][:] = new_arr[:]
		# data["STATUS"][:] = out_arr[:]
	else:
		f.create_dataset('galaxy/STATUS', data=out_arr,  dtype=np.int32)

	f.close()
	print("TIME: It took {} seconds to insert the column the bits.".format(time.time()-start), flush=True)


class FOOT_NZ():
	def __init__(self, config_file, args, galtype = None):
		config     = configparser.ConfigParser()
		config.read(config_file)

		self.boxL       =  config.getint('sim', 'boxL')
		self.zmin       =  config.getfloat('sim', 'zmin')
		self.zmax       =  config.getfloat('sim', 'zmax')
		
		nz_par = dict()

		nz_par["galtype"]       = galtype
		nz_par["zmin"]          = config.getfloat(f'{galtype}', 'zmin', fallback=self.zmin)
		nz_par["zmax"]          = config.getfloat(f'{galtype}', 'zmax', fallback=self.zmax)
		nz_par["galtype_index"] = config.getint(f'{galtype}', 'sample_index')
	
		self.nz_par = nz_par
	
	def shell(self, path_instance, nproc=5, fullfootprint=0, todo=1):
		
		infiles = glob.glob(path_instance.shells_out_path + "/*.h5py")

		args = product(infiles, [self.boxL], [self.nz_par], [fullfootprint], [todo])
		# counter = 0
		# jobs = []
		# print(len(infiles))
		# for file_ in infiles:
		# 	p = mp.Process(target=generate_shell, args=(file_, self.boxL, self.nz_par, fullfootprint, todo))
		# 	jobs.append(p)
		# 	p.start()
		# 	counter = counter + 1
		# 	if (counter == nproc):
		# 		for proc in jobs:
		# 			proc.join()
		# 		counter = 0
		# 		jobs = []
		# try:


		pool = mp.Pool(processes=nproc)
	
		pool.map_async(generate_shell, args)
		
		pool.close()
		pool.join()

		# except:
			# print("Nu merge")
		# with mp.Pool(processes=nproc) as pool:
		# 	pool.map_async(generate_shell, args)
