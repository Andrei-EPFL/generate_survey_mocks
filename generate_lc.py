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
import camb
import numpy as np
import healpy as hp
import numexpr as ne
from astropy.io import fits
#from desimodel.io import fits
from astropy.table import Table, vstack
import desimodel.footprint as foot
import desimodel.io
import h5py
import fitsio

		
def write_catalog_fits(out_file_beg, px, py, pz, ra, dec, z_cosmo, z_rsd, nz):
	##Writing out the output fits file
	out_file_tmp = out_file_beg + "_tmp.fits"
	out_file 	 = out_file_beg + ".fits"

	c1 = fits.Column(name='RA'     , array=ra    , format='E')
	c2 = fits.Column(name='DEC'    , array=dec   , format='E')
	c3 = fits.Column(name='Z'      , array=z_rsd , format='D')
	c4 = fits.Column(name='Z_COSMO', array=z_cosmo     , format='D')
	c5 = fits.Column(name='DZ_RSD' , array=z_rsd - z_cosmo, format='E')
	c6 = fits.Column(name='NZ'     , array=nz    , format='E')
	c7 = fits.Column(name='X0'      , array=px , format='D')
	c8 = fits.Column(name='Y0'      , array=py , format='D')
	c9 = fits.Column(name='Z0'      , array=pz , format='D')

	hdu             = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9])
	hdr             = fits.Header()
	primary_hdu     = fits.PrimaryHDU(header=hdr)
	hdul            = fits.HDUList([primary_hdu, hdu])

	start = time.time()
	hdul.writeto(out_file_tmp, overwrite=True)
	print("TIME: fits, It took {} seconds to write the file.".format(time.time()-start))
	
	os.rename(out_file_tmp, out_file)


def tp2rd(tht, phi):
	""" convert theta,phi to ra/dec """
	ra  = phi/np.pi*180.0
	dec = -1*(tht/np.pi*180.0-90.0)
	return ra, dec

class Paths_LC():
	def __init__(self, config_file, args, input_name, shells_path):
		config     = configparser.ConfigParser()
		config.read(config_file)

		self.dir_out                 = args.dir_out
		self.input_name              = input_name
		self.shells_path 			 = shells_path
		self.dir_gcat                = args.dir_gcat

		if self.dir_out is None:
			self.dir_out       =  config.get('dir', 'dir_out')
		if self.input_name is None:
			self.input_name    =  config.get('dir', 'input_name')
		if self.shells_path is None:
			self.shells_path   =  config.get('dir', 'shells_path')
		if self.dir_gcat is None:
			self.dir_gcat      =  config.get('dir', 'dir_gcat')
	
		self.input_file = self.dir_gcat + self.input_name
		self.shells_out_path = self.create_outpath()
		self.begin_out_shell = self.shells_out_path + self.input_name[:-4]

	def create_outpath(self):	
		start = time.time()
		out_path = self.dir_out + "/"+ self.shells_path
		print(out_path)
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		print("TIME: It took {} seconds to create the dir.".format(time.time()-start))
		return out_path

class LC():
	def __init__(self, config_file, args):
		config     = configparser.ConfigParser()
		config.read(config_file)

		self.file_camb      =  config.get('dir', 'file_camb')
		self.boxL           =  config.getint('sim', 'boxL')
		self.shellwidth     =  config.getint('sim', 'shellwidth')
		self.zmin       =  config.getfloat('sim', 'zmin')
		self.zmax       =  config.getfloat('sim', 'zmax')
		self.origin  = [0, 0, 0]
		self.clight  = 299792458.

		self.h, self.results = self.run_camb()
		# self.f_distance2redshift = self.interpolate_redshift_distance()
		file_alist     =  config.get('dir','file_alist')
		self.alist = np.loadtxt(file_alist)

	def run_camb(self):
		#Load all parameters from camb file 
		start = time.time()
		pars = camb.read_ini(self.file_camb)
		h    = pars.h
		pars.set_for_lmax(2000, lens_potential_accuracy=3)
		pars.set_matter_power(redshifts=[0.], kmax=200.0)
		pars.NonLinearModel.set_params(halofit_version='takahashi')
		camb.set_feedback_level(level=100)
		results   = camb.get_results(pars)
		print("TIME: It took {} seconds to run the CAMB part.".format(time.time()-start))
		return h, results

	def interpolate_redshift_distance(self):
		z_array = np.linspace(0, 5, 25001)

		distance = self.results.comoving_radial_distance(z_array)
		return interp1d(distance, z_array, kind="cubic")

	def checkslicehit(self, chilow, chihigh, xx, yy, zz):
		""" pre-select so that we're not loading non-intersecting blocks """
		boxL = self.boxL
		origin = self.origin
		bvx=np.array([0, boxL, boxL,   0,   0, boxL, boxL,   0])
		bvy=np.array([0,    0, boxL, boxL,   0,   0, boxL, boxL])
		bvz=np.array([0,    0,   0,   0, boxL, boxL, boxL, boxL])

		boo = 0
		r   = np.zeros(8)
		for i in range(0, 8):
			sx  = (bvx[i] - origin[0] + boxL * xx);
			sy  = (bvy[i] - origin[1] + boxL * yy);
			sz  = (bvz[i] - origin[2] + boxL * zz);
			r[i]= np.sqrt(sx * sx + sy * sy + sz * sz)
		if chihigh<np.min(r):
			boo=boo+1
		if chilow>np.max(r):
			boo=boo+1
		#print(chilow,chihigh,np.min(r),np.max(r))
		if (boo==0):
			return True
		else:
			return False

	def convert_xyz2rdz(self, data, preffix, chilow, chiupp):
		""" Generates and saves a single lightcone shell """		
		clight = self.clight
		boxL = self.boxL
		origin = self.origin

		ntiles = int(np.ceil(chiupp/boxL))
		print(preffix + "tiling [%dx%dx%d]"%(2*ntiles,2*ntiles,2*ntiles))
		print(preffix + 'Generating map for halos in the range [%3.f - %.3f Mpc/h]'%(chilow,chiupp))
		
		px    = data['x']
		py    = data['y']
		pz    =	data['z']
		vx    = data['vx']
		vy    = data['vy']
		vz    = data['vz']
		
		ngalbox=len(px)
		print(preffix + "using %d halos"%len(px))
		
		#-------------------------------------------------------------------

		totra   = np.array([])
		totdec  = np.array([])
		totz    = np.array([])
		totdz   = np.array([])
		totvlos = np.array([])
		totpx   = np.array([])
		totpy   = np.array([])
		totpz   = np.array([])
		
		for xx in range(-ntiles,ntiles):
			for yy in range(-ntiles,ntiles):
				for zz in range(-ntiles,ntiles):

					slicehit = self.checkslicehit(chilow,chiupp,xx,yy,zz)             # Check if box intersects with shell

					if slicehit==True:

						sx  = ne.evaluate("px -%d + boxL * xx"%origin[0])
						sy  = ne.evaluate("py -%d + boxL * yy"%origin[1])
						sz  = ne.evaluate("pz -%d + boxL * zz"%origin[2])
						r   = ne.evaluate("sqrt(sx*sx + sy*sy + sz*sz)")
						zi  = self.results.redshift_at_comoving_radial_distance(r/self.h) # interpolated distance from position
						# zi  = self.f_distance2redshift(r/self.h) # interpolated distance from position
						idx = np.where((r>chilow) & (r<chiupp))[0]              # only select halos that are within the shell

						if idx.size!=0:
							ux=sx[idx]/r[idx]
							uy=sy[idx]/r[idx]
							uz=sz[idx]/r[idx]
							qx=vx[idx]*1000.
							qy=vy[idx]*1000.
							qz=vz[idx]*1000.
							zp=zi[idx]
							pxtmp = px[idx]
							pytmp = py[idx]
							pztmp = pz[idx]
							tht,phi = hp.vec2ang(np.c_[ux,uy,uz])
							ra,dec  = tp2rd(tht,phi)
							vlos    = ne.evaluate("qx*ux + qy*uy + qz*uz")
							dz      = ne.evaluate("(vlos/clight)*(1+zp)")

							totpx   = np.append(totpx,pxtmp)
							totpy   = np.append(totpy,pytmp)
							totpz   = np.append(totpz,pztmp)
							totra   = np.append(totra,ra)
							totdec  = np.append(totdec,dec)
							totz    = np.append(totz,zp)
							totdz   = np.append(totdz,dz)
							totvlos = np.append(totvlos,vlos/1000.) # to convert back to km/s
		
		return totpx, totpy, totpz, totra, totdec, totz, totz + totdz, ngalbox #, totdz, totvlos

	def getnearestsnap(self, zmid):
		""" get the closest snapshot """
		# zsnap  = 1/self.alist[:,1]-1.
		zsnap  = 1/self.alist[:,2]-1.
		return self.alist[np.argmin(np.abs(zsnap-zmid)),0]

	def obtain_data(self, subbox, shellnum, shellnums, snapshot, cutsky, path_instance, random):
		start = time.time()
		preffix = f"[shellnum={shellnum}; subbox={subbox}] "

		chilow = self.shellwidth*(shellnum+0)
		chiupp = self.shellwidth*(shellnum+1)
		chimid = 0.5*(chilow+chiupp)
		
		if not cutsky:
			print("LightCone")
			# zmid        = self.results.redshift_at_comoving_radial_distance(chimid / self.h)
			zmid        = self.f_distance2redshift(chimid / self.h)
			nearestsnap = int(self.getnearestsnap(zmid))

			infile = path_instance.input_file.format(nearestsnap, subbox)
	
		else:
			print("Cutsky")
			infile = path_instance.input_file.format(snapshot, subbox)

		print(infile)
		
		try:
			hdul = fits.open(infile)
			data = hdul[1].data
			hdul.close()
			print(f"The size of the data is:", len(data["x"]))

		except IOError:
			print(preffix + f"WARNING: Couldn't open {infile}.", file=sys.stderr)
			sys.exit()
		
		current, peak_ = tracemalloc.get_traced_memory()
		print(f"Current memory usage is {current / 10**6}MB; Peak was {peak_ / 10**6}MB")
		print("TIME: It took {} seconds to read the fits file.".format(time.time()-start))

		if random != None:
			print("ATTENTION: Compute random catalogs!")
			np.random.seed(int(random + 100*subbox + 10000*shellnum))
			data_r = {}
			length = len(data["x"])
			data_r["x"]    = np.random.uniform(low=0, high=self.boxL, size=length) #self.d[idx,0]
			data_r["y"]    = np.random.uniform(low=0, high=self.boxL, size=length) #self.d[idx,1]
			data_r["z"]    = np.random.uniform(low=0, high=self.boxL, size=length) #self.d[idx,2]
			data_r["vx"] 	= np.zeros(length)
			data_r["vy"] 	= np.zeros(length)
			data_r["vz"] 	= np.zeros(length)
			return data_r, preffix, chilow, chiupp

		return data, preffix, chilow, chiupp

	def generate_shell(self, subbox, i, shellnum, shellnums, snapshot, cutsky, path_instance, random, return_dict):
		start_time = time.time()

		### Read Data
		data, preffix, chilow, chiupp = self.obtain_data(subbox, shellnum, shellnums, snapshot, cutsky, path_instance, random)
		
		### Convert XYZ to RA DEC Z
		px0, py0, pz0, ra0, dec0, zz0, zz_rsd0, ngalbox = self.convert_xyz2rdz(data, preffix, chilow, chiupp)
		n_mean = ngalbox/(1.* self.boxL**3)
		print("The size of the LC is: ", len(px0))
		shell_subbox_dict = {"px0": px0, "py0": py0, "pz0": pz0, "ra0": ra0, "dec0": dec0, "zz0": zz0, "zz_rsd0": zz_rsd0}
		return_dict[subbox] = shell_subbox_dict
		return_dict["NGAL" + str(subbox)] = ngalbox 
		return_dict["LC" + str(subbox)] = len(px0)
		print("TIME: It took {} seconds to process the {} subbox: {}/{} shell.".format(time.time()-start_time, subbox, i+1, len(shellnums)))			

	def compute_shellnums(self):
		start = time.time()
		shellnum_min = int(self.results.comoving_radial_distance(self.zmin)*self.h // self.shellwidth)
		shellnum_max = int(self.results.comoving_radial_distance(self.zmax)*self.h // self.shellwidth + 1)
		shellnums = list(range(shellnum_min, shellnum_max+1))
		print(f"INFO: There are {len(shellnums)} shells.")
		print("TIME: It took {} seconds to run compute the shellnums.".format(time.time()-start))
		return shellnums

	def generate_shells(self, path_instance, snapshot=999, cutsky=True, nproc=5, Nsubboxes=27, random=None):
		jobs = []
		ne.set_num_threads(4)
		manager = mp.Manager()
		
		shellnums = self.compute_shellnums()
		for i, shellnum in enumerate(shellnums):
			out_file_beg = path_instance.begin_out_shell + "_shell_" + str(shellnum)

			chilow = self.shellwidth*(shellnum+0)
			chiupp = self.shellwidth*(shellnum+1)
			chimid = 0.5*(chilow+chiupp)
			
			if not cutsky:
				print("LightCone")
				# zmid        = self.results.redshift_at_comoving_radial_distance(chimid / self.h)
				zmid        = self.f_distance2redshift(chimid / self.h)
				nearestsnap = int(self.getnearestsnap(zmid))
				snapshot = nearestsnap

			out_file_beg = out_file_beg.format(snapshot, "all")
			# Don't reprocess files already done
			if os.path.isfile(out_file_beg+".h5py"):
				continue
		
			return_dict = manager.dict()
			counter = 0
			for subbox in range(Nsubboxes):
				p = mp.Process(target=self.generate_shell, args=(subbox, i, shellnum, shellnums, snapshot, cutsky, path_instance, random, return_dict))
				jobs.append(p)
				p.start()
				counter = counter + 1
				if (counter == nproc) or (subbox == Nsubboxes - 1):
					for proc in jobs:
						proc.join()
					counter = 0
					jobs = []

			# counter_LC = 0
			# for subbox in range(Nsubboxes):
			# 	counter_LC += return_dict["LC" + str(subbox)]

			ra0_array = np.empty(0)
			dec0_array = np.empty(0)
			zz0_array = np.empty(0)
			zz_rsd0_array = np.empty(0)
			px0_array = np.empty(0)
			py0_array = np.empty(0)
			pz0_array = np.empty(0)

			counter_NGAL = 0
			for subbox in range(Nsubboxes):
				counter_NGAL += return_dict["NGAL" + str(subbox)]
				shell_subbox_dict = return_dict[subbox]
				ra0_array = np.concatenate((ra0_array, shell_subbox_dict["ra0"]))
				dec0_array = np.concatenate((dec0_array, shell_subbox_dict["dec0"]))
				zz0_array = np.concatenate((zz0_array, shell_subbox_dict["zz0"]))
				zz_rsd0_array = np.concatenate((zz_rsd0_array, shell_subbox_dict["zz_rsd0"]))
				px0_array = np.concatenate((px0_array, shell_subbox_dict["px0"]))
				py0_array = np.concatenate((py0_array, shell_subbox_dict["py0"]))
				pz0_array = np.concatenate((pz0_array, shell_subbox_dict["pz0"]))
			
			
			start = time.time()
			out_file_tmp = out_file_beg + "_tmp.h5py"
			out_file 	 = out_file_beg + ".h5py"

			with h5py.File(out_file_tmp, 'w') as ff:
				ff.create_group('galaxy')
				ff.create_dataset('galaxy/RA',      data=ra0_array,    dtype=np.float32)
				ff.create_dataset('galaxy/DEC',     data=dec0_array,   dtype=np.float32)
				ff.create_dataset('galaxy/Z_RSD', 	  data=zz_rsd0_array, dtype=np.float32)
				ff.create_dataset('galaxy/Z_COSMO', data=zz0_array,     dtype=np.float32)
				ff.create_dataset('galaxy/PX', data=px0_array,     dtype=np.float32)
				ff.create_dataset('galaxy/PY', data=py0_array,     dtype=np.float32)
				ff.create_dataset('galaxy/PZ', data=pz0_array,     dtype=np.float32)
				ff.attrs['NGAL'] = counter_NGAL

			print("TIME: h5py, It took {} seconds to write the file.".format(time.time()-start), flush=True)

			os.rename(out_file_tmp, out_file)

			# out_type=[('RA',np.float64),('DEC',np.float64), ('Z_COSMO',np.float64), ('Z_RSD',np.float64), ('PX',np.float64), ('PY',np.float64), ('PZ',np.float64)]
			# outarr=np.zeros(ra0_array.size, dtype=out_type)
			# outarr['RA']=ra0_array
			# outarr['DEC']=dec0_array
			# outarr['Z_COSMO']=zz0_array
			# outarr['Z_RSD']=zz_rsd0_array
			# outarr['PX']=px0_array
			# outarr['PY']=py0_array
			# outarr['PZ']=pz0_array

			# with fitsio.FITS(out_file_beg+".fits",'rw') as fout:
			# 	fout.write(outarr)
			# 	fout[-1].write_key("NGAL", return_dict["NGAL"])

			# 	fout[-1].write_checksum()

			# fits = fitsio.FITS(out_file_beg+".fits", "rw")
			# fits.write(np.array([ra0_array, dec0_array, zz0_array, zz_rsd0_array, px0_array, py0_array, pz0_array]), names=["ra","dec","z_cosmo","z_rsd", "px", "py", "pz"])
			
## RA DEC Z ...  BIT (1, 2, 3)  

## 1)Load the file
## 2) Reapet box()

## 3) write box()
			
			# if len(ra_desi) != 0:
			# 	write_catalog_fits(out_file_beg, px_desi, py_desi, pz_desi, ra_desi, dec_desi, zz_desi, zz_rsd_desi, nzz_desi)