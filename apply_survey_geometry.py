# Generate lightcone for DESI mocks
# from HOD catalogues generated by Shadab

import sys
import glob
import configparser
from itertools import product
import multiprocessing as mp

import numpy as np
from astropy.table import Table, vstack
import desimodel.footprint as foot
import desimodel.io
import h5py


def bits(ask="try"):
    "Used"
    if ask == "LC":              return 0   #(0 0 0 0 0 0)
    if ask == "downsample":      return 1   #(0 0 0 0 0 1)
    if ask == "Y5foot":          return 2   #(0 0 0 0 1 0)
    if ask == "downsample_LOP":  return 4   #(0 0 0 1 0 0)
    if ask == "Y1foot":          return 8   #(0 0 1 0 0 0)
    if ask == "Y1footbright":    return 16  #(0 1 0 0 0 0)
    print(f"You have asked for {ask}. Which does not exist. Please check bits() function in apply_survey_geometry.py.")
    os._exit(1)


def mask(nz=0, Y5=0, nz_lop=0, Y1=0, Y1BRIGHT=0):
    return nz * (2**0) + Y5 * (2**1) + nz_lop * (2**2) + Y1 * (2**3) + Y1BRIGHT * (2**4) 


def apply_footprint(ra, dec, footprint_mask):
    """ apply desi ootprint """

    bitval = 0
    # footprint_mask possibilities
    # 0 - Y5 DESI; 1, 2, 3 - Y1 DESI

    if footprint_mask == 0:
        tiles_0 = Table.read('/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv')
        mask_y5 = (tiles_0['PROGRAM'] != 'BACKUP')
        tiles = tiles_0[mask_y5]
        bitval = bits(ask="Y5foot")
    elif footprint_mask == 1:
        tiles = Table.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits')
        bitval = bits(ask="Y1foot")
    elif footprint_mask == 2:
        tiles = Table.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-BRIGHT.fits')
        bitval = bits(ask="Y1footbright")
    # elif footprint_mask == 3:
    #     tiles_dark = Table.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits')
    #     tiles_bright = Table.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-BRIGHT.fits')
    #     tiles = vstack([tiles_dark, tiles_bright])
    #     bitval = bits(ask="Y1foot")
    else:
        print("ERROR: Wrong footprint.", flush=True)
        os._exit(1)

    point = foot.is_point_in_desi(tiles, ra, dec)
    idx   = np.where(point)

    print("FOOTPRINT: Selected {} out of {} galaxies.".format(len(idx[0]), len(ra)), flush=True)

    newbits = np.zeros(len(ra), dtype=np.int32)
    newbits[idx] = bitval

    return newbits


class SurveyGeometry():
    def __init__(self, config_file, args, galtype=None):
        config     = configparser.ConfigParser()
        config.read(config_file)
        self.config = config

        self.box_length =  config.getint('sim', 'box_length')
        self.zmin       =  config.getfloat('sim', 'zmin')
        self.zmax       =  config.getfloat('sim', 'zmax')

        self.galtype = galtype

        self.tracer_id = 0

        self.mock_random_ic = args.mock_random_ic
        if self.mock_random_ic is None:
            self.mock_random_ic = config.get('sim', 'mock_random_ic')

        if self.mock_random_ic != "ic":
            if galtype in ("LRG", "LRG_main"):
                self.tracer_id = 0
            elif galtype == "ELG":
                self.tracer_id = 1
            elif galtype == "QSO":
                self.tracer_id = 2

            print(f"INFO: {self.galtype} with {self.tracer_id} ID")


    def get_nz(self, z_cat, ask=None):
        ''' The function where the n(z) is read
        and the NZ column is computed for the given
        redshifts.
        '''
        config = self.config
        z, nz = np.loadtxt(config["nz"][ask], usecols=(config.getint("nz", "col_z"), config.getint("nz", "col_nz")), unpack=True)

        z_n = z
        nz_n = (nz / ( 1 - config.getfloat('failurerate', ask) )) * 1.0

        np.savetxt(config["nz"][ask + "_red_cor"], np.array([z_n, nz_n]).T)

        return np.interp(z_cat, z_n, nz_n, left=0, right=0)


    def downsample_aux(self, z_cat, ran, n_mean, ask=None):
        """ downsample galaxies following n(z) model specified in galtype"""

        nz = self.get_nz(z_cat, ask=ask)

        # downsample
        nz_selected = ran < nz / n_mean
        n = nz / n_mean		
        idx         = np.where(nz_selected)
        print("DOWNSAMPLE: Selected {} out of {} galaxies.".format(len(idx[0]), len(z_cat)), flush=True)

        bitval = bits(ask=ask)

        newbits = np.zeros(len(z_cat), dtype=np.int32)
        newbits[idx] = bitval
        return newbits, nz
        
    def downsample(self, z_cat, n_mean):
        """ downsample galaxies following n(z) model specified in galtype"""

        ran_i     = np.random.rand(len(z_cat))
        outbits = []

        if self.galtype == "LRG":
            outbits     , _ = self.downsample_aux(z_cat, ran_i, n_mean, ask="downsample")			
            ran = [ran_i]

        elif self.galtype == "ELG":
            newbits, nz         = self.downsample_aux(z_cat, ran_i, n_mean, ask="downsample")
            ran_n               = np.random.rand(len(z_cat))
            ran_n[newbits == 0] = np.inf
            newbits_LOP, _      = self.downsample_aux(z_cat, ran_n, 1 , ask="downsample_LOP")
            
            outbits = np.bitwise_or(newbits, newbits_LOP)
            ran = [ran_i, ran_n]
        
        elif self.galtype == "QSO":	
            outbits, _ = self.downsample_aux(z_cat, ran_i, n_mean, ask="downsample")
            ran = [ran_i]
        
        else:
            print("Wrong galaxy type.")
            os._exit(1)

        return outbits, ran
        

    def generate_shell(self, args):
        infile, footprint_mask, todo = args
        print(f"INFO: Read {infile}")

        f = h5py.File(infile, 'r+')
        n_mean = f.attrs["NGAL"] / (f.attrs["BOX_LENGTH"]**3)

        shellnum = f.attrs["SHELLNUM"]
        cat_seed = f.attrs["CAT_SEED"]

        unique_seed = self.tracer_id * 500500 + 250 * cat_seed + shellnum
        print("INFO: UNIQUE SEED:", unique_seed, flush=True)
        np.random.seed(unique_seed)

        data = f['galaxy']
        ra = data['RA'][()]
        dec = data['DEC'][()]
        z_cosmo = data['Z_COSMO'][()]

        foot_bit_0 = apply_footprint(ra, dec, 0)
        foot_bit_1 = apply_footprint(ra, dec, 1)
        
        if self.mock_random_ic != "ic":
            down_bit, ran_arr = self.downsample(z_cosmo, n_mean)
            out_arr = np.bitwise_or(np.bitwise_or(foot_bit_0, foot_bit_1), down_bit)

        else:
            foot_bit_2 = apply_footprint(ra, dec, 2)
            out_arr = np.bitwise_or(np.bitwise_or(foot_bit_0, foot_bit_1), foot_bit_2)
        
        
        out_arr = out_arr.astype(np.int32)

        if "STATUS" in data.keys():
            print("WARNING: STATUS EXISTS. New STATUS has not been written.")
        else:
            f.create_dataset('galaxy/STATUS', data=out_arr,  dtype=np.int32)

        if self.mock_random_ic != "ic":
            if "RAN_NUM_0_1" in data.keys():
                print("WARNING: RAN_NUM_0_1 EXISTS. New RAN_NUM_0_1 has not been written.")
            else:
                f.create_dataset('galaxy/RAN_NUM_0_1', data=ran_arr[0], dtype=np.float32)
                if self.galtype == "ELG":
                    f.create_dataset('galaxy/RAN_NUM_0_1_LOP', data=ran_arr[1], dtype=np.float32)

        f.close()

    def shell(self, path_instance, nproc=5, footprint_mask=0, todo=1):

        infiles = glob.glob(path_instance.shells_out_path + "/*.hdf5")

        args = product(infiles, [footprint_mask], [todo])

        if nproc > len(infiles):
            nproc = len(infiles)

        with mp.Pool(processes=nproc) as pool:
            pool.map_async(self.generate_shell, args)

            pool.close()
            pool.join()

    def shell_series(self, path_instance, footprint_mask=0, todo=1):

        infiles = glob.glob(path_instance.shells_out_path + "/*.hdf5")
        print(infiles)
        for file_ in infiles:
            args = [file_, footprint_mask, todo]
            # args = [infiles[1], footprint_mask, todo]
            self.generate_shell(args)
