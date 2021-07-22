#!/usr/bin/env python

# Generate lightcone for DESI mocks
# from HOD catalogues generated by Shadab

import sys
import os
import glob
import multiprocessing as mp

import numpy as np
import fitsio


def write_file(path, outpath, file_):
    print(file_)
    fits = fitsio.FITS(file_)	
    data = fits[0].read()
    # print(data)
    # print(data.shape)
    out_type=[('RA',np.float64),('DEC',np.float64), ('Z_COSMO',np.float64), ('Z_RSD',np.float64), ('PX',np.float64), ('PY',np.float64), ('PZ',np.float64)]
    outarr=np.zeros(data.shape[1], dtype=out_type)
    outarr['RA']=data[0]
    outarr['DEC']=data[1]
    outarr['Z_COSMO']=data[2]
    outarr['Z_RSD']=data[3]
    outarr['PX']=data[4]
    outarr['PY']=data[5]
    outarr['PZ']=data[6]

    name_ = os.path.basename(file_)

    with fitsio.FITS(path + outpath + name_,'rw') as fout:
        fout.write(outarr)
        fout[-1].write_checksum()



# path = "/global/cscratch1/sd/avariu/desi/UNIT_3GPC/sv3_v0.1_nz_largerdensity_mycosmo"
# inpath = "/cutsky_shells_sv3_nz_radecz_xyz_LRG_old/"
# outpath = "/cutsky_shells_sv3_nz_radecz_xyz_LRG/"

path = "/global/cscratch1/sd/avariu/desi/UNIT_3GPC/sv3_v0.1_nz_largerdensity_mycosmo"
inpath = "/cutsky_shells_sv3_nz_radecz_xyz_LRG_old/"

infiles = glob.glob(path + inpath + "*")
counter = 0
nproc = 15
jobs = []
for file_ in infiles:


    p = mp.Process(target=write_file, args=(path, outpath, file_))
    jobs.append(p)
    p.start()
    counter = counter + 1
    if (counter == nproc):
        for proc in jobs:
            proc.join()
        counter = 0
        jobs = []

