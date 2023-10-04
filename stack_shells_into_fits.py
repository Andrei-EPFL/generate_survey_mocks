import os
import glob
import numpy as np
import h5py
import fitsio

from apply_survey_geometry import mask
from redshift_error_QSO import sample_redshift_error


def count_aux(status_tmp, ra_tmp):
    idx = np.arange(len(status_tmp))
    mask_Y5 = mask(nz=0, Y5=1)
        
    range_NGC = (ra_tmp < 303.25) & (ra_tmp > 84)
    range_SGC = ~range_NGC

    idx_Y5_TOT = idx[ ((status_tmp & (mask_Y5)) == mask_Y5) ]
    idx_Y5_NGC = idx[ ((status_tmp & (mask_Y5)) == mask_Y5) & range_NGC ]
    idx_Y5_SGC = idx[ ((status_tmp & (mask_Y5)) == mask_Y5) & range_SGC ]

    return idx_Y5_TOT, idx_Y5_NGC, idx_Y5_SGC


def count(files):
    print(len(files))

    counter_TOT = 0
    counter_NGC = 0
    counter_SGC = 0

    for i, file_ in enumerate(files):
        f = h5py.File(file_, 'r')
        data = f['galaxy']
        ra_tmp      = data['RA'][()]
        status_tmp  = data['STATUS'][()]

        idx_Y5_TOT, idx_Y5_NGC, idx_Y5_SGC = count_aux(status_tmp, ra_tmp)
        counter_NGC = counter_NGC + len(status_tmp[idx_Y5_NGC])
        counter_SGC = counter_SGC + len(status_tmp[idx_Y5_SGC])
        counter_TOT = counter_TOT + len(status_tmp[idx_Y5_TOT])

        f.close()
    return counter_TOT, counter_NGC, counter_SGC


def fill_array(output_data_array, input_data_array, columns, idx, index_i, n_mean, survey_geometry_instance):
    z_cosmo_tmp = input_data_array["Z_COSMO"][()]
    size_ = len(z_cosmo_tmp[idx])

    if "Z_RSD" in input_data_array.keys():
        z_rsd_tmp   = input_data_array["Z_RSD"][()]
    
    index_f = size_ + index_i

    for col, type_ in columns:
        if col == "NZ":
            output_data_array["NZ"][index_i: index_f]      = survey_geometry_instance.get_nz(z_cosmo_tmp[idx], ask="downsample")
        elif col == "NZ_LOP":
            output_data_array["NZ_LOP"][index_i: index_f] = survey_geometry_instance.get_nz(z_cosmo_tmp[idx], ask="downsample_LOP")
        elif col == "RAW_NZ":
            output_data_array["RAW_NZ"][index_i: index_f] = np.ones(len(z_cosmo_tmp[idx])) * n_mean
        elif col == "Z_ERR_3GAUSS":
            output_data_array["Z_ERR_3GAUSS"][index_i: index_f] = sample_redshift_error(z_rsd_tmp[idx], error_model='3gauss')
        elif col == "Z_ERR_SIG500":
            output_data_array["Z_ERR_SIG500"][index_i: index_f] = sample_redshift_error(z_rsd_tmp[idx], error_model='sig500')
        elif col == "Z":
            output_data_array["Z"][index_i: index_f] = z_rsd_tmp[idx]
        else:
            array_col = input_data_array[col][()]
            output_data_array[col][index_i: index_f] = array_col[idx]

    return index_f


def stack_shells(survey_geometry_instance, inpath="test", out_file="test", seed=0, max_seed=0, min_seed=1, mock_random_ic=None, ngc_sgc_tot=None):
    files = glob.glob(inpath + "/*hdf5")
    files.sort()

    print("INFO: The number of shells: ", len(files))
    counter_TOT, counter_NGC, counter_SGC = count(files)
    print(f"The number of tracers: TOT={counter_TOT}; NGC={counter_NGC}; SGC={counter_SGC}")

    general_columns = [('RA', 'f4'), ('DEC', 'f4'), ('Z_COSMO', 'f4'), ('STATUS', 'i4')]

    if mock_random_ic != "ic":
        general_columns += [('RAW_NZ', 'f4'), ('RAN_NUM_0_1', 'f4'),('NZ', 'f4')]

    if mock_random_ic == "mock":
        add_columns = [('Z', 'f4')]
    elif mock_random_ic == "random":
        add_columns = [('ID', 'i4')]
    elif mock_random_ic == "ic":
        add_columns = [('ONEplusDELTA', 'f4')]

    if survey_geometry_instance.galtype == "ELG":
        specific_columns = [('NZ_LOP', 'f4'), ('RAN_NUM_0_1_LOP', 'f4')]    
    elif survey_geometry_instance.galtype == "QSO":
        specific_columns =  [('Z_ERR_3GAUSS', 'f4'), ('Z_ERR_SIG500', 'f4')]
    else:
        specific_columns = []

    all_columns = general_columns + add_columns + specific_columns
    
    if ngc_sgc_tot == "TOT":
        data_fits_TOT = np.zeros(counter_TOT, dtype=all_columns)
        index_i_TOT = 0

    elif ngc_sgc_tot == "NGC_SGC":
        data_fits_NGC = np.zeros(counter_NGC, dtype=all_columns)
        index_i_NGC = 0
        
        data_fits_SGC = np.zeros(counter_SGC, dtype=all_columns)
        index_i_SGC = 0
    else:
        print("ERROR: Choose NGC_SGC or TOT for ngc_sgc_tot variable")

    for i, file_ in enumerate(files):
        print(f"File {i + 1}/{len(files)}")
        f = h5py.File(file_, 'r')
        data = f['galaxy']
        ngalbox    = f.attrs["NGAL"]
        n_mean = ngalbox / (survey_geometry_instance.box_length ** 3)
        ra_tmp      = data['RA'][()]
        status_tmp  = data['STATUS'][()]

        idx_Y5_TOT, idx_Y5_NGC, idx_Y5_SGC = count_aux(status_tmp, ra_tmp)

        if ngc_sgc_tot == "TOT":
            index_i_TOT = fill_array(data_fits_TOT, data, all_columns, idx_Y5_TOT, index_i_TOT, n_mean, survey_geometry_instance)

        elif ngc_sgc_tot == "NGC_SGC":
            index_i_NGC = fill_array(data_fits_NGC, data, all_columns, idx_Y5_NGC, index_i_NGC, n_mean, survey_geometry_instance)
            index_i_SGC = fill_array(data_fits_SGC, data, all_columns, idx_Y5_SGC, index_i_SGC, n_mean, survey_geometry_instance)


        f.close()

    hdict = {'Y5_TOT_AREA':14850.4, 'Y5_SGC_AREA':4666.5, 'Y5_NGC_AREA':10183.9}

    if ngc_sgc_tot == "TOT":
        print("Check last value of RA: ", data_fits_TOT["RA"][-1], ra_tmp[idx_Y5_TOT][-1])

        # out_file = out_file.format(phase=seed) + "_Y5_TOT.fits"

        fits = fitsio.FITS(out_file + "_tmp", "rw")
        fits.write(data_fits_TOT, header=hdict)
        fits.close()
        os.rename(out_file + "_tmp", out_file)

    elif ngc_sgc_tot == "NGC_SGC":
        print("Check last value of RA: ", data_fits_NGC["RA"][-1], ra_tmp[idx_Y5_NGC][-1])
        print("Check last value of RA: ", data_fits_SGC["RA"][-1], ra_tmp[idx_Y5_SGC][-1])

        out_file_NGC = out_file.format(phase=seed) + "_Y5_NGC.fits"
        out_file_SGC = out_file.format(phase=max_seed - seed + min_seed) + "_Y5_SGC.fits"

        fits = fitsio.FITS(out_file_NGC+"_tmp", "rw")
        fits.write(data_fits_NGC, header=hdict)
        fits.close()
        os.rename(out_file_NGC+"_tmp", out_file_NGC)

        fits = fitsio.FITS(out_file_SGC+"_tmp", "rw")
        fits.write(data_fits_SGC, header=hdict)
        fits.close()
        os.rename(out_file_SGC+"_tmp", out_file_SGC)
