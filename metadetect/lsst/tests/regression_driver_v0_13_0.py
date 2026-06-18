"""
Driver for the bit-identical regression test against tag 0.13.0, see
test_regression_vs_0_13_0.py

This script is run in subprocesses with PYTHONPATH pointing at different
checkouts of metadetect, so it must only use API that exists both at tag
0.13.0 and on the current branch.  In particular the deblender is selected
through config['deblender'] at 0.13.0 but through the deblender keyword on
the current branch; we detect which API is present from the signature.

usage: python regression_driver_v0_13_0.py outfile deblender
"""
import inspect
import sys

import numpy as np

SIM_SEED = 116
MDET_SEED = 55
BANDS = ['r', 'i']
COADD_DIM = 251


def make_lsst_sim(seed, mag=20, hlr=0.5):
    import descwl_shear_sims

    rng = np.random.RandomState(seed=seed)

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=COADD_DIM,
        buff=20,
        layout='grid',
        mag=mag,
        hlr=hlr,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    return descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=COADD_DIM,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=BANDS,
    )


def run(deblender):
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp
    from metadetect.lsst.metadetect import run_metadetect
    from metadetect.lsst import util

    sim_data = make_lsst_sim(SIM_SEED)

    rng = np.random.RandomState(seed=MDET_SEED)
    coadd_data_list = [
        make_coadd_nowarp(
            exp=sim_data['band_data'][band][0],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,
        )
        for band in BANDS
    ]
    data = util.extract_multiband_coadd_data(coadd_data_list)

    kwargs = {}
    config = {}
    if 'deblender' in inspect.signature(run_metadetect).parameters:
        kwargs['deblender'] = deblender
    else:
        config['deblender'] = deblender

    return run_metadetect(rng=rng, config=config, **kwargs, **data)


def main(outfile, deblender):
    import metadetect

    res = run(deblender)

    out = {key: res[key] for key in res}
    out['__metadetect_file__'] = np.array(metadetect.__file__)
    np.savez(outfile, **out)
    print('wrote', outfile, 'using metadetect from', metadetect.__file__)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
