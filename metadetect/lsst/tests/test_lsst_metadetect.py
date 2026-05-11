"""
test using lsst simple sim
"""
import sys
import numpy as np
import pytest

import logging
import ngmix
import metadetect
from metadetect import procflags
import metadetect.lsst.metadetect as lsst_mdet
from metadetect.lsst.metadetect import run_metadetect
from metadetect.lsst.measure import get_pgauss_fitter
from metadetect.lsst.configs import get_config
from metadetect.lsst import util
from metadetect.lsst import vis
import lsst.afw.image as afw_image

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)


def test_fit_original_psfs_mbexp_returns_per_band(monkeypatch):
    rng = np.random.RandomState(seed=117)
    bands = ['r', 'i', 'z']

    class FakeExposure:
        def getWcs(self):
            return None

        def getBBox(self):
            return None

    class FakeMbexp:
        def __init__(self, bands):
            self.bands = bands
            self.exposures = [FakeExposure() for _ in bands]

        def __iter__(self):
            return iter(self.exposures)

    class FakeJacobian:
        def copy(self):
            return self

        def set_cen(self, row, col):
            pass

    class FakePSFRunner:
        def __init__(self, fitter, guesser, ntry):
            self._results = [
                {'flags': 0, 'e': (0.1, 0.2), 'T': 1.1},
                {'flags': 1, 'e': (0.3, 0.4), 'T': 2.2},
                {'flags': 0, 'e': (0.5, 0.6), 'T': 3.3},
            ]

        def go(self, obs):
            return self._results.pop(0)

    monkeypatch.setattr(
        lsst_mdet,
        'get_integer_center',
        lambda wcs, bbox, as_double: ((0, 0), None),
    )
    monkeypatch.setattr(
        lsst_mdet,
        'get_jacobian',
        lambda exp, cen: FakeJacobian(),
    )
    monkeypatch.setattr(
        lsst_mdet.measure,
        'extract_psf_image',
        lambda exp, cen: np.zeros((3, 3)),
    )
    monkeypatch.setattr(
        lsst_mdet.ngmix,
        'Observation',
        lambda image, jacobian: object(),
    )
    monkeypatch.setattr(
        lsst_mdet.ngmix.runners,
        'PSFRunner',
        FakePSFRunner,
    )

    psf_stats = lsst_mdet.fit_original_psfs_mbexp(
        mbexp=FakeMbexp(bands),
        rng=rng,
    )

    # Check that it returns a numpy structured array
    import numpy as np
    assert isinstance(psf_stats, np.ndarray)
    assert psf_stats.dtype.names == ('e1', 'e2', 'T', 'flags')
    assert len(psf_stats) == len(bands)

    # Check values by index (bands are in order: r, i, z)
    assert psf_stats['flags'][0] == 0  # r band
    assert psf_stats['e1'][0] == 0.1
    assert psf_stats['e2'][0] == 0.2
    assert psf_stats['T'][0] == 1.1

    assert psf_stats['flags'][1] == procflags.PSF_FAILURE  # i band
    assert psf_stats['e1'][1] == -9999.0
    assert psf_stats['e2'][1] == -9999.0
    assert psf_stats['T'][1] == -9999.0

    assert psf_stats['flags'][2] == 0  # z band
    assert psf_stats['e1'][2] == 0.5
    assert psf_stats['e2'][2] == 0.6
    assert psf_stats['T'][2] == 3.3


def test_average_psf_stats():
    import numpy as np
    # Create numpy structured array as returned by fit_original_psfs_mbexp
    psf_stats = np.zeros(3, dtype=[('e1', 'f8'), ('e2', 'f8'), ('T', 'f8'), ('flags', 'i4')])
    psf_stats['e1'] = [0.1, 0.3, 0.5]  # g, r, i bands
    psf_stats['e2'] = [0.2, 0.4, 0.6]
    psf_stats['T'] = [1.0, 3.0, 5.0]
    psf_stats['flags'] = [0, 0, 0]

    wgts = [1.0, 1.0, 1.0]  # weights by index: g, r, i

    result = lsst_mdet.average_psf_stats(
        psf_stats=psf_stats,
        wgts=wgts,
    )
    assert result['flags'] == 0
    assert result['g1'] == pytest.approx(0.3)
    assert result['g2'] == pytest.approx(0.4)
    assert result['T'] == pytest.approx(3.0)

    result = lsst_mdet.average_psf_stats(
        psf_stats=psf_stats,
        wgts=wgts,
    )
    assert result['flags'] == 0
    assert result['g1'] == pytest.approx(0.4)
    assert result['g2'] == pytest.approx(0.5)
    assert result['T'] == pytest.approx(4.0)


def test_average_psf_stats_with_failure():
    import numpy as np
    # Create numpy structured array with a failure in the i band
    psf_stats = np.zeros(3, dtype=[('e1', 'f8'), ('e2', 'f8'), ('T', 'f8'), ('flags', 'i4')])
    psf_stats['e1'] = [0.3, -9999.0, 0.5]  # r, i, z bands
    psf_stats['e2'] = [0.4, -9999.0, 0.6]
    psf_stats['T'] = [3.0, -9999.0, 5.0]
    psf_stats['flags'] = [0, procflags.PSF_FAILURE, 0]

    wgts = [1.0, 1.0, 1.0]  # weights by index: r, i, z

    result = lsst_mdet.average_psf_stats(
        psf_stats=psf_stats,
        wgts=wgts,
    )
    assert result['flags'] == procflags.PSF_FAILURE
    assert result['g1'] == -9999.0
    assert result['g2'] == -9999.0
    assert result['T'] == -9999.0

    result = lsst_mdet.average_psf_stats(
        psf_stats=psf_stats,
        wgts=wgts,
    )
    assert result['flags'] == 0
    assert result['g1'] == pytest.approx(0.4)
    assert result['g2'] == pytest.approx(0.5)
    assert result['T'] == pytest.approx(4.0)


def test_average_psf_stats_missing_band():
    import numpy as np
    # Create numpy structured array with only r band
    psf_stats = np.zeros(1, dtype=[('e1', 'f8'), ('e2', 'f8'), ('T', 'f8'), ('flags', 'i4')])
    psf_stats['e1'][0] = 0.3
    psf_stats['e2'][0] = 0.4
    psf_stats['T'][0] = 3.0
    psf_stats['flags'][0] = 0

    wgts = [1.0]  # weights by index: r

    with pytest.raises(RuntimeError, match='Not all requested bands'):
        lsst_mdet.average_psf_stats(
            psf_stats=psf_stats,
            wgts=wgts,
        )


def make_lsst_sim(
    seed, mag=20, hlr=0.5, bands=None, layout='grid', psf_type='gauss',
):
    import descwl_shear_sims

    rng = np.random.RandomState(seed=seed)
    coadd_dim = 251

    if bands is None:
        bands = ['i']

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
        layout=layout,
        mag=mag,
        hlr=hlr,
    )

    # This way we get different PSFs per band for PS PSF
    sim_data = {'band_data': {}}
    for band in bands:
        if psf_type == 'ps':
            psf = descwl_shear_sims.psfs.make_ps_psf(
                rng=rng,
                dim=300,
            )
        else:
            psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type=psf_type)

        tsim_data = descwl_shear_sims.make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=coadd_dim,
            g1=0.02,
            g2=0.00,
            psf=psf,
            bands=bands,
        )
        for key in tsim_data:
            if key == 'band_data':
                sim_data['band_data'][band] = tsim_data['band_data'][band]
            else:
                sim_data[key] = tsim_data[key]
    return sim_data


def do_coadding(rng, sim_data, nowarp):
    from descwl_coadd.coadd import make_coadd
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp

    bands = list(sim_data['band_data'].keys())

    if nowarp:
        coadd_data_list = [
            make_coadd_nowarp(
                exp=sim_data['band_data'][band][0],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data['band_data'][band],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                remove_poisson=False,
            )
            for band in bands
        ]

    return util.extract_multiband_coadd_data(coadd_data_list)


@pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize("metacal_types_option", [None, "1p1m", "full"])
def test_lsst_metadetect_smoke(subtract_sky, metacal_types_option):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    if metacal_types_option is not None:
        if metacal_types_option == "1p1m":
            metacal_types = ['noshear', '1p', '1m']
            config['metacal'] = {}
        elif metacal_types_option == "full":
            metacal_types = ['noshear', '1p', '1m', '2p', '2m']
            config['metacal'] = {}
        config['metacal']['types'] = metacal_types
    else:
        metacal_types = ['noshear', '1p', '1m']

    detected = afw_image.Mask.getPlaneBitMask('DETECTED')
    res = run_metadetect(rng=rng, config=config, **data)

    # we remove the DETECTED bit
    assert np.all(res['noshear']['bmask'] & detected == 0)

    for metacal_type in metacal_types:
        assert (
            metacal_type in res.keys()
        ), f"metacal_type={metacal_type} not in res.keys()"

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res['noshear'].dtype.names

        flux_name = f'{front}_band_flux'

        for shear in metacal_types:
            # 5x5 grid
            assert res[shear].size == 25

            assert np.any(res[shear][f"{front}_flags"] == 0)
            assert np.all(res[shear]["mfrac"] == 0)

            assert len(res[shear][flux_name].shape) == len(bands)
            assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.parametrize("metacal_reconv_option", [None, "fitgauss", "gauss"])
def test_lsst_metadetect_reconv(metacal_reconv_option):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}

    if metacal_reconv_option is not None:
        config['metacal'] = {}
        config['metacal']['reconv_type'] = metacal_reconv_option

    test_config = get_config(config)

    if metacal_reconv_option is not None:
        assert test_config['metacal']['reconv_type'] == metacal_reconv_option
    else:
        assert test_config['metacal']['reconv_type'] == 'fitgauss'

    res = run_metadetect(rng=rng, config=config, **data)  # noqa


@pytest.mark.xfail
def test_lsst_metadetect_reconv_size():
    """
    This currently fails because the PSF images have no noise.  fitgauss
    will outperform gauss for noisy PSFs
    """
    rng = np.random.RandomState(seed=232)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(5520, bands=bands, psf_type='ps')
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}
    config['metacal'] = {}
    config['metacal']['reconv_type'] = 'fitgauss'
    res_fitgauss = run_metadetect(rng=rng, config=config, **data)  # noqa

    config['metacal']['reconv_type'] = 'gauss'
    res_gauss = run_metadetect(rng=rng, config=config, **data)  # noqa

    mT_fitgauss = res_fitgauss['noshear']['gauss_psf_T'].mean()
    mT_gauss = res_gauss['noshear']['gauss_psf_T'].mean()

    fwhm_fitgauss = ngmix.moments.T_to_fwhm(mT_fitgauss)
    fwhm_gauss = ngmix.moments.T_to_fwhm(mT_gauss)

    assert fwhm_fitgauss < fwhm_gauss, (
        'expected fitgauss fwhm < gauss fwhm, '
        f'got {fwhm_fitgauss} > {fwhm_gauss}'
    )


def test_lsst_metadetect_shear_bands_missing():
    rng = np.random.RandomState(seed=116)

    bands = ['g', 'r', 'i', 'z']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)
    config = {"shear_bands": ["r", "Y"]}
    with pytest.raises(RuntimeError) as e:
        run_metadetect(rng=rng, config=config, **data)

    assert "'r', 'Y'" in str(e.value)


def test_lsst_metadetect_shear_bands():
    rng = np.random.RandomState(seed=116)

    bands = ['g', 'r', 'i', 'z']
    shear_bands = ['r', 'z']

    nband = len(bands)
    sim_data = make_lsst_sim(116, bands=bands, psf_type='ps')
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {"shear_bands": shear_bands}
    metacal_types = ['noshear', '1p', '1m']

    detected = afw_image.Mask.getPlaneBitMask('DETECTED')
    res = run_metadetect(rng=rng, config=config, **data)
    diagnostics = res.pop('_diagnostics')

    # we remove the DETECTED bit
    assert np.all(res['noshear']['bmask'] & detected == 0)

    for metacal_type in metacal_types:
        assert (
            metacal_type in res.keys()
        ), f"metacal_type={metacal_type} not in res.keys()"

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res['noshear'].dtype.names

        flux_name = f'{front}_band_flux'

        for shear in metacal_types:
            # 5x5 grid
            assert res[shear].size == 25

            assert np.any(res[shear][f"{front}_flags"] == 0)
            assert np.all(res[shear]["mfrac"] == 0)
            assert res[shear][flux_name].shape == (25, nband)

    perband = diagnostics['psf_stats_perband']
    assert perband.size == len(bands)

    wgts = diagnostics['weight_perband']
    assert len(wgts) == len(bands)
    assert all(band in bands for band in wgts)
    for iband, band in enumerate(bands):
        assert perband['band'][iband] == band
        if band in shear_bands:
            assert wgts[band] > 0
        else:
            assert wgts[band] == 0

    wgts_list = list(wgts.values())
    psf_stats = diagnostics['psf_stats_average']
    T = np.average(perband['T'], weights=wgts_list)
    e1 = np.average(perband['e1'], weights=wgts_list)
    e2 = np.average(perband['e2'], weights=wgts_list)
    g1, g2 = ngmix.shape.e1e2_to_g1g2(e1, e2)

    assert psf_stats['T'] == T
    assert psf_stats['e1'] == e1
    assert psf_stats['e2'] == e2
    assert psf_stats['g1'] == g1
    assert psf_stats['g2'] == g2

    np.testing.assert_allclose(res['noshear']['psfrec_g'][:, 0], g1)
    np.testing.assert_allclose(res['noshear']['psfrec_g'][:, 1], g2)
    np.testing.assert_allclose(res['noshear']['psfrec_T'], T)

    for shear in metacal_types:
        assert np.all(res[shear]["shear_bands"] == np.array([["13"]]))
        # g and i band should be all NaNs for gauss
        assert np.all(np.isnan(res[shear]["gauss_band_flux"][:, 0]))
        assert np.all(np.isnan(res[shear]["gauss_band_flux"][:, 2]))
        # rest should be finite
        assert np.all(np.isfinite(res[shear]["gauss_band_flux"][:, 1]))
        assert np.all(np.isfinite(res[shear]["gauss_band_flux"][:, 3]))
        assert np.all(np.isfinite(res[shear]["pgauss_band_flux"]))


def test_lsst_metadetect_pgauss():
    rng = np.random.RandomState(seed=882)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    fwhm = 2.0
    config = {
        'pgauss': {
            'fwhm': fwhm,
        }
    }

    config = get_config(config)
    fitter = get_pgauss_fitter(pgauss_config=config['pgauss'])
    assert fitter.fwhm == fwhm

    res = run_metadetect(rng=rng, config=config, **data)

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res['noshear'].dtype.names

        flux_name = f'{front}_band_flux'

        for shear in ('noshear', '1p', '1m'):
            # 5x5 grid
            assert res[shear].size == 25

            assert np.any(res[shear][f"{front}_flags"] == 0)
            assert np.all(res[shear]["mfrac"] == 0)

            assert len(res[shear][flux_name].shape) == len(bands)
            assert len(res[shear][flux_name][0]) == len(bands)


def test_lsst_metadetect_fullcoadd_smoke():
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(882, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

    config = {}
    res = run_metadetect(config=config, rng=rng, **data)

    front = 'gauss'
    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear][f"{front}_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


def test_lsst_zero_weights(show=False):
    """
    At time of writing, DM stack will still detect in regions with inf
    variance.  Test this continues to be true.

    However, we don't have detections in BRIGHT, see test
    test_lsst_masked_as_bright
    """
    nobj = []
    seed = 55
    for do_zero in [False, True]:
        rng = np.random.RandomState(seed)
        sim_data = make_lsst_sim(seed, mag=23)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        if do_zero:
            data['mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['noise_mbexp']['i'].variance.array[50:100, 50:100] = np.inf

            if show:
                import matplotlib.pyplot as mplt
                fig, axs = mplt.subplots(ncols=2)
                axs[0].imshow(data['mbexp']['i'].image.array)
                axs[1].imshow(data['mbexp']['i'].variance.array)
                mplt.show()

        resdict = run_metadetect(rng=rng, config=None, **data)
        del resdict['_diagnostics']

        if do_zero:
            for shear_type, tres in resdict.items():
                w, = np.where(
                    tres['stamp_flags'] & procflags.ZERO_WEIGHTS != 0
                )
                assert w.size > 0, 'expected some stamp_flags set'
                assert np.all(tres['gauss_flags'][w] == procflags.NO_ATTEMPT)
                assert np.all(tres['pgauss_flags'][w] == procflags.NO_ATTEMPT)

        else:
            for shear_type, tres in resdict.items():
                # 5x5 grid
                assert tres.size == 25

        nobj.append(resdict['noshear'].size)

    assert nobj[0] == nobj[1]


def test_lsst_masked_as_bright(show=False):
    """
    Make sure we don't detect in areas marked BRIGHT
    """
    seed = 55
    afw_image.Mask.addMaskPlane('BRIGHT')
    bright = afw_image.Mask.getPlaneBitMask('BRIGHT')
    for do_zero in [False, True]:
        rng = np.random.RandomState(seed)
        sim_data = make_lsst_sim(seed, mag=23)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        if do_zero:
            data['mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['mbexp']['i'].mask.array[50:100, 50:100] |= bright
            data['noise_mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['noise_mbexp']['i'].mask.array[50:100, 50:100] |= bright

        resdict = run_metadetect(rng=rng, config=None, **data)
        del resdict['_diagnostics']

        if show:
            import matplotlib.pyplot as mplt
            fig, axs = mplt.subplots(ncols=2)
            axs[0].imshow(data['mbexp']['i'].image.array)
            axs[1].imshow(data['mbexp']['i'].variance.array)

            axs[0].scatter(
                resdict['noshear']['col'] - resdict['noshear']['col0'],
                resdict['noshear']['row'] - resdict['noshear']['row0'],
                s=4,
                c='red',
            )
            mplt.show()

        if do_zero:
            for shear_type, tres in resdict.items():
                assert tres.size == 24
        else:
            for shear_type, tres in resdict.items():
                # 5x5 grid
                assert tres.size == 25


def test_lsst_metadetect_prepsf_stars():
    seed = 55
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(seed, hlr=1.0e-4, mag=23)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}

    res = run_metadetect(rng=rng, config=config, **data)

    n = metadetect.util.Namer(front='pgauss')

    data = res['noshear']

    wlowT, = np.where(data[n('flags')] != 0)
    wgood, = np.where(data[n('flags')] == 0)

    # some will have T < 0 due to noise. Expect some with flags set
    assert wlowT.size > 0

    assert np.any((data[n('flags')][wlowT] & ngmix.flags.NONPOS_SIZE) != 0)

    for field in data.dtype.names:
        if field != "shear_bands":
            assert np.all(np.isfinite(data[field][wgood])), field


def test_lsst_metadetect_mfrac_ormask(show=False):
    rng = np.random.RandomState(seed=116)

    ntrial = 1
    flag = 2**30

    for trial in range(ntrial):
        sim_data = make_lsst_sim(rng.randint(0, 2**30))
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        data['mfrac_mbexp']['i'].image.array[:, :] = rng.uniform(
            size=data['mbexp']['i'].image.array.shape, low=0.2, high=0.8
        )

        for ormask in data['ormasks']:
            ormask[30:150, 30:150] = flag
            if show:
                import matplotlib.pyplot as mplt
                fig, axs = mplt.subplots(ncols=2)
                axs[0].imshow(data['mbexp']['i'].image.array)
                axs[1].imshow(ormask)
                mplt.show()

        res = run_metadetect(config=None, rng=rng, **data)

        for shear in ('noshear', '1p', '1m'):
            assert np.any(res[shear]["gauss_flags"] == 0)
            assert np.any(
                (res[shear]["mfrac"] > 0.40)
                & (res[shear]["mfrac"] < 0.60)
            )
            assert np.any(res[shear]["ormask"] & flag != 0)


@pytest.mark.parametrize('deblender', ['sdss', 'scarlet'])
def test_lsst_metadetect_deblender_grid(deblender):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {
        'deblender': deblender,
    }

    res = run_metadetect(rng=rng, config=config, **data)

    metacal_types = ['noshear', '1p', '1m']

    for metacal_type in metacal_types:
        assert (
            metacal_type in res.keys()
        ), f"metacal_type={metacal_type} not in res.keys()"

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res['noshear'].dtype.names

        flux_name = f'{front}_band_flux'

        for shear in metacal_types:
            # 5x5 grid
            assert res[shear].size == 25

            assert np.any(res[shear][f"{front}_flags"] == 0)
            assert np.all(res[shear]["mfrac"] == 0)

            assert len(res[shear][flux_name].shape) == len(bands)
            assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.parametrize('deblender', ['sdss', 'scarlet'])
def test_lsst_metadetect_deblender_random(deblender, show=False):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, mag=24, bands=bands, layout='random')
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    if show:
        vis.show_image(data['mbexp']['i'].image.array)

    config = {
        'deblender': deblender,
    }

    res = run_metadetect(rng=rng, config=config, **data)

    if show:
        vis.show_image(data['mbexp']['i'].image.array, cat=res['noshear'])

    metacal_types = ['noshear', '1p', '1m']

    for metacal_type in metacal_types:
        assert (
            metacal_type in res.keys()
        ), f"metacal_type={metacal_type} not in res.keys()"

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res['noshear'].dtype.names

        flux_name = f'{front}_band_flux'

        for shear in metacal_types:
            assert np.any(res[shear][f"{front}_flags"] == 0)
            assert np.all(res[shear]["mfrac"] == 0)

            assert len(res[shear][flux_name].shape) == len(bands)
            assert len(res[shear][flux_name][0]) == len(bands)


if __name__ == '__main__':
    # test_lsst_metadetect_deblender_random('sdss', show=True)
    test_lsst_metadetect_reconv_size()
    # test_lsst_metadetect_deblender_random('scarlet', show=True)
