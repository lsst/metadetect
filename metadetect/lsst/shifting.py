"""
For the purpose of shifting centroids in the metacal procedure which is
necessary within the coadd framework of metadetect
"""

import numpy as np
import galsim
from lsst.afw.image import ExposureF
from metadetect.lsst.metacal_exposures import get_shear_from_type


def shift_objects(
    shift_method,
    sources,
    shear_str,
    mcal_mbexp,
    meas_task,
):
    """
    Parameters
    ----------
    shift_method : str
        Method of shifting centroids, options are None, sdss_centroid,
        galsim, and a_matrix where default is galsim
    sources : lsst.afw.table.SourceCatalog
        Source catalog containing objects to shift
    shear_str : str
        Shear type of the exposure
    mcal_mbexp :
        Multiband exposure object
    meas_task :

    Returns
    -------
    sources : lsst.afw.table.SourceCatalog
        Source catalog with shifted centroids according to the artificial shear
    """

    dbtask = meas_task

    # Shift according to the shear type
    if shift_method == "sdss_centroid":
        centroid_exp = ExposureF(mcal_mbexp["i"], deep=True)  # make this for any band?
        dbtask.run(measCat=sources, exposure=centroid_exp)

    # Using ID column, shift every source
    for source_id in sources["id"]:
        source = sources.find(source_id)
        if source.get("deblend_nChild") != 0:
            continue
        x0, y0 = source.getX(), source.getY()

        if shift_method == 'sdss_centroid':
            x1 = source.get("base_SdssCentroid_x")
            y1 = source.get("base_SdssCentroid_y")

            # If need for diagnostics
            # dx, dy = x1 - x0, y1 - y0

        # if shift_method is None:
            # dx, dy = 0, 0

        # Image origin definition for Galsim and A_Matrix:
        width, height = mcal_mbexp["i"].getWidth(), mcal_mbexp["i"].getHeight()
        xc, yc = (width - 1.0) / 2.0, (height - 1.0) / 2.0

        if shift_method == 'a_matrix':

            if shear_str != 'noshear':
                shear = get_shear_from_type(shear_str)
                g1, g2 = shear.g1, shear.g2
            else:
                shear = galsim.Shear(g1=0, g2=0)
                g1, g2 = shear.g1, shear.g2

            g = np.sqrt(g1**2 + g2**2)

            x1 = (((1 + g1) * (x0 - xc) + g2 * (y0 - yc)) / (1-g**2)) + xc
            y1 = ((g2 * (x0 - xc) + (1 - g1) * (y0 - yc)) / (1-g**2)) + yc

            # If need for diagnostics
            # dx, dy = x1 - x0, y1 - y0

            source.set("base_SdssCentroid_x", x1)
            source.set("base_SdssCentroid_y", y1)

        if shift_method == "galsim":

            if shear_str != 'noshear':
                shear = get_shear_from_type(shear_str)
            else:
                shear = galsim.Shear(g1=0, g2=0)

            pos = galsim.PositionD(x0 - xc, y0 - yc)
            sheared_pos = pos.shear(shear)

            x1, y1 = sheared_pos.x + xc, sheared_pos.y + yc

            # If need for diagnostics
            # dx, dy = x1 - x0, y1 - y0

            source.set("base_SdssCentroid_x", x1)
            source.set("base_SdssCentroid_y", y1)

    return sources
