"""
Check that this branch gives bit-identical results to tag 0.13.0

A git worktree of the tag is created and regression_driver_v0_13_0.py is run
in subprocesses, once with PYTHONPATH pointing at the worktree and once with
it pointing at this checkout, in the same environment so the only variable is
the metadetect code.  Requires the git history, so this is skipped unless
METADETECT_REGRESSION_TEST=1 is set.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

TAG = '0.13.0'
DRIVER = os.path.join(
    os.path.dirname(__file__), 'regression_driver_v0_13_0.py'
)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)


def run_driver(checkout, outfile, deblender):
    # prepend so the checkout wins over any other metadetect on PYTHONPATH,
    # e.g. from eups; the stack itself is also on PYTHONPATH so we must
    # extend rather than replace
    pythonpath = checkout + os.pathsep + os.environ.get('PYTHONPATH', '')
    env = dict(os.environ, PYTHONPATH=pythonpath)
    subprocess.run(
        [sys.executable, DRIVER, outfile, deblender],
        env=env, check=True, cwd=os.path.dirname(outfile),
    )


@pytest.mark.skipif(
    os.environ.get('METADETECT_REGRESSION_TEST') != '1',
    reason='set METADETECT_REGRESSION_TEST=1 to run',
)
@pytest.mark.parametrize('deblender', ['sdss', 'scarlet'])
def test_bit_identical_vs_0_13_0(deblender, tmp_path):
    worktree = str(tmp_path / f'metadetect-{TAG}')
    subprocess.run(
        ['git', '-C', REPO_ROOT, 'worktree', 'add', '--detach', worktree, TAG],
        check=True,
    )

    try:
        old_file = str(tmp_path / f'old-{deblender}.npz')
        new_file = str(tmp_path / f'new-{deblender}.npz')

        run_driver(worktree, old_file, deblender)
        run_driver(REPO_ROOT, new_file, deblender)

        with np.load(old_file) as old, np.load(new_file) as new:
            # make sure each run imported the intended version
            assert str(old['__metadetect_file__']).startswith(worktree)
            assert str(new['__metadetect_file__']).startswith(REPO_ROOT)

            old_keys = set(old.files) - {'__metadetect_file__'}
            new_keys = set(new.files) - {'__metadetect_file__'}
            assert old_keys == new_keys

            for key in sorted(old_keys):
                old_res, new_res = old[key], new[key]
                assert old_res.dtype == new_res.dtype, key
                assert old_res.shape == new_res.shape, key
                assert old_res.tobytes() == new_res.tobytes(), key
    finally:
        subprocess.run(
            ['git',
             '-C',
             REPO_ROOT,
             'worktree',
             'remove',
             '--force',
             worktree],
            check=True,
        )
