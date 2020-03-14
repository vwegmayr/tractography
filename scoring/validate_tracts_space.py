#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os

import nibabel as nb
import numpy as np

import tractconverter as tc

def _lps_allowed(tracts_filename):
    tracts_format = tc.detect_format(tracts_filename)
    tracts_file = tracts_format(tracts_filename)

    if isinstance(tracts_file, tc.formats.vtk.VTK):
        return True

    return False


def _is_format_supported(tracts_filename):
    tracts_format = tc.detect_format(tracts_filename)
    tracts_file = tracts_format(tracts_filename)

    if isinstance(tracts_file, tc.formats.tck.TCK) \
        or isinstance(tracts_file, tc.formats.trk.TRK) \
        or isinstance(tracts_file, tc.formats.vtk.VTK):
        return True

    return False


def _print_required_and_found(required_mins, required_maxs,
                              found_mins, found_maxs, lps_oriented):
    if lps_oriented:
        orient_str = "LPS oriented"
    else:
        orient_str = "RAS oriented"

    debug_message = "You told us that you were " + orient_str + "\n"
    debug_message += "In that case, we should find points that are between\n"
    debug_message += "{0} and {1}\n".format(required_mins, required_maxs)
    debug_message += "We found points between\n"
    debug_message += "{0} and {1}".format(found_mins, found_maxs)
    return debug_message


def _is_tracts_space_valid(tracts_filename, lps_oriented):
    tracts_format = tc.detect_format(tracts_filename)
    tracts_file = tracts_format(tracts_filename)

    # Compute boundaries of volume
    if lps_oriented:
        required_mins = np.array([-1.0, -1.0, -1.0])
        required_maxs = np.array([179.0, 215.0, 179.0])
    else:
        required_mins = np.array([-179.0, -215.0, -1.0])
        required_maxs = np.array([1.0, 1.0, 179.0])

    # We compute them directly in the loop inside the format dependent code
    # to avoid 2 loops and to avoid loading everything in memory.
    minimas = []
    maximas = []

    # Load tracts
    if isinstance(tracts_file, tc.formats.vtk.VTK) \
        or isinstance(tracts_file, tc.formats.tck.TCK):
        for s in tracts_file:
            minimas.append(np.min(s, axis=0))
            maximas.append(np.max(s, axis=0))
    elif isinstance(tracts_file, tc.formats.trk.TRK):
         # Use nb.trackvis to read directly in correct space
        try:
            streamlines, _ = nb.trackvis.read(tracts_filename,
                                              as_generator=True,
                                              points_space='rasmm')
        except nb.trackvis.HeaderError as er:
            msg = "\n------ ERROR ------\n\n" +\
                  "TrackVis header is malformed or incomplete.\n" +\
                  "Please make sure all fields are correctly set.\n\n" +\
                  "The error message reported by Nibabel was:\n" +\
                  str(er)
            return msg

        for s in streamlines:
            minimas.append(np.min(s[0], axis=0))
            maximas.append(np.max(s[0], axis=0))

    global_min = np.min(minimas, axis=0)
    global_max = np.max(maximas, axis=0)

    if np.all(global_min > required_mins) and \
        np.all(global_max < required_maxs):
        return "Tracts seem to be in the correct space"
    elif isinstance(tracts_file, tc.formats.vtk.VTK) and\
         np.all(global_min * np.array([-1.0, -1.0, 1.0]) > required_mins) \
         and np.all(global_max * np.array([-1.0, -1.0, 1.0]) < required_maxs):
        return "Tracts seem to be reverted. Did you use the --lps flag?\n" +\
                "If so, it means the tracts are not in the correct space."

    return "Tracts do not seem to be in the correct space.\n\n" + \
           _print_required_and_found(required_mins, required_maxs,
                                     global_min, global_max, lps_oriented)


def _buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Loads a tractography file and checks if the points '
                    'seem to fit with the ISMRM 2015 challenge dataset.')
    p.add_argument('tracts', action='store',
                   metavar='TRACTS', type=str,
                   help='path of the tracts file, in a format supported by ' +
                        'the TractConverter (.tck, .trk, or VTK).')
    p.add_argument('--lps', action='store_true',
                   help='Set this flag if your tools work using a LPS reference' +
                        ' instead of the default RAS reference.\n' +
                        'Mainly impacts VTK files.\n' +
                        'Tools like MITK and ITK work in LPS.')
    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if not _is_format_supported(args.tracts):
        parser.error('Format of "{0}" not supported by the Challenge.'.format(args.tracts))

    if args.lps and not _lps_allowed(args.tracts):
        parser.error('LPS orientation not allowed with your file format.')

    valid_message = _is_tracts_space_valid(args.tracts, args.lps)

    print(valid_message)


if __name__ == "__main__":
    main()
