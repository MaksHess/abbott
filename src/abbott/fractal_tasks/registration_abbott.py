import h5py
import argparse
from abbott.conversions import to_itk, to_numpy
from abbott.h5_files import h5_select, h5_copy_attributes
from abbott.itk_elastix_registration import register_transform_only, apply_transform, load_parameter_files
from pathlib import Path
import os
import numpy as np
from scipy.stats import linregress
import itk
import shutil


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('idx', type=int)
    CLI.add_argument('fld', type=str)
    CLI.add_argument('--parameter_files', nargs='*', type=str)
    CLI.add_argument('--ref_cycle', type=int, default=0)
    CLI.add_argument('--ref_stain', type=str, default='DAPI')
    CLI.add_argument('--level', type=int, default=0)
    CLI.add_argument('--out_dir_suffix', type=str, default='aligned')
    CLI.add_argument('--transform_other_channels', type=bool, default=True)
    args = CLI.parse_args()

    fns = sorted(list(Path(args.fld).glob('*.h5')))
    fn = fns[args.idx]

    align_cycles_in_file(fn, args.parameter_files, ref_stain=args.ref_stain, ref_cycle=args.ref_cycle, level=args.level,
                         out_dir_suffix=args.out_dir_suffix, transform_other_channels=args.transform_other_channels)


def align_cycles_in_file(
        fn_in, parameter_files, cycles=None, ref_cycle=0, ref_stain="DAPI", level=0, overwrite=True,
        out_dir_suffix='aligned', transform_other_channels=True
):
    fn_in = Path(fn_in)
    fn_out = fn_in.parent.parent / f"{'_'.join(fn_in.parent.name.split('_')[:-1])}_{out_dir_suffix}" / fn_in.name
    fn_out.parent.mkdir(exist_ok=True)

    _ = load_parameter_files(parameter_files)  # Make sure the files are ok before starting to process.
    params_folder = fn_out.parent / 'registration' / 'parameters'
    params_folder.mkdir(exist_ok=True, parents=True)
    for parameter_file in parameter_files:
        dst = params_folder / Path(parameter_file).name
        if not dst.exists():
            shutil.copy(parameter_file, dst)

    with h5py.File(fn_in) as f_in:
        with h5py.File(fn_out, "a") as f_out:
            # Copy intensity images of ref_cycle
            if transform_other_channels:
                channels = h5_select(f_in, {"cycle": ref_cycle, "img_type": "intensity", "level": level})
            else:
                channels = h5_select(f_in, {"cycle": ref_cycle, "img_type": "intensity", "level": level, "stain": ref_stain})
            for ch in channels:
                if f_out.get(ch.name) is None:
                    ch_new = f_out.create_dataset(
                        name=ch.name,
                        data=ch[...],
                        compression="gzip",
                        chunks=True,
                    )
                    h5_copy_attributes(f_in, f_out)
                    h5_copy_attributes(ch, ch_new)

            # Iterate trough other cycles and align them based on ref_stain
            if cycles is None:
                cycles = set(
                    dset.attrs["cycle"]
                    for dset in h5_select(f_in, {"img_type": "intensity", "level": level})
                )
                cycles.remove(ref_cycle)
            for cycle in cycles:
                align_cycle(
                    f_in,
                    f_out,
                    cycle,
                    ref_cycle=ref_cycle,
                    ref_stain=ref_stain,
                    parameter_files=parameter_files,
                    level=level,
                    overwrite=overwrite,
                    transform_other_channels=transform_other_channels
                )


def align_cycle(
        f_in: h5py.File,
        f_out: h5py.File,
        cycle: int,
        ref_cycle: int,
        ref_stain: str,
        parameter_files: list[str],
        level: int,
        intensity_normalization: bool = True,
        overwrite: bool = True,
        transform_other_channels: bool = True
):
    ref_dset = h5_select(
        f_out, {"img_type": "intensity", "cycle": ref_cycle, "stain": ref_stain, "level": level}
    )[0]
    move_dset = h5_select(
        f_in, {"img_type": "intensity", "cycle": cycle, "stain": ref_stain, "level": level}
    )[0]
    ref = to_itk(ref_dset)
    move = to_itk(move_dset)
    if intensity_normalization:
        ref_norm = quantile_rescale_exp(ref)
        move_norm = quantile_rescale_exp(move)
    else:
        ref_norm = ref
        move_norm = move
    trans = register_transform_only(ref_norm, move_norm, parameter_files)

    # Write transform parameter files
    for i in range(trans.GetNumberOfParameterMaps()):
        trans_map = trans.GetParameterMap(i)
        fn = Path(f_out.filename).parent / 'registration' / 'transforms' / (
                Path(f_out.filename).stem + "_cy{}-t{}.txt".format(cycle, i)
        )
        fn.parent.mkdir(exist_ok=True, parents=True)
        trans.WriteParameterFile(trans_map, fn.as_posix())

    # Apply transform to initial image
    img = apply_transform(move, trans)

    # Write transformed channel
    if move_dset.name in f_out and overwrite:
        del f_out[move_dset.name]
    f_out.create_dataset(
        name=move_dset.name,
        data=to_numpy(img)
            .clip(min=np.iinfo(move_dset.dtype).min, max=np.iinfo(move_dset.dtype).max)
            .astype(move_dset.dtype),
        compression="gzip",
        chunks=True,
    )
    h5_copy_attributes(move_dset, f_out[move_dset.name])

    if transform_other_channels:
        # Apply transform to other channels and write result
        for channel in h5_select(
                f_in, {"img_type": "intensity", "cycle": cycle, "level": level}, {"stain": ref_stain}
        ):
            img = apply_transform(to_itk(channel), trans)
            if channel.name in f_out and overwrite:
                del f_out[channel.name]
            f_out.create_dataset(
                name=channel.name,
                data=to_numpy(img)
                    .clip(min=np.iinfo(channel.dtype).min, max=np.iinfo(channel.dtype).max)
                    .astype(channel.dtype),
                compression="gzip",
                chunks=True,
            )
            h5_copy_attributes(channel, f_out[channel.name])


def quantile_rescale_exp(img_itk: itk.Image, q: tuple[float, float] = (0.01, 0.999),
                         rejected_planes: tuple[int, int] = (15, 50)) -> itk.Image:
    img = to_numpy(img_itk)
    lower, upper = np.quantile(img, q, axis=(1, 2))
    x = np.arange(len(lower))
    lower_r = lower[rejected_planes[0]:-rejected_planes[1]]
    upper_r = upper[rejected_planes[0]:-rejected_planes[1]]
    x_r = x[rejected_planes[0]:-rejected_planes[1]]

    lm_lower = linregress(x_r, np.log1p(lower_r))
    lm_upper = linregress(x_r, np.log1p(upper_r))

    lower_bounds = np.expm1(lm_lower.intercept + lm_lower.slope * x)
    upper_bounds = np.expm1(lm_upper.intercept + lm_upper.slope * x)

    res = (img - np.expand_dims(lower_bounds, axis=(1, 2))) / np.expand_dims(upper_bounds - lower_bounds, axis=(1, 2))

    out_itk = to_itk(res)
    out_itk.SetSpacing(img_itk.GetSpacing())
    out_itk.SetOrigin(img_itk.GetOrigin())
    return out_itk


if __name__ == "__main__":
    main()
