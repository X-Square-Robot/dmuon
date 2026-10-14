"""Run the same single-NPU cases against an installed or frozen DMuon package.

This utility records values for a migration comparison; it does not implement
an optimizer reference. A legacy compatibility bootstrap may be supplied only
when executing the frozen pre-migration package in a separate process.
"""

import argparse
from pathlib import Path
import runpy
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--legacy-bootstrap", type=Path)
    parser.add_argument(
        "--no-verify-restore",
        dest="verify_restore",
        action="store_false",
        help="Skip only the frozen legacy package's known broken restore path",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--pathway", choices=("ddp", "fsdp2"), default="fsdp2")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.verify_restore and args.legacy_bootstrap is None:
        parser.error("--no-verify-restore requires --legacy-bootstrap")
    if args.package_root is not None:
        sys.path.insert(0, str(args.package_root.resolve()))
    if args.legacy_bootstrap is not None:
        runpy.run_path(str(args.legacy_bootstrap), run_name="__dmuon_legacy_bootstrap__")

    import torch
    import torch_npu  # Registers native NPU support; no transfer shim.
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    import dmuon

    from ascend_cases import (
        ns_cases,
        run_clip,
        run_empty_clip,
        run_nonfinite_clip_error,
        run_ns,
        run_stream_lifetime,
        run_training,
    )

    if args.legacy_bootstrap is None:
        assert "torch_npu.contrib.transfer_to_npu" not in sys.modules
    package_path = Path(dmuon.__file__).resolve()
    if args.package_root is not None:
        assert package_path.is_relative_to(args.package_root.resolve())
    metadata = {
        "momentum_policy": "optim_dtype",
        "package": str(package_path),
        "pathway": args.pathway,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "legacy_bootstrap": str(args.legacy_bootstrap),
        "verify_restore_requested": args.verify_restore,
        "restore_verified": False,
        "legacy_restore_verified": False,
    }
    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    result = {"ns": {}, "clip": {}, "training": {}}
    for shape, dtype, family in ns_cases():
        key = f"{shape}:{dtype}:{family}"
        result["ns"][key] = run_ns(dmuon, device, shape, dtype, family)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for segmented in (False, True):
            for value in ([3.0, 4.0], [0.0, 0.0], [float("inf"), 1.0], [float("nan"), 1.0]):
                for max_norm in (None, 2.5):
                    key = f"{dtype}:{segmented}:{value}:{max_norm}"
                    result["clip"][key] = run_clip(
                        dmuon,
                        device,
                        value,
                        max_norm,
                        segmented=segmented,
                        dtype=dtype,
                    )
    result["empty_clip"] = run_empty_clip(dmuon, device)
    result["nonfinite_errors"] = {
        f"{dtype}:{value}": run_nonfinite_clip_error(dmuon, device, value, dtype=dtype)
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
        for value in (float("inf"), float("nan"))
    }
    result["stream_lifetime"] = run_stream_lifetime(dmuon, device)

    with tempfile.TemporaryDirectory(prefix="dmuon-npu-test-") as directory:
        dist.init_process_group(
            "hccl",
            init_method=Path(directory, "rendezvous").as_uri(),
            rank=0,
            world_size=1,
        )
        try:
            mesh = init_device_mesh("npu", (1,))
            for dtype, cast, asynchronous, no_sync in (
                (torch.float32, False, False, False),
                (torch.float32, False, False, True),
                (torch.float32, True, True, True),
                (torch.float16, False, True, False),
                (torch.bfloat16, False, True, True),
            ):
                key = f"{dtype}:{cast}:{asynchronous}:{no_sync}"
                try:
                    result["training"][key] = run_training(
                        dmuon,
                        mesh,
                        device,
                        dtype=dtype,
                        cast=cast,
                        pathway=args.pathway,
                        asynchronous=asynchronous,
                        no_sync=no_sync,
                        verify_restore=args.verify_restore,
                    )
                except AssertionError as error:
                    failure = getattr(error, "restore_failure", None)
                    if failure is not None:
                        failure["training_case"] = key
                        failed_path = args.output.with_name(args.output.stem + "-failure.pt")
                        with failed_path.open("xb") as handle:
                            torch.save(
                                {"metadata": metadata, "cases": result, "failure": failure}, handle
                            )
                        raise AssertionError(f"{error}; failure snapshot: {failed_path}") from error
                    raise
            default = result["training"]["torch.float32:False:False:False"][0]["gradients"]
            accumulated = result["training"]["torch.float32:False:False:True"][0]["gradients"]
            assert default.keys() == accumulated.keys()
            for name in default:
                # The no_sync path merges buffers before its final reduction;
                # accumulation order may differ from reducing every microbatch.
                try:
                    torch.testing.assert_close(
                        default[name],
                        accumulated[name],
                        rtol=1e-5,
                        atol=1e-6,
                        msg=f"default versus no_sync gradient: {name}",
                    )
                except AssertionError as error:
                    delta = default[name].float() - accumulated[name].float()
                    failure = {
                        "stage": "default_vs_no_sync",
                        "parameter": name,
                        "shape": list(default[name].shape),
                        "dtype": str(default[name].dtype),
                        "max_abs_diff": float(delta.abs().max()),
                        "relative_l2_diff": float(
                            delta.norm() / accumulated[name].float().norm().clamp_min(1e-30)
                        ),
                        "default_gradient": default[name],
                        "no_sync_gradient": accumulated[name],
                        "error": str(error),
                    }
                    failed_path = args.output.with_name(args.output.stem + "-failure.pt")
                    with failed_path.open("xb") as handle:
                        torch.save(
                            {"metadata": metadata, "cases": result, "failure": failure}, handle
                        )
                    raise AssertionError(f"{error}; failure snapshot: {failed_path}") from error
        finally:
            torch.npu.synchronize()
            dist.destroy_process_group()
    metadata["restore_verified"] = args.verify_restore
    metadata["legacy_restore_verified"] = args.legacy_bootstrap is not None and args.verify_restore
    torch.save({"metadata": metadata, "cases": result}, args.output)
    print(
        f"Saved {len(result['ns'])} NS, {len(result['clip'])} clip and "
        f"{len(result['training'])} training cases to {args.output}"
    )


if __name__ == "__main__":
    main()
