"""Check native sharded-AdamW accumulation without an incorrect legacy oracle."""

import argparse
from pathlib import Path
import sys
import tempfile


def assert_accumulation_close(actual, expected, *, path="root"):
    import torch

    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True,
                                   msg=lambda detail: f"{path}: {detail}")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_accumulation_close(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_accumulation_close(left, right, path=f"{path}[{index}]")
    else:
        assert actual == expected, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.package_root is not None:
        sys.path.insert(0, str(args.package_root.resolve()))

    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    import dmuon
    from ascend_cases import run_training, tensor_deltas

    assert "torch_npu.contrib.transfer_to_npu" not in sys.modules
    if args.package_root is not None:
        assert Path(dmuon.__file__).resolve().is_relative_to(args.package_root.resolve())
    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    result = {"metadata": {"package": str(Path(dmuon.__file__).resolve()),
                           "torch": torch.__version__, "torch_npu": torch_npu.__version__},
              "cases": {}}
    with tempfile.TemporaryDirectory(prefix="dmuon-sharded-accum-") as directory:
        dist.init_process_group(
            "hccl", init_method=Path(directory, "rendezvous").as_uri(), rank=0, world_size=1,
        )
        try:
            mesh = init_device_mesh("npu", (1,))
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                pair = {}
                result["cases"][str(dtype)] = pair
                for no_sync in (False, True):
                    pair[str(no_sync)] = run_training(
                        dmuon, mesh, device, dtype=dtype, cast=False, pathway="fsdp2",
                        asynchronous=True, no_sync=no_sync, sharded_adamw=True,
                        grad_dtype=torch.float32,
                    )
                # Every entry contains four-microbatch gradients, model updates,
                # optimizer states and an exact restored-next-update check.
                try:
                    assert_accumulation_close(pair["True"], pair["False"], path=str(dtype))
                except AssertionError as error:
                    result["failure"] = {"stage": "default_vs_no_sync", "dtype": str(dtype),
                                         "error": str(error),
                                         "deltas": tensor_deltas(pair["True"], pair["False"])}
                    raise
        except AssertionError as error:
            if hasattr(error, "restore_failure"):
                result["failure"] = error.restore_failure
            elif "failure" not in result:
                result["failure"] = {"stage": "training_case", "dtype": str(dtype),
                                     "no_sync": no_sync, "error": str(error)}
            failed_path = args.output.with_name(args.output.stem + "-failure.pt")
            with failed_path.open("xb") as handle:
                torch.save(result, handle)
            raise AssertionError(f"{error}; failure snapshot: {failed_path}") from error
        finally:
            torch.npu.synchronize()
            dist.destroy_process_group()
    with args.output.open("xb") as handle:
        torch.save(result, handle)
    print(f"PASS: three compute dtypes, default/no_sync accumulation and active-state restoration: {args.output}")


if __name__ == "__main__":
    main()
