#!/usr/bin/env python3
"""
Main entry point for the TrackerKit application.

This module provides the command-line interface and GUI launcher for the
TrackerKit tracking system (part of HYDRA Suite).
"""

import argparse
import logging
import os
import sys
from typing import Sequence

from hydra_suite.core.inference.config import (
    DEFAULT_CALIBRATION_BUDGET_SECONDS,
    MAXIMUM_CALIBRATION_BUDGET_SECONDS,
    MINIMUM_CALIBRATION_BUDGET_SECONDS,
)
from hydra_suite.trackerkit.cli import run_tracking_cli

# Fix OpenMP conflict on macOS (PyTorch + OpenCV + NumPy can load multiple
# OpenMP libraries)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Set up logging
def setup_logging(
    log_level: object = logging.INFO,
    _enable_file_logging: object = False,
    log_dir: object = None,
) -> object:
    """Set up logging configuration for the multi-tracker application.

    Note: File logging is now handled per-session in main_window.py.
    This only sets up console logging.
    """

    # Only set up console logging - session logs are created in main_window.py
    handlers = [logging.StreamHandler(sys.stdout)]

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info("TrackerKit starting up...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the TrackerKit argument parser (no parsing side effects).

    Extracted so tests can exercise argument parsing (choices, mutually
    exclusive groups, subcommands) without invoking ``main`` or
    ``parse_arguments``'s post-parse validation.
    """
    parser = argparse.ArgumentParser(
        description="TrackerKit - GUI and basic config-driven tracking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trackerkit                    # Launch GUI
    trackerkit track video.mp4    # Run tracking for one video
    trackerkit track a.mp4 b.mp4 --keystone-override
    trackerkit track --video-list batch.txt
  trackerkit --log-level DEBUG  # Launch with debug logging
  trackerkit --no-file-log      # Disable file logging
        """,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-file-log", action="store_true", help="Disable file logging (console only)"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files (default: current directory)",
    )

    parser.add_argument("--version", action="version", version="TrackerKit 1.0.0")

    subparsers = parser.add_subparsers(dest="command")
    track_parser = subparsers.add_parser(
        "track",
        help="Run config-driven tracking on one or more videos",
        allow_abbrev=False,
    )
    track_parser.add_argument(
        "videos",
        nargs="*",
        default=[],
        help="One or more video paths. The first video is the keystone for batch fallback.",
    )
    track_parser.add_argument(
        "--video",
        dest="video_flag",
        type=str,
        default=None,
        help=(
            "Single video path, as an alternative to the positional videos "
            "argument (matches `trackerkit calibrate --video`)."
        ),
    )
    track_parser.add_argument(
        "--video-list",
        type=str,
        help="Optional plain-text batch list file with one video path per line, matching the GUI import/export format.",
    )
    track_parser.add_argument(
        "--config",
        type=str,
        help="Optional config file to use for the first video. For multi-video batches, an explicit config automatically becomes the batch keystone config for all videos.",
    )
    track_parser.add_argument(
        "--keystone-override",
        action="store_true",
        help="Force all later batch videos to use the first video's effective config when no explicit batch config was supplied.",
    )
    track_parser.add_argument(
        "--sahi-profile",
        type=str,
        help=(
            "Name or id of a calibration profile from the direct model's "
            ".slice_meta.json sidecar. Overrides the profile saved in the "
            "config. Use __training__ for the model's training geometry. "
            "Applies to every video in the batch."
        ),
    )
    autotune_group = track_parser.add_mutually_exclusive_group()
    autotune_group.add_argument(
        "--apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_true",
        default=None,
        help="Apply a validated inference profile if one exists for this "
        "configuration. Produce one with `trackerkit calibrate`.",
    )
    autotune_group.add_argument(
        "--no-apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_false",
        help="Ignore any stored inference profile and use configured values.",
    )
    track_parser.add_argument(
        "--inference-autotune-manual",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "Keep one tuning coordinate at its configured value. May be "
            "repeated; for example pose_batch_size or pipeline_depth."
        ),
    )

    track_parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Run videos as parallel child processes, one per listed GPU. "
            "Accepts ordinals (0,1,2), ranges (0-8), GPU- UUID prefixes, or "
            "'auto' for every GPU nvidia-smi reports. Each child sees exactly "
            "one GPU via CUDA_VISIBLE_DEVICES. NVIDIA hosts only."
        ),
    )
    track_parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Maximum concurrent videos. Defaults to one per selected GPU, "
            "else 1 (in-process sequential). With --gpus it is clamped to the "
            "number of GPUs; without --gpus, N>1 runs N children that share "
            "the current device visibility."
        ),
    )
    track_parser.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help=(
            "Opt-in CPU thread cap per child (sets OMP/MKL/OPENBLAS/NUMBA "
            "*_NUM_THREADS when not already set). Off by default."
        ),
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Measure and persist a validated inference-throughput profile "
            "for one video/config, on THIS box. The headless equivalent of "
            "the GUI's Calibrate button."
        ),
        allow_abbrev=False,
    )
    calibrate_parser.add_argument(
        "--video",
        dest="video",
        type=str,
        required=True,
        help="Video path to calibrate against.",
    )
    calibrate_parser.add_argument(
        "--config",
        type=str,
        help="Optional config file. Uses the same defaults as `track` when omitted.",
    )
    calibrate_parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_CALIBRATION_BUDGET_SECONDS,
        help=(
            "Wall-clock budget for the calibration search, between "
            f"{MINIMUM_CALIBRATION_BUDGET_SECONDS:g} and "
            f"{MAXIMUM_CALIBRATION_BUDGET_SECONDS:g} seconds "
            f"(default: {DEFAULT_CALIBRATION_BUDGET_SECONDS:g})."
        ),
    )
    # Manual fields feed ``compute_baseline_digest`` -> ``key.baseline_digest``
    # (integration.py), so a project that pins a coordinate keys its profile
    # differently from one that leaves it free. Without this flag here,
    # ``calibrate`` could not produce the key a
    # ``track --inference-autotune-manual ...`` run looks up.
    calibrate_parser.add_argument(
        "--inference-autotune-manual",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "Keep one tuning coordinate at its configured value. May be "
            "repeated. MUST match the `track` run that will use the "
            "resulting profile -- it is part of the profile key."
        ),
    )
    # Deliberately NO --gpus / --jobs: concurrent calibration on one box
    # measures contention, not throughput, and would silently produce a
    # confidently wrong profile. Neither flag is registered on this
    # subparser, so passing either is an "unrecognized arguments" SystemExit
    # from argparse itself -- a loud rejection, not a silent ignore.

    job_parser = subparsers.add_parser(
        "job",
        help="Package, move, run and retrieve a portable tracking job",
        allow_abbrev=False,
    )
    job_subparsers = job_parser.add_subparsers(dest="job_command")

    job_pack = job_subparsers.add_parser(
        "pack", allow_abbrev=False, help="Package a video/config batch into a job dir"
    )
    job_pack.add_argument("job_dir", type=str)
    job_pack.add_argument("videos", nargs="*", default=[])
    job_pack.add_argument("--video-list", type=str)
    job_pack.add_argument("--config", type=str)
    job_pack.add_argument("--keystone-override", action="store_true")
    job_pack.add_argument("--sahi-profile", type=str)
    pack_autotune_group = job_pack.add_mutually_exclusive_group()
    pack_autotune_group.add_argument(
        "--apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_true",
        default=None,
    )
    pack_autotune_group.add_argument(
        "--no-apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_false",
    )
    job_pack.add_argument(
        "--inference-autotune-manual", action="append", default=[], metavar="FIELD"
    )
    job_pack.add_argument("--job-name", type=str, default=None)
    job_pack.add_argument("--copy-videos", action="store_true")
    pack_shared_group = job_pack.add_mutually_exclusive_group()
    pack_shared_group.add_argument("--no-shared", action="store_true")
    pack_shared_group.add_argument("--shared-only", action="store_true")
    # Deliberately NO --gpus / --jobs / --threads-per-job: they describe the
    # COMPUTE BOX, not the experiment, and belong to `job run`. Not
    # registering them makes passing one an "unrecognized arguments"
    # SystemExit from argparse itself -- a loud rejection, not a silent
    # ignore. Same reasoning as the calibrate subparser above.
    # Fix X8: --force alone only allows repacking a non-empty job_dir
    # (clears models/config, removes only the OLD manifest's own
    # video/sidecar pack artifacts). It does NOT authorize discarding
    # pulled outputs sitting under videos/ -- that needs the separate,
    # explicit --discard-outputs.
    job_pack.add_argument("--force", action="store_true")
    job_pack.add_argument("--discard-outputs", action="store_true")

    job_verify = job_subparsers.add_parser(
        "verify", allow_abbrev=False, help="Offline integrity check for a packed job"
    )
    job_verify.add_argument("job_dir", type=str)
    job_verify.add_argument("--fast", action="store_true")

    job_shared_root = job_subparsers.add_parser(
        "shared-root",
        allow_abbrev=False,
        help="Manage the host shared-root mount table",
    )
    shared_root_subparsers = job_shared_root.add_subparsers(dest="shared_root_command")
    shared_root_add = shared_root_subparsers.add_parser("add", allow_abbrev=False)
    shared_root_add.add_argument("alias", type=str)
    shared_root_add.add_argument("path", type=str)
    shared_root_remove = shared_root_subparsers.add_parser("remove", allow_abbrev=False)
    shared_root_remove.add_argument("alias", type=str)
    shared_root_subparsers.add_parser("list", allow_abbrev=False)

    job_push = job_subparsers.add_parser(
        "push", allow_abbrev=False, help="rsync a packed job to a compute box"
    )
    job_push.add_argument("job_dir", type=str)
    job_push.add_argument("target", type=str)
    job_push.add_argument("--remote-bootstrap", type=str, default="")

    job_pull = job_subparsers.add_parser(
        "pull", allow_abbrev=False, help="rsync outputs back beside their origin videos"
    )
    job_pull.add_argument("target", type=str)
    job_pull.add_argument("job_dir", type=str)
    job_pull.add_argument("--dry-run", action="store_true")
    job_pull.add_argument("--no-caches", action="store_true")
    job_pull.add_argument("--overwrite", action="store_true")
    # Fix A5: still-running guard is ON by default.
    job_pull.add_argument("--force", action="store_true", default=False)

    job_preflight = job_subparsers.add_parser(
        "preflight",
        allow_abbrev=False,
        help="Host checks + shared-root materialization",
    )
    job_preflight.add_argument("job_dir", type=str)
    job_preflight.add_argument(
        "--shared-root", action="append", default=[], metavar="ALIAS=PATH"
    )
    job_preflight.add_argument("--allow-tier-fallback", action="store_true")
    job_preflight.add_argument("--fast", action="store_true")

    job_run = job_subparsers.add_parser(
        "run", allow_abbrev=False, help="Run a packed job locally or on a remote box"
    )
    job_run.add_argument("target", type=str)
    job_run.add_argument("--gpus", type=str, default=None)
    job_run.add_argument("--jobs", type=int, default=None)
    job_run.add_argument("--threads-per-job", type=int, default=None)
    job_run.add_argument("--detach", action="store_true")
    job_run.add_argument("--calibrate", action="store_true")
    job_run.add_argument("--allow-tier-fallback", action="store_true")
    job_run.add_argument(
        "--shared-root", action="append", default=[], metavar="ALIAS=PATH"
    )
    # Fix A2b: default must be "" -- an empty bootstrap fails loudly against
    # a box where trackerkit is not on a bare ssh PATH, rather than silently
    # mis-scheduling.
    job_run.add_argument("--remote-bootstrap", type=str, default="")
    job_run.add_argument(
        "--budget-seconds", type=float, default=DEFAULT_CALIBRATION_BUDGET_SECONDS
    )
    # Deliberately NO --sahi-profile / --apply-tuned-inference /
    # --inference-autotune-manual: fixed at pack time and always forwarded
    # from track_args.

    job_calibrate = job_subparsers.add_parser(
        "calibrate", allow_abbrev=False, help="Run calibration for a packed job"
    )
    job_calibrate.add_argument("target", type=str)
    job_calibrate.add_argument(
        "--inference-autotune-manual", action="append", default=[], metavar="FIELD"
    )
    job_calibrate.add_argument("--remote-bootstrap", type=str, default="")
    job_calibrate.add_argument(
        "--budget-seconds", type=float, default=DEFAULT_CALIBRATION_BUDGET_SECONDS
    )

    job_status = job_subparsers.add_parser(
        "status", allow_abbrev=False, help="Show a job's manifest summary + run history"
    )
    job_status.add_argument("target", type=str)
    job_status.add_argument("--remote-bootstrap", type=str, default="")

    # Hidden: appends one JSON line to logs/runs.jsonl. Fix Y2 -- add_parser
    # does NOT accept metavar=; help=argparse.SUPPRESS is the kwarg that
    # hides a subparser from -h while leaving it fully callable.
    job_record_run = job_subparsers.add_parser("_record-run", help=argparse.SUPPRESS)
    job_record_run.add_argument("--started", type=str, required=True)
    job_record_run.add_argument("--exit-code", type=int, required=True)
    job_record_run.add_argument("argv", nargs=argparse.REMAINDER)

    return parser


def _subparser_choices(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    """The ``{name: subparser}`` map, without reaching into private attrs."""
    for action in parser._actions:  # noqa: SLF001 - argparse has no public accessor
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def parse_arguments(argv: list[str] | None = None) -> object:
    """Parse command line arguments."""
    parser = build_parser()
    subcommands = _subparser_choices(parser)
    track_parser = subcommands["track"]
    calibrate_parser = subcommands["calibrate"]

    args = parser.parse_args(argv)

    if args.command == "track":
        videos = list(getattr(args, "videos", []) or [])
        video_flag = getattr(args, "video_flag", None)
        if video_flag:
            videos.append(str(video_flag))
        args.videos = videos
        video_list = getattr(args, "video_list", None)
        if videos and video_list:
            track_parser.error(
                "use either explicit video paths or --video-list, not both"
            )
        if not videos and not video_list:
            track_parser.error("provide at least one video path or --video-list")
        jobs = getattr(args, "jobs", None)
        if jobs is not None and int(jobs) < 1:
            track_parser.error("--jobs must be >= 1")
        tpj = getattr(args, "threads_per_job", None)
        if tpj is not None and int(tpj) < 1:
            track_parser.error("--threads-per-job must be >= 1")

    if args.command == "calibrate":
        budget = getattr(args, "budget_seconds", None)
        if budget is not None and not (
            MINIMUM_CALIBRATION_BUDGET_SECONDS
            <= float(budget)
            <= MAXIMUM_CALIBRATION_BUDGET_SECONDS
        ):
            calibrate_parser.error(
                "--budget-seconds must be between "
                f"{MINIMUM_CALIBRATION_BUDGET_SECONDS:g} and "
                f"{MAXIMUM_CALIBRATION_BUDGET_SECONDS:g}"
            )

    if args.command == "job":
        job_parser = subcommands["job"]
        job_subcommands = _subparser_choices(job_parser)
        job_command = getattr(args, "job_command", None)
        if job_command == "pack":
            pack_parser = job_subcommands["pack"]
            videos = list(getattr(args, "videos", []) or [])
            video_list = getattr(args, "video_list", None)
            if videos and video_list:
                pack_parser.error(
                    "use either explicit video paths or --video-list, not both"
                )
            if not videos and not video_list:
                pack_parser.error("provide at least one video path or --video-list")

    return args


def load_video_list(video_list_path: str) -> list[str]:
    """Load a GUI-format batch list text file, keeping keystone-first ordering."""

    if not os.path.isfile(video_list_path):
        raise FileNotFoundError(f"Video list not found: {video_list_path}")

    try:
        with open(video_list_path, "r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n").strip() for line in handle if line.strip()]
    except OSError as exc:
        raise OSError(f"Failed to read video list: {video_list_path}") from exc

    if not lines:
        raise ValueError(f"Video list contains no video paths: {video_list_path}")

    valid = [path for path in lines if os.path.isfile(path)]
    missing = [path for path in lines if not os.path.isfile(path)]

    if not valid:
        raise FileNotFoundError(
            f"None of the video paths in the list could be found: {video_list_path}"
        )
    if lines[0] not in valid:
        raise FileNotFoundError(
            f"The keystone video (first line) does not exist: {lines[0]}"
        )

    if missing:
        logging.getLogger(__name__).warning(
            "Skipping %s missing video path(s) from %s",
            len(missing),
            video_list_path,
        )

    return valid


def resolve_track_video_inputs(
    videos: Sequence[str] | None,
    video_list_path: str | None = None,
) -> list[str]:
    """Resolve CLI video inputs from either explicit paths or a batch list file."""

    normalized_videos = [
        str(path).strip() for path in (videos or []) if str(path).strip()
    ]
    if normalized_videos and video_list_path:
        raise ValueError("Use either explicit video paths or --video-list, not both.")
    if video_list_path:
        return load_video_list(video_list_path)
    if normalized_videos:
        return normalized_videos
    raise ValueError("At least one video path or --video-list is required.")


def check_dependencies() -> object:
    """Check that all required dependencies are available."""
    required_modules = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
    ]

    missing_modules = []
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(package_name)

    if missing_modules:
        print("Error: Missing required dependencies:")
        for package in missing_modules:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print(f"conda install -c conda-forge {' '.join(missing_modules)}")
        print("or")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    return True


def main(argv: list[str] | None = None) -> object:
    """
    Application entry point.

    Parses command line arguments, sets up logging, checks dependencies,
    creates Qt application, initializes main window, and starts event loop.
    """
    # Parse command line arguments
    args = parse_arguments(argv)

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(
        log_level=log_level,
        _enable_file_logging=not args.no_file_log,
        log_dir=args.log_dir,
    )

    logger = logging.getLogger(__name__)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    if args.command == "track":
        try:
            resolved_videos = resolve_track_video_inputs(args.videos, args.video_list)
            exit_code = run_tracking_cli(
                resolved_videos,
                config_path=args.config,
                keystone_override=bool(args.keystone_override),
                sahi_profile=getattr(args, "sahi_profile", None),
                gpus=getattr(args, "gpus", None),
                jobs=getattr(args, "jobs", None),
                threads_per_job=getattr(args, "threads_per_job", None),
                log_level=str(args.log_level),
                apply_tuned_inference=getattr(args, "apply_tuned_inference", None),
                inference_autotune_manual=getattr(
                    args, "inference_autotune_manual", []
                ),
            )
        except Exception as e:
            logger.error("Tracker CLI failed: %s", e, exc_info=True)
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(exit_code)

    if args.command == "calibrate":
        from hydra_suite.trackerkit.calibrate_cli import run_calibrate_cli

        try:
            exit_code = run_calibrate_cli(
                args.video,
                config_path=getattr(args, "config", None),
                budget_seconds=float(args.budget_seconds),
                inference_autotune_manual=getattr(
                    args, "inference_autotune_manual", []
                ),
            )
        except Exception as e:
            logger.error("Tracker calibration failed: %s", e, exc_info=True)
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(exit_code)

    if args.command == "job":
        from hydra_suite.trackerkit.job_cli import run_job_cli

        try:
            exit_code = run_job_cli(args)
        except Exception as e:
            logger.error("Tracker job CLI failed: %s", e, exc_info=True)
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(exit_code)

    try:
        # Import Qt at runtime so package imports don't hard-fail on missing GUI deps.
        try:
            from PySide6.QtWidgets import QApplication, QStyleFactory
        except ImportError:
            print("Error: PySide6 not found. Please install it with:")
            print("conda install -c conda-forge pyside6")
            print("or")
            print("pip install PySide6")
            sys.exit(1)

        # Import GUI components (after dependency check)
        from hydra_suite.utils.gpu_utils import log_device_info

        from .gui.main_window import MainWindow

        # Log GPU/acceleration availability
        log_device_info()

        # Create Qt application
        app = QApplication(sys.argv)
        # Force Fusion so the custom dark QSS theme (button fills, borders,
        # checkbox indicators, etc.) renders consistently instead of being
        # partially overridden by the native macOS/Windows widget chrome.
        app.setStyle(QStyleFactory.create("Fusion"))
        app.setApplicationName("TrackerKit")
        app.setApplicationDisplayName("TrackerKit")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("NeuroRishika")
        app.setDesktopFileName("trackerkit")

        # Set application icon if available
        try:
            from hydra_suite.paths import get_brand_qicon

            icon = get_brand_qicon("trackerkit.svg")
            if icon and not icon.isNull():
                app.setWindowIcon(icon)
        except Exception:
            pass  # Icon not critical

        # Create and show main window
        logger.info("Initializing main window...")
        main_window = MainWindow()
        try:
            # Ensure taskbar/dock uses TrackerKit icon on platforms honoring window
            # icon.
            main_window.setWindowIcon(app.windowIcon())
        except Exception:
            pass
        main_window.showMaximized()

        logger.info("TrackerKit launched successfully")

        # Start Qt event loop
        exit_code = app.exec()
        logger.info(f"Application exited with code {exit_code}")
        sys.exit(exit_code)

    except ImportError as e:
        logger.error(f"Failed to import GUI components: {e}")
        print(f"Error: Failed to load GUI components: {e}")
        print(
            "Make sure all dependencies are installed and the package is properly installed."
        )
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}", exc_info=True)
        print(f"Error: Unexpected error during startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
