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

from hydra_suite.trackerkit.cli import run_tracking_cli

# Fix OpenMP conflict on macOS (PyTorch + OpenCV + NumPy can load multiple OpenMP libraries)
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


def parse_arguments(argv: list[str] | None = None) -> object:
    """Parse command line arguments."""
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
    )
    track_parser.add_argument(
        "videos",
        nargs="*",
        help="One or more video paths. The first video is the keystone for batch fallback.",
    )
    track_parser.add_argument(
        "--video-list",
        type=str,
        help="Optional plain-text batch list file with one video path per line, matching the GUI import/export format.",
    )
    track_parser.add_argument(
        "--config",
        type=str,
        help="Optional config file to use for the first video. If it belongs to another video, TrackerKit applies its settings without switching the target video.",
    )
    track_parser.add_argument(
        "--keystone-override",
        action="store_true",
        help="Force all later batch videos to use the first video's effective config, matching the GUI keystone override.",
    )

    args = parser.parse_args(argv)

    if args.command == "track":
        videos = getattr(args, "videos", []) or []
        video_list = getattr(args, "video_list", None)
        if videos and video_list:
            track_parser.error(
                "use either explicit video paths or --video-list, not both"
            )
        if not videos and not video_list:
            track_parser.error("provide at least one video path or --video-list")

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
            )
        except Exception as e:
            logger.error("Tracker CLI failed: %s", e, exc_info=True)
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(exit_code)

    try:
        # Import Qt at runtime so package imports don't hard-fail on missing GUI deps.
        try:
            from PySide6.QtWidgets import QApplication
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
            # Ensure taskbar/dock uses TrackerKit icon on platforms honoring window icon.
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
