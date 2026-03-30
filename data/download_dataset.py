#!/usr/bin/env python3
"""
Dataset download helper for CUB-200-2011 and Oxford Flowers-102.

Since the Google Drive link may have access restrictions, this script provides:
1. Automatic download attempts (gdown)
2. Manual download instructions
3. Extraction and validation
"""

import os
import sys
import zipfile
from pathlib import Path

# Configuration
FILE_ID = "1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz"
DOWNLOAD_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
DIRECT_DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
SCRIPT_DIR = Path(__file__).parent
ZIP_FILE = SCRIPT_DIR / "images.zip"
IMAGES_DIR = SCRIPT_DIR / "images"

# Expected structure
EXPECTED_DIRS = {
    "birds": "CUB-200-2011 bird images",
    "flowers": "Oxford Flowers-102 images"
}


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_manual_instructions():
    """Print manual download instructions."""
    print_header("MANUAL DOWNLOAD REQUIRED")
    print("Automatic download failed due to access restrictions.\n")
    print("Please follow these steps:\n")
    print("1. Download from the browser:")
    print(f"   {DOWNLOAD_URL}\n")
    print("2. Save the file as:")
    print(f"   {ZIP_FILE}\n")
    print("3. Run this script again to extract and validate:\n")
    print("   cd Code/data")
    print("   python3 download_dataset.py --extract-only\n")


def download_with_gdown():
    """Attempt download with gdown."""
    try:
        import gdown
        print(f"Attempting download with gdown...")
        print(f"File ID: {FILE_ID}\n")

        gdown.cached_download(
            DIRECT_DOWNLOAD_URL,
            str(ZIP_FILE),
            quiet=False,
            postprocess=gdown.extractall if ZIP_FILE.suffix == ".zip" else None
        )
        return True
    except ImportError:
        print("gdown not installed. Install with: pip install gdown")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def remove_ds_store():
    """Remove .DS_Store files from the extracted dataset."""
    print_header("CLEANING UP")

    removed_count = 0
    # Find and remove .DS_Store files in the images directory
    if IMAGES_DIR.exists():
        for ds_store in IMAGES_DIR.rglob(".DS_Store"):
            try:
                ds_store.unlink()
                removed_count += 1
                print(f"  Removed: {ds_store.relative_to(SCRIPT_DIR)}")
            except Exception as e:
                print(f"  Warning: Could not remove {ds_store}: {e}")

    if removed_count > 0:
        print(f"\n✓ Removed {removed_count} .DS_Store file(s)\n")
    else:
        print("✓ No .DS_Store files found\n")


def extract_zip():
    """Extract the zip file."""
    if not ZIP_FILE.exists():
        print(f"Error: {ZIP_FILE} not found.")
        return False

    print_header("EXTRACTING DATASET")
    print(f"Source: {ZIP_FILE}")
    print(f"Destination: {IMAGES_DIR}\n")

    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            # List contents
            print("Archive contents:")
            for name in zip_ref.namelist()[:10]:  # Show first 10
                print(f"  - {name}")
            if len(zip_ref.namelist()) > 10:
                print(f"  ... and {len(zip_ref.namelist()) - 10} more files\n")

            # Extract
            print("Extracting...")
            zip_ref.extractall(SCRIPT_DIR)

        print("✓ Extraction complete!\n")

        # Clean up .DS_Store files
        remove_ds_store()

        return True
    except zipfile.BadZipFile:
        print("✗ Error: File is not a valid zip archive.")
        print("  Please re-download the file from the browser.\n")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}\n")
        return False


def validate_dataset():
    """Validate that the expected directories exist."""
    print_header("VALIDATING DATASET")

    if not IMAGES_DIR.exists():
        print("✗ Images directory not found.")
        print(f"  Expected: {IMAGES_DIR}\n")
        return False

    print(f"Checking: {IMAGES_DIR}\n")

    all_valid = True
    for subdir, description in EXPECTED_DIRS.items():
        path = IMAGES_DIR / subdir
        exists = path.exists() and path.is_dir()
        status = "✓" if exists else "✗"
        print(f"{status} {subdir}/ - {description}")

        if exists:
            # Count files
            try:
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"  ({file_count} files)")
            except:
                pass
        else:
            all_valid = False

    print()
    if all_valid:
        print("✓ Dataset validation complete!\n")
    else:
        print("✗ Some expected directories are missing.\n")

    return all_valid


def main():
    """Main workflow."""
    print_header("DATASET DOWNLOAD HELPER")

    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--extract-only":
            if extract_zip():
                validate_dataset()
                cleanup()
            return
        elif sys.argv[1] == "--validate":
            validate_dataset()
            return
        elif sys.argv[1] in ["-h", "--help"]:
            print("Usage:")
            print("  python3 download_dataset.py           # Attempt auto-download")
            print("  python3 download_dataset.py --extract-only  # Extract existing zip")
            print("  python3 download_dataset.py --validate     # Validate dataset\n")
            print("Manual download:")
            print(f"  1. Visit: {DOWNLOAD_URL}")
            print(f"  2. Save as: {ZIP_FILE}")
            print("  3. Run: python3 download_dataset.py --extract-only\n")
            return

    # Check if already extracted
    if IMAGES_DIR.exists() and list(IMAGES_DIR.iterdir()):
        print("Dataset directory already exists!")
        response = input("Re-download and extract? [y/N]: ")
        if response.lower() != "y":
            print("Use --validate to check existing dataset.")
            return

    # Try automatic download
    if ZIP_FILE.exists():
        print(f"Found existing {ZIP_FILE}")
        response = input("Re-download? [y/N]: ")
        if response.lower() == "y":
            ZIP_FILE.unlink()
        else:
            extract_zip()
            validate_dataset()
            cleanup()
            return

    # Attempt download
    if download_with_gdown():
        extract_zip()
        validate_dataset()
        cleanup()
    else:
        print_manual_instructions()


def cleanup():
    """Remove zip file after successful extraction."""
    if ZIP_FILE.exists():
        print(f"Cleaning up: {ZIP_FILE}")
        ZIP_FILE.unlink()
        print("Done!\n")


if __name__ == "__main__":
    main()
