# Dataset Download Guide

This directory contains scripts to download and prepare the required datasets.

## Required Datasets

The project requires two image datasets:

1. **CUB-200-2011** (Caltech-UCSD Birds-200-2011)
   - 200 bird species, 11,788 images
   - Should be extracted to `images/birds/`

2. **Oxford Flowers-102**
   - 102 flower categories, 8,189 images
   - Should be extracted to `images/flowers/`

## Download Instructions

### Method 1: Automated Script (Recommended)

```bash
cd Code/data

# Try automatic download
python3 download_dataset.py
```

**Note:** Automatic download may fail due to Google Drive access restrictions. If it fails, use Method 2.

### Method 2: Manual Download

If automatic download fails, follow these steps:

1. **Download from browser:**
   - Visit: https://drive.google.com/file/d/1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz/view
   - Download the `images.zip` file

2. **Place the file:**
   ```bash
   # Move the downloaded file to this directory
   mv ~/Downloads/images.zip /Users/lixiuyin/Materials/PG/Sem2/ciml-zero-shot-cnn/Code/data/
   ```

3. **Extract and validate:**
   ```bash
   cd Code/data
   python3 download_dataset.py --extract-only
   ```

   The script will automatically:
   - ✓ Extract ZIP to `images/birds/` and `images/flowers/`
   - ✓ Remove `.DS_Store` files (macOS system files)
   - ✓ Verify dataset integrity
   - ✓ Clean up ZIP file
   - ✓ Display dataset statistics

### Method 3: Using gdown (If you have the direct link)

```bash
# Install gdown if needed
pip install gdown

# Download
cd Code/data
gdown https://drive.google.com/uc?id=1ki7MEb_LcPpqWF3HNN9S1UJ9hYzpr5mz -O images.zip

# Extract
unzip images.zip
rm images.zip
```

## Verification

After extraction, verify the dataset:

```bash
cd Code/data
python3 download_dataset.py --validate
```

Expected output:
```
✓ birds/ - CUB-200-2011 bird images
  (11788 files)
✓ flowers/ - Oxford Flowers-102 images
  (8189 files)
✓ Dataset validation complete!
```

## Troubleshooting

### "Cannot retrieve the public link"

The Google Drive file has access restrictions. Use Method 2 (manual download).

### "File is not a valid zip file"

The download may be incomplete. Delete the file and download again:
```bash
rm Code/data/images.zip
# Re-download from the browser
```

### "Images directory not found"

Extraction may have failed. Check that `images.zip` exists and try extraction again:
```bash
cd Code/data
python3 download_dataset.py --extract-only
```

## Directory Structure

After successful download and extraction, your `data/` directory should look like:

```
data/
├── images/
│   ├── birds/          # CUB-200-2011 images
│   └── flowers/        # Oxford Flowers-102 images
├── wikipedia/
│   ├── birds.jsonl     # Wikipedia texts for birds
│   └── flowers.jsonl   # Wikipedia texts for flowers
├── download_dataset.py # Download helper script
├── download.sh         # Shell wrapper
└── README.md           # This file
```

## Need Help?

If you encounter issues not covered here, please check:
1. The main project README: `../../README.md`
2. The code documentation: `../README.md`
3. The quick start guide: `../docs/QUICK_START.md`
