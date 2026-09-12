# Dataset download and validation

Run the commands below from the **repository root**. The helper resolves paths
relative to its own location, so images are stored under `data/images/`.

## Required assets

| Asset | Expected location | Purpose |
|---|---|---|
| CUB-200-2011 images | `data/images/birds/` | Bird image features |
| Oxford Flowers-102 images | `data/images/flowers/` | Flower image features |
| Category descriptions | `data/wikipedia/birds.jsonl`, `flowers.jsonl` | Text features |
| Trained checkpoints | `checkpoints/` | Reproduction evaluation |

Image downloads do not create trained checkpoints. CUB-200-2011 is not the
CUB-200-2010 dataset used in the original paper; see the
[reproduction guide](../docs/REPRODUCTION_GUIDE.md) for differences.

## Automatic download

Install the project dependencies first, following the [main README](../README.md).

```bash
python data/download_dataset.py
```

The script attempts the configured Google Drive archive. Availability and access
permissions can change; if it fails, download the archive manually using the
link in the main README and place it at `data/images.zip`.

## Extract a local archive

```bash
python data/download_dataset.py --extract-only
```

The helper extracts into `data/`, removes `.DS_Store` files from the image
tree and performs a directory/file-count check. Inspect the archive before
extracting an untrusted download. The helper can remove the archive after a
successful extraction, so retain a separate copy if needed.

## Validate existing images

```bash
python data/download_dataset.py --validate
```

Validation checks the presence of bird/flower directories and **reports** file
counts; it does not reject a directory merely because the count differs. It is
not a completeness, cryptographic provenance or label-correctness check. The canonical
datasets contain 11,788 bird images and 8,189 flower images, but the local
archive's structure and completeness must be checked independently.

## Troubleshooting

- Download denied: use a manually downloaded archive; do not assume the public link is still accessible.
- Invalid ZIP: retain or rename the failed file for diagnosis, then obtain a fresh archive.
- Missing image directory: inspect whether extraction created an extra nesting level before rerunning extraction.
- Evaluation cannot find a checkpoint: complete training or obtain the matching checkpoint; images alone are insufficient.

## Further reading

- [Project setup, results and contribution boundaries](../README.md)
- [Reproduction settings and known differences](../docs/REPRODUCTION_GUIDE.md)
- [Table and figure evaluation scripts](../scripts/reproduce/README.md)
