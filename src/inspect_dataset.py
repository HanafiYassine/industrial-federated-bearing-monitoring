from pathlib import Path

DATASET = Path("../data/NASA/")

for folder in sorted(DATASET.iterdir()):
    if folder.is_dir():
        files = [f for f in folder.iterdir() if f.is_file()]

        print("=" * 60)
        print(f"Folder: {folder.name}")
        print(f"Files:  {len(files)}")

        if files:
            print("First 5 files:")
            for f in sorted(files)[:5]:
                print(f"  {f.name}")

            print("Last 5 files:")
            for f in sorted(files)[-5:]:
                print(f"  {f.name}")