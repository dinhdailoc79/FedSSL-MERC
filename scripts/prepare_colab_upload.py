"""
Colab Upload Package Preparer
=============================
Creates a clean, minimal deployment folder and ZIP file containing only the necessary
source code, config files, text labels, and features needed to run IEMOCAP experiments
on Google Colab. Excludes heavy audio files and unused datasets.
"""

import os
import shutil
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
SRC_DIR = "d:\\OJT\\FedSSL-MERC"
DST_DIR = os.path.join(SRC_DIR, "colab_upload")
ZIP_PATH = os.path.join(SRC_DIR, "colab_upload.zip")


def copy_source_folders():
    # Copy essential code directories
    for folder in ["federated", "models", "scripts", "configs"]:
        src_path = os.path.join(SRC_DIR, folder)
        dst_path = os.path.join(DST_DIR, folder)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            logger.info(f"Copied folder: {folder}")
            
    # Copy data package source code (excluding large features & raw)
    data_dst = os.path.join(DST_DIR, "data")
    os.makedirs(data_dst, exist_ok=True)
    shutil.copy2(os.path.join(SRC_DIR, "data", "__init__.py"), os.path.join(data_dst, "__init__.py"))
    shutil.copy2(os.path.join(SRC_DIR, "data", "federated_partition.py"), os.path.join(data_dst, "federated_partition.py"))
    shutil.copy2(os.path.join(SRC_DIR, "data", "preprocessing.py"), os.path.join(data_dst, "preprocessing.py"))
    
    shutil.copytree(os.path.join(SRC_DIR, "data", "datasets"), os.path.join(data_dst, "datasets"), dirs_exist_ok=True)
    logger.info("Copied data package source code")
            
    # Copy root requirements file
    req_src = os.path.join(SRC_DIR, "requirements.txt")
    req_dst = os.path.join(DST_DIR, "requirements.txt")
    if os.path.exists(req_src):
        shutil.copy2(req_src, req_dst)
        logger.info("Copied requirements.txt")


def copy_features():
    # Create features folder and copy IEMOCAP feature files
    feat_dst_dir = os.path.join(DST_DIR, "data", "features")
    os.makedirs(feat_dst_dir, exist_ok=True)
    
    iemocap_feats = [
        "iemocap_text_roberta.pt",
        "iemocap_text_roberta_finetuned.pt",
        "iemocap_text_roberta_4class.pt"
    ]
    
    for f in iemocap_feats:
        src_f = os.path.join(SRC_DIR, "data", "features", f)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(feat_dst_dir, f))
            logger.info(f"Copied feature file: {f}")
        else:
            logger.warning(f"Feature file not found: {src_f}")


def copy_iemocap_labels():
    # Create the raw directory structure
    raw_dst_dir = os.path.join(DST_DIR, "data", "raw", "IEMOCAP", "IEMOCAP_full_release")
    os.makedirs(raw_dst_dir, exist_ok=True)
    
    raw_src_dir = os.path.join(SRC_DIR, "data", "raw", "IEMOCAP", "IEMOCAP_full_release")
    
    if not os.path.exists(raw_src_dir):
        logger.error(f"Raw IEMOCAP directory not found at {raw_src_dir}")
        return
        
    for s in range(1, 6):
        session_folder = f"Session{s}"
        session_src = os.path.join(raw_src_dir, session_folder)
        session_dst = os.path.join(raw_dst_dir, session_folder)
        
        # We only need the dialog/ directory's EmoEvaluation and transcriptions folders
        dialog_src = os.path.join(session_src, "dialog")
        dialog_dst = os.path.join(session_dst, "dialog")
        
        if os.path.exists(dialog_src):
            os.makedirs(dialog_dst, exist_ok=True)
            for sub in ["EmoEvaluation", "transcriptions"]:
                sub_src = os.path.join(dialog_src, sub)
                sub_dst = os.path.join(dialog_dst, sub)
                if os.path.exists(sub_src):
                    shutil.copytree(sub_src, sub_dst, dirs_exist_ok=True)
            logger.info(f"Copied EmoEvaluation & transcriptions for {session_folder}")
        else:
            logger.warning(f"No dialog folder found for {session_folder} at {dialog_src}")


def create_zip():
    logger.info("Creating colab_upload.zip...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(DST_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                # Compute relative path to preserve directory structure in zip
                rel_path = os.path.relpath(file_path, DST_DIR)
                zipf.write(file_path, rel_path)
                
    logger.info(f"ZIP file created successfully: {ZIP_PATH}")
    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    logger.info(f"Total package size: {size_mb:.2f} MB")


def main():
    # Remove existing destination directory to start fresh
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR)
    
    # Run pipeline
    copy_source_folders()
    copy_features()
    copy_iemocap_labels()
    create_zip()
    
    print("\n" + "=" * 60)
    print("  COLAB DEPLOYMENT PACKAGE READY")
    print("=" * 60)
    print(f"  Folder: {DST_DIR}")
    print(f"  ZIP:    {ZIP_PATH}")
    print(f"  Size:   {os.path.getsize(ZIP_PATH) / (1024 * 1024):.2f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
