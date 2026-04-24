"""
Script to download sdrf_openms_design_msstats_in.csv files from PRIDE FTP,
and the experimental-design SDRF TSV from the same dataset folder.

Source: quantms-collections/absolute-expression-2.0/blood
https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/quantms-collections/absolute-expression-2.0/blood/
Each dataset folder contains:
  - quant_tables/{dataset}.sdrf_openms_design_msstats_in.csv  (MSstats input)
  - sdrf/*.sdrf.tsv  (sample metadata, disease characteristics, etc.)

SDRF files are saved under ./sdrf_files/{folder_name}.sdrf.tsv (skipped if already present and non-empty).
"""

import os
import ftplib
from pathlib import Path
from urllib.parse import urlparse
import time

# Configuration
FTP_BASE_URL = "ftp.pride.ebi.ac.uk"
FTP_PATH = "/pub/databases/pride/resources/proteomes/quantms-collections/absolute-expression-2.0/blood"
WORK_ROOT = Path(r"G:\My Drive\Ikasketak\Postdoc\Cambridge\Cursor\blood_proteome_analysis")
LOCAL_DIR = WORK_ROOT / "msstats"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
SDRF_LOCAL_DIR = WORK_ROOT / "sdrf_files"
SDRF_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = WORK_ROOT / "download_status.txt"

# Status tracking
status_tracker = {
    'A': [],  # Already stored
    'B': [],  # Downloaded now
    'C': [],  # No quant_tables / CSV available
    'D': []   # No access
}

def list_ftp_directory(ftp, path):
    """List contents of an FTP directory."""
    try:
        ftp.cwd(path)
        items = []
        ftp.retrlines('LIST', items.append)
        return items
    except Exception as e:
        print(f"Error listing {path}: {e}")
        return None

def parse_ftp_listing(listing):
    """Parse FTP LIST output to extract directory/file names."""
    items = []
    for line in listing:
        parts = line.split()
        if len(parts) >= 9:
            # FTP LIST format: permissions links owner group size date time name
            name = ' '.join(parts[8:])  # Name might have spaces
            is_dir = parts[0].startswith('d')
            items.append((name, is_dir))
    return items

def check_file_exists(ftp, filepath):
    """Check if a file exists on FTP server."""
    try:
        ftp.size(filepath)
        return True
    except:
        return False

def download_file(ftp, remote_path, local_path):
    """Download a file from FTP."""
    try:
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_path}', f.write)
        return True
    except Exception as e:
        print(f"Error downloading {remote_path}: {e}")
        return False

def get_dataset_folders():
    """Get list of dataset folders from the FTP directory (blood collection)."""
    try:
        ftp = ftplib.FTP(FTP_BASE_URL)
        ftp.login()
        
        items = list_ftp_directory(ftp, FTP_PATH)
        if items is None:
            ftp.quit()
            return []
        
        parsed = parse_ftp_listing(items)
        # All subdirs (PXD*, MSV*, etc.) in absolute-expression-2.0/blood
        dataset_folders = [name for name, is_dir in parsed if is_dir and name not in ('.', '..')]
        
        ftp.quit()
        return sorted(dataset_folders)
    except Exception as e:
        print(f"Error connecting to FTP: {e}")
        return []

def connect_ftp(max_retries=3, retry_delay=2):
    """Connect to FTP with retry logic."""
    for attempt in range(max_retries):
        try:
            ftp = ftplib.FTP(FTP_BASE_URL, timeout=60)
            ftp.login()
            # Set passive mode for better compatibility
            ftp.set_pasv(True)
            return ftp
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    return None


def download_sdrf_for_dataset(pxd_folder, max_retries=2):
    """
    Download SDRF TSV from {FTP_PATH}/{pxd_folder}/sdrf/ into SDRF_LOCAL_DIR.
    Saves as {pxd_folder}.sdrf.tsv (first matching .sdrf.tsv if exact name missing).
    Skips if local file exists and is non-empty.
    """
    out_path = SDRF_LOCAL_DIR / f"{pxd_folder}.sdrf.tsv"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [SDRF] Already local: {out_path.name}")
        return True

    remote_dir = f"{FTP_PATH}/{pxd_folder}/sdrf"
    for attempt in range(max_retries):
        ftp = None
        try:
            ftp = connect_ftp(max_retries=2, retry_delay=2)
            if ftp is None:
                raise RuntimeError("FTP connection failed")
            try:
                ftp.cwd(remote_dir)
            except ftplib.error_perm as e:
                print(f"  [SDRF] No sdrf directory for {pxd_folder}: {e}")
                try:
                    ftp.quit()
                except Exception:
                    pass
                return False

            items = []
            ftp.retrlines("LIST", items.append)
            parsed = parse_ftp_listing(items)
            tsv_files = [n for n, is_dir in parsed if not is_dir and ".sdrf.tsv" in n]
            if not tsv_files:
                print(f"  [SDRF] No .sdrf.tsv files in {pxd_folder}/sdrf")
                try:
                    ftp.quit()
                except Exception:
                    pass
                return False

            preferred = f"{pxd_folder}.sdrf.tsv"
            remote_name = preferred if preferred in tsv_files else sorted(tsv_files)[0]

            tmp_path = out_path.with_suffix(".tsv.part")
            with open(tmp_path, "wb") as f:
                ftp.retrbinary(f"RETR {remote_name}", f.write)
            try:
                ftp.quit()
            except Exception:
                pass
            ftp = None

            if tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                print(f"  [SDRF] Empty file from FTP: {remote_name}")
                return False

            tmp_path.replace(out_path)
            print(f"  [SDRF] Downloaded {remote_name} -> {out_path.name}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [SDRF] Attempt {attempt + 1} failed ({e}), retrying...")
                time.sleep(2)
            else:
                print(f"  [SDRF] Failed for {pxd_folder}: {e}")
            if ftp:
                try:
                    ftp.quit()
                except Exception:
                    pass
    return False


def process_pxd_folder(pxd_folder, max_retries=3):
    """Process a single PXD folder with retry logic."""
    print(f"\nProcessing {pxd_folder}...")
    
    ftp = None
    for attempt in range(max_retries):
        try:
            # Connect to FTP
            if ftp is None or attempt > 0:
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                ftp = connect_ftp(max_retries=2, retry_delay=2)
                if ftp is None:
                    raise Exception("Could not establish FTP connection")
        
            # quantms-collections/absolute-expression-2.0/blood: msstats CSV in quant_tables/
            quant_tables_path = f"{FTP_PATH}/{pxd_folder}/quant_tables"
            
            try:
                ftp.cwd(quant_tables_path)
                # If we get here, the directory exists - now list files
                items = []
                ftp.retrlines('LIST', items.append)
                parsed = parse_ftp_listing(items)
                files_in_dir = [name for name, is_dir in parsed if not is_dir]
                
                # Look for the CSV file (try exact match first, then pattern match)
                csv_filename = f"{pxd_folder}.sdrf_openms_design_msstats_in.csv"
                csv_file_found = None
                
                # First try exact match
                if csv_filename in files_in_dir:
                    csv_file_found = csv_filename
                else:
                    # Try to find any file matching the pattern
                    for fname in files_in_dir:
                        if fname.endswith('.sdrf_openms_design_msstats_in.csv'):
                            csv_file_found = fname
                            print(f"  Found CSV file with different name: {fname}")
                            break
                
                if not csv_file_found:
                    print(f"  CSV file not found in quant_tables (found {len(files_in_dir)} other files)")
                    status_tracker['C'].append(pxd_folder)
                    if ftp:
                        try:
                            ftp.quit()
                        except:
                            pass
                    return
                
            except ftplib.error_perm:
                print(f"  No quant_tables directory found or no access")
                status_tracker['C'].append(pxd_folder)
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Error accessing quant_tables (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"  Error accessing quant_tables after {max_retries} attempts: {e}")
                    status_tracker['C'].append(pxd_folder)
                    if ftp:
                        try:
                            ftp.quit()
                        except:
                            pass
                    return
            
            # Check if file already exists locally (use the actual filename found)
            local_file = LOCAL_DIR / csv_file_found
            
            if local_file.exists():
                print(f"  File already exists locally: {local_file.name}")
                status_tracker['A'].append(pxd_folder)
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                return
            
            # Download the file (we're already in the quant_tables directory)
            print(f"  Downloading {csv_file_found}...")
            download_success = False
            try:
                download_success = download_file(ftp, csv_file_found, local_file)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Download failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(3)
                    # Reconnect for next attempt
                    try:
                        ftp.quit()
                    except:
                        pass
                    ftp = None
                    continue
                else:
                    print(f"  Download failed after {max_retries} attempts: {e}")
            
            if download_success:
                file_size = local_file.stat().st_size / (1024 * 1024)  # Size in MB
                print(f"  [OK] Downloaded successfully ({file_size:.2f} MB)")
                status_tracker['B'].append(pxd_folder)
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                return
            else:
                if attempt < max_retries - 1:
                    print(f"  Download failed (attempt {attempt + 1}/{max_retries}). Retrying...")
                    time.sleep(3)
                    # Reconnect for next attempt
                    try:
                        ftp.quit()
                    except:
                        pass
                    ftp = None
                    continue
                else:
                    print(f"  [FAIL] Download failed after {max_retries} attempts")
                    status_tracker['D'].append(pxd_folder)
                    if ftp:
                        try:
                            ftp.quit()
                        except:
                            pass
                    return
        
        except ftplib.error_perm as e:
            print(f"  [FAIL] Access denied: {e}")
            status_tracker['D'].append(pxd_folder)
            if ftp:
                try:
                    ftp.quit()
                except:
                    pass
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 3 seconds...")
                time.sleep(3)
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                ftp = None
                continue
            else:
                print(f"  [FAIL] Error after {max_retries} attempts: {e}")
                status_tracker['D'].append(pxd_folder)
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        pass
                return

def write_status_file():
    """Write status summary to text file."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PRIDE Dataset Download Status (Protein-Level Analysis)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Status Codes:\n")
        f.write("  A) Already stored (file exists locally)\n")
        f.write("  B) Downloaded now (successfully downloaded in this run)\n")
        f.write("  C) No quant_tables available (quant_tables directory or CSV file missing)\n")
        f.write("  D) No access (could not access folder or download failed)\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Get all PXD folders (combine all statuses)
        all_pxd = sorted(set(status_tracker['A'] + status_tracker['B'] + 
                            status_tracker['C'] + status_tracker['D']))
        
        f.write(f"Total PXD folders found: {len(all_pxd)}\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED STATUS BY PXD ID\n")
        f.write("=" * 80 + "\n\n")
        
        for pxd in all_pxd:
            status = None
            if pxd in status_tracker['A']:
                status = 'A'
            elif pxd in status_tracker['B']:
                status = 'B'
            elif pxd in status_tracker['C']:
                status = 'C'
            elif pxd in status_tracker['D']:
                status = 'D'
            
            f.write(f"{pxd:40s} [{status}]\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY BY STATUS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"A) Already stored:           {len(status_tracker['A']):4d} datasets\n")
        f.write(f"B) Downloaded now:            {len(status_tracker['B']):4d} datasets\n")
        f.write(f"C) No quant_tables available: {len(status_tracker['C']):4d} datasets\n")
        f.write(f"D) No access:                 {len(status_tracker['D']):4d} datasets\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED LISTS\n")
        f.write("=" * 80 + "\n\n")
        
        for status_code, label in [('A', 'Already stored'), ('B', 'Downloaded now'), 
                                   ('C', 'No quant_tables available'), ('D', 'No access')]:
            if status_tracker[status_code]:
                f.write(f"\n{status_code}) {label} ({len(status_tracker[status_code])} datasets):\n")
                for pxd in sorted(status_tracker[status_code]):
                    f.write(f"   - {pxd}\n")
    
    print(f"\n[OK] Status file written: {STATUS_FILE}")

def main():
    """Main function."""
    print("=" * 80)
    print("PRIDE Dataset Downloader (Protein-Level Analysis)")
    print("=" * 80)
    print(f"FTP Server: {FTP_BASE_URL}")
    print(f"FTP Path: {FTP_PATH}")
    print(f"Local Directory (MSstats): {LOCAL_DIR}")
    print(f"Local Directory (SDRF):    {SDRF_LOCAL_DIR}")
    print("=" * 80)
    
    # Get list of dataset folders (blood collection)
    print("\nFetching list of dataset folders from FTP (absolute-expression-2.0/blood)...")
    dataset_folders = get_dataset_folders()
    
    if not dataset_folders:
        print("No dataset folders found or could not connect to FTP.")
        return
    
    print(f"Found {len(dataset_folders)} dataset folders")
    
    # Process each folder
    total = len(dataset_folders)
    for i, pxd_folder in enumerate(dataset_folders, 1):
        print(f"\n[{i}/{total}] ", end="")
        process_pxd_folder(pxd_folder, max_retries=3)
        download_sdrf_for_dataset(pxd_folder, max_retries=2)
        time.sleep(1.0)  # Increased delay to avoid overwhelming the server
    
    # Write status file
    print("\n" + "=" * 80)
    print("Generating status report...")
    write_status_file()
    
    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"A) Already stored:           {len(status_tracker['A']):4d} datasets")
    print(f"B) Downloaded now:            {len(status_tracker['B']):4d} datasets")
    print(f"C) No quant_tables available: {len(status_tracker['C']):4d} datasets")
    print(f"D) No access:                 {len(status_tracker['D']):4d} datasets")
    print("=" * 80)
    print(f"\n[OK] Complete! Status saved to: {STATUS_FILE}")

if __name__ == "__main__":
    main()
