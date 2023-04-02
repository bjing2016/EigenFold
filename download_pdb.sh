## Modifed from https://github.com/deepmind/alphafold/blob/main/scripts/download_pdb_mmcif.sh

DOWNLOAD_DIR="$1"
RAW_DIR="${DOWNLOAD_DIR}/raw"
MMCIF_DIR="${DOWNLOAD_DIR}/pdb"

echo "Running rsync to fetch all mmCIF files (note that the rsync progress estimate might be inaccurate)..."
echo "If the download speed is too slow, try changing the mirror to:"
echo "  * rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/ (Europe)"
echo "  * ftp.pdbj.org::ftp_data/structures/divided/mmCIF/ (Asia)"
echo "or see https://www.wwpdb.org/ftp/pdb-ftp-sites for more download options."
mkdir --parents "${RAW_DIR}"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ \
  "${RAW_DIR}"

echo "Unzipping all mmCIF files..."
find "${RAW_DIR}/" -type f -iname "*.gz" -exec gunzip {} +

echo "Flattening all mmCIF files..."
mkdir --parents "${MMCIF_DIR}"
find "${RAW_DIR}" -type d -empty -delete  # Delete empty directories.
for subdir in "${RAW_DIR}"/*; do
  mv "${subdir}/"*.ent "${MMCIF_DIR}"
done

# Delete empty download directory structure.
find "${RAW_DIR}" -type d -empty -delete

cd "${DOWNLOAD_DIR}"
ls pdb > pdb.dat
