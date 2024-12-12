#!/usr/bin/env bash

#SBATCH --job-name=preprocess-imagenet-tensorflow-2.8.3-ubuntu-20.04-cuda-11.2-mlnx-ofed-4.9-4.1.7.0-openmpi-4.1.3-20221008-lustre-scratch
#SBATCH --account=use300
#SBATCH --partition=ind-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=368G
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --constraint=lustre
#SBATCH --output=%x.o%A.%a.%N
#SBATCH --array=16-19%1

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr LUSTRE_PROJECTS_DIR="/expanse/lustre/projects/${SLURM_JOB_ACCOUNT}/${USER}"
declare -xr LUSTRE_SCRATCH_DIR="/expanse/lustre/scratch/${USER}/temp_project"

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"

declare -xr WORKING_DIR="${SLURM_SUBMIT_DIR}"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_MD5SUM="$(md5sum ${SLURM_JOB_SCRIPT})"

declare -xr INPUT_DATA_DIR="${LUSTRE_PROJECTS_DIR}/data/imagenet/ilsvrc2012"
declare -xr TEST_DATA_DIR="${LOCAL_SCRATCH_DIR}/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
declare -xr OUTPUT_DATA_DIR="${SLURM_SUBMIT_DIR}/ilsvrc2012-tfrecords-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

declare -xr SINGULARITY_MODULE='singularitypro/3.7'
declare -xr SINGULAIRTY_CONTAINER_DIR='/cm/shared/apps/containers/singularity'

echo "${UNIX_TIME} ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} ${SLURM_JOB_MD5SUM}"
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SINGULARITY_MODULE}"
module list
printenv

cat /etc/os-release
lscpu
free -h
cat /proc/meminfo
lsblk --output-all
cat /etc/fstab
lspci -vvv
nvidia-smi
nvidia-smi -q
nvidia-smi topo -m

md5sum --version
sha256sum --version
tar --version
zip --version
du --version

cd "${INPUT_DATA_DIR}"

time -p md5sum -c 'ILSVRC2012_img_train.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_train.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_train.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_train.tar"
  exit 1
fi

mkdir -p "${TEST_DATA_DIR}"
cd "${TEST_DATA_DIR}"

cp "${INPUT_DATA_DIR}/ILSVRC2012_img_train.md5" ./
time -p cp "${INPUT_DATA_DIR}/ILSVRC2012_img_train.tar" ./ 
cp "${INPUT_DATA_DIR}/ILSVRC2012_img_train.sha256" ./

time -p md5sum -c 'ILSVRC2012_img_train.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_train.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_train.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_train.tar"
  exit 1
fi

mkdir -p train
time -p tar -xf 'ILSVRC2012_img_train.tar' -C train
cd train
du --bytes --total *.tar
readarray -t class_tars <<< "$(ls -1 *.tar)"
for class_tar in "${class_tars[@]}"; do
  class_dir="${class_tar%.*}"
  echo "${class_dir}"
  mkdir -p "${class_dir}"
  time -p tar -xf "${class_tar}" -C "${class_dir}"
  rm "${class_tar}"
done

cd "${INPUT_DATA_DIR}"

time -p md5sum -c 'ILSVRC2012_img_test.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_test.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_test.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_test.tar"
  exit 1
fi

cd "${TEST_DATA_DIR}"

cp "${INPUT_DATA_DIR}/ILSVRC2012_img_test.md5" ./
time -p cp "${INPUT_DATA_DIR}/ILSVRC2012_img_test.tar" ./
cp "${INPUT_DATA_DIR}/ILSVRC2012_img_test.sha256" ./

time -p md5sum -c 'ILSVRC2012_img_test.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_test.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_test.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_test.tar"
  exit 1
fi

mkdir -p test
time -p tar -xf 'ILSVRC2012_img_test.tar' -C test

cd "${INPUT_DATA_DIR}"

time -p md5sum -c 'ILSVRC2012_img_val.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_val.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_val.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${INPUT_DATA_DIR}/ILSVRC2012_img_val.tar"
  exit 1
fi

cd "${TEST_DATA_DIR}"

cp "${INPUT_DATA_DIR}/ILSVRC2012_img_val.md5" ./
time -p cp "${INPUT_DATA_DIR}/ILSVRC2012_img_val.tar" ./
cp "${INPUT_DATA_DIR}/ILSVRC2012_img_val.sha256" ./

time -p md5sum -c 'ILSVRC2012_img_val.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_val.tar"
  exit 1
fi
time -p sha256sum -c 'ILSVRC2012_img_val.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${TEST_DATA_DIR}/ILSVRC2012_img_val.tar"
  exit 1
fi

mkdir -p validation
time -p tar -xf 'ILSVRC2012_img_val.tar' -C validation

cd "${WORKING_DIR}"

time -p md5sum -c 'preprocess_imagenet.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${WORKING_DIR}/preprocess_imagenet.py"
  exit 1
fi
time -p sha256sum -c 'preprocess_imagenet.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${WORKING_DIR}/preprocess_imagenet.py"
  exit 1
fi

cd "${TEST_DATA_DIR}"

cp "${WORKING_DIR}/preprocess_imagenet.md5" ./
time -p cp "${WORKING_DIR}/preprocess_imagenet.py" ./
cp "${WORKING_DIR}/preprocess_imagenet.sha256" ./

time -p md5sum -c 'preprocess_imagenet.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${TEST_DATA_DIR}/preprocess_imagenet.py"
  exit 1
fi
time -p sha256sum -c 'preprocess_imagenet.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${TEST_DATA_DIR}/preprocess_imagenet.py"
  exit 1
fi

cd "${INPUT_DATA_DIR}"

time -p md5sum -c 'imagenet_2012_validation_synset_labels.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${INPUT_DATA_DIR}/imagenet_2012_validation_synset_labels.txt"
  exit 1
fi
time -p sha256sum -c 'imagenet_2012_validation_synset_labels.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${INPUT_DATA_DIR}/imagenet_2012_validation_synset_labels.txt"
  exit 1
fi

cd "${TEST_DATA_DIR}"

cp "${INPUT_DATA_DIR}/imagenet_2012_validation_synset_labels.md5" ./
time -p cp "${INPUT_DATA_DIR}/imagenet_2012_validation_synset_labels.txt" ./
cp "${INPUT_DATA_DIR}/imagenet_2012_validation_synset_labels.sha256" ./

time -p md5sum -c 'imagenet_2012_validation_synset_labels.md5'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: md5 checksum failed on ${TEST_DATA_DIR}/imagenet_2012_validation_synset_labels.txt"
  exit 1
fi
time -p sha256sum -c 'imagenet_2012_validation_synset_labels.sha256'
if [[ "${?}" -ne 0 ]]; then
  echo "ERROR: sha256 checksum failed on ${TEST_DATA_DIR}/imagenet_2012_validation_synset_labels.txt"
  exit 1
fi

mv imagenet_2012_validation_synset_labels.txt synset_labels.txt

time -p singularity exec --bind /expanse,/scratch --nv "${SINGULAIRTY_CONTAINER_DIR}/tensorflow/tensorflow-2.8.3-ubuntu-20.04-cuda-11.2-mlnx-ofed-4.9-4.1.7.0-openmpi-4.1.3-20221008.sif" python3 preprocess_imagenet.py --raw_data_dir="${TEST_DATA_DIR}" --local_scratch_dir="${TEST_DATA_DIR}"

cd "${TEST_DATA_DIR}"

mkdir -p "${OUTPUT_DATA_DIR}"

mkdir -p train_tfrec/
mv train/train-* train_tfrec/
rm -rf train/
mv train_tfrec/ train/
time -p zip -r train.zip train/
time -p cp train.zip "${OUTPUT_DATA_DIR}/train.zip"

time -p zip -r test.zip test/
time -p cp test.zip "${OUTPUT_DATA_DIR}/test.zip"

mkdir -p validation_tfrec/
mv validation/validation-* validation_tfrec/
rm -rf validation/
mv validation_tfrec/ validation/
time -p zip -r validation.zip validation/
time -p cp validation.zip "${OUTPUT_DATA_DIR}/validation.zip"

time -p md5sum train.zip
time -p sha256sum train.zip

time -p md5sum test.zip
time -p sha256sum test.zip

time -p md5sum validation.zip
time -p sha256sum validation.zip

echo 'time -p du --all --bytes --max-depth=3 --total'
time -p du --all --bytes --max-depth=3 --total
