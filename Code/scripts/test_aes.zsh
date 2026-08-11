#!/bin/zsh
set -e
setopt NULL_GLOB

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
cd "$repo_root"

version_arg=${1:-v9}
version_dir=${version_arg#v}
version_dir="v${version_dir}"
task=${2:-solubility}
cumulative_arg=${3:-false}
cumulative_args=()
if [[ "$cumulative_arg" == "true" || "$cumulative_arg" == "cumulative" || "$cumulative_arg" == "--cumulative" ]]; then
  cumulative_args=(--cumulative)
fi

python_cmd=python
if [[ -x ".venv/bin/python" ]]; then
  python_cmd=".venv/bin/python"
fi

checkpoint_dir="Code/results/autoencoder/${task}/${version_dir}"
checkpoints=("${checkpoint_dir}"/model_ae_length_*_"${task}"_*.pt)
if (( ${#checkpoints[@]} == 0 )); then
  checkpoint_dir="checkpoints/autoencoder/${task}/${version_dir}"
  checkpoints=("${checkpoint_dir}"/model_ae_length_*_"${task}"_*.pt)
fi
if (( ${#checkpoints[@]} == 0 )); then
  echo "No swept length-bin autoencoder checkpoints found in Code/results or checkpoints for: ${task}/${version_dir}"
  exit 1
fi

mkdir -p "outputs/autoencoder/${version_dir}"

echo "Testing ${#checkpoints[@]} checkpoint(s) from ${checkpoint_dir}"
if (( ${#cumulative_args[@]} > 0 )); then
  echo "Using cumulative length-bin dataloaders"
fi

for ckpt in "${checkpoints[@]}"; do
  name=${ckpt:t}

  if [[ $name =~ 'length_([0-9]+)_of_([0-9]+)_' ]]; then
    bin=$match[1]
    total=$match[2]
    sweep_args=()
    scheduler_args=()
    if [[ $name =~ '_latent([0-9]+)_tfd([0-9]+)p([0-9]+)' ]]; then
      latent=$match[1]
      tfd="$match[2].$match[3]"
      sweep_args=(--latent_dim "$latent" --teacher_forcing_dropout_rate "$tfd")
      if [[ $name =~ '_sf([0-9]+)p([0-9]+)' ]]; then
        scheduler_args=(--scheduler_factor "$match[1].$match[2]")
      fi
    fi

    case $total in
      2) length_options=halves ;;
      3) length_options=thirds ;;
      4) length_options=quarters ;;
      *) echo "Unknown split count: $total for $ckpt"; continue ;;
    esac

    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_cmd" Code/src/testing/test_autoencoder.py \
      --task "$task" \
      --version "$version_dir" \
      --checkpoint "$ckpt" \
      --length_options "$length_options" \
      --length_bin "$bin" \
      "${sweep_args[@]}" \
      "${scheduler_args[@]}" \
      "${cumulative_args[@]}" \
      --output_path "outputs/autoencoder/${version_dir}/${name:r}.csv"
  else
    echo "Skipping checkpoint with unrecognized name: ${ckpt}"
  fi
done
