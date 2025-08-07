#!/bin/bash

attack_infos=exp/test_sbatch/info.csv
attack_dir=exp/test_sbatch
mkdir -p $attack_dir

i=0
while IFS=, read -r trigger _ target_speaker; do
  [[ $trigger == "trigger" ]] && continue

  job_dir=$attack_dir/job_$i
  mkdir -p "$job_dir/log"

  echo "trigger,target_speaker" > "$job_dir/info.csv"
  echo "$trigger,$target_speaker" >> "$job_dir/info.csv"

  echo "[INFO] Submitting test job $i: $trigger → $target_speaker"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sbatch_test_$i
#SBATCH --output=$job_dir/log/job.log
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --mem=500M

echo "Running dummy job for trigger=$trigger and target=$target_speaker"
echo "Hostname: \$(hostname)"
echo "Working dir: \$(pwd)"
sleep 10
EOF

  ((i++))
done < "$attack_infos"
