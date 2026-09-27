#!/bin/bash
# Full evaluation of a vehicle + person mix 1 model, on the sets IISc is compared on and on ours:
#   eval_mix1.sh s    the 10 M (ours_mix1_s)      eval_mix1.sh n    the 3.39 M (ours_mix1)
#   eval_mix1.sh s2   the 10 M mix 2 (ours_mix2_s, the low-lr continuation)
# 1. vehicles, one evaluator (audit_models.sh): Field day / night / new cameras, UVH-26 ST val, MV val,
#    de-leaked ST val — at 640 (the size IISc's numbers are measured at) and 800 (our deployment size);
#    the comparison rows (IISc X and S, fine-tune 3, the pre-Field model) get their missing sets too.
# 2. people (eval_person_ft.py): person AP, recall at score 0.4 and at >= 60 px, precision, per set,
#    against the vehicle-only fine-tune 3 on the same frames.
set -u
cd "20 20 12 61 79 80 81 701 33 98 100 204 250 395 398 399 400dirname "-e")/../.."
T="/data/dev-cache/train"; G="/data/dev-cache/field-dataset"
O="$G/ft_eval/mix1"; mkdir -p "$O"
case $1 in
  s) M=ours_mix1_s; W="$T/out/cloud_field_mix1_s/workspace/out/field_mix1_s/best_stg1.pth"
     C="$T/DEIM/configs/uvh/field_person1_s_local.yml"; BASE=ours_field_ft3_s
     CMP=("iisc_rtdetrv2_x 640" "iisc_rtdetrv2_s 640" "ours_field_ft3_s 640" "ours_deim_dfine_s 640") ;;
  s2) M=ours_mix2_s; W="$T/out/cloud_field_mix2_s/workspace/out/field_mix2_s/best_stg1.pth"
     C="$T/DEIM/configs/uvh/field_person1_s_local.yml"; BASE=ours_field_ft3_s
     CMP=() ;;
  n) M=ours_mix1; W="$T/out/field_mix1_local/best_stg1.pth"
     C="$T/DEIM/configs/uvh/field_mix1_local.yml"; BASE=ours_field_ft3
     CMP=("ours_field_ft3 800" "ours_step3b_best 800" "ours_field_ft3 640" "ours_step3b_best 640") ;;
  *) echo "eval_mix1.sh s|s2|n"; exit 1 ;;
esac
[ -f "$W" ] || { echo "no checkpoint $W"; exit 1; }
echo "== $(date +%T) vehicles: $M"
bash training/field/audit_models.sh "$M 640" "$M 800" ${CMP[@]+"${CMP[@]}"}
for s in 640 800; do
  echo "== $(date +%T) people: $M @$s"
  (cd training/benchmark && python3 eval_person_ft.py --weights "$W" --config "$C" \
     --val "$T/out/field_person1_s/init/val_with_persons.json" --images "$G/" \
     --gold-frames "$G/person_r0/gold_frames.json" --out "$O/$M" --base-preset $BASE --size $s)
done
echo "== EVAL DONE $M"
