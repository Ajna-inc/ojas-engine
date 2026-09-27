# RT-DETRv2 training recipes (official PyTorch trainer)

Configs here are copied into `…/dev-cache/train/RT-DETR/rtdetrv2_pytorch/configs/uvh/` and run
by the pipeline scripts, which are **resumable**: rerun the same script after a crash, reboot or
kill and it continues — finished evaluations are recognised by their logs, finished arms by a
`DONE` marker, and an interrupted arm resumes from its `last.pth` (the trainer checkpoints
model, EMA, optimizer, both schedulers, the AMP scaler and the epoch every epoch).

Restart after a reboot:

    cd ojas-engine/training/rtdetrv2 && nohup ./step1_pipeline.sh >> "/data/dev-cache/train/out/step1_pipeline.log" 2>&1 &

Outputs: `…/dev-cache/train/out/step1_recipe_s_{st,mv}/` (checkpoints), `…/out/step1_logs/`
(train and eval logs), `…/out/step1_pipeline.log` (one `RESULT` line per arm × checkpoint × split).
