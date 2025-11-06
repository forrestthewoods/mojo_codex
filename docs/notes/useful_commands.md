# run commands
pixi run python -m src.benchmarks.render_bench compare --duration 0.5 --warmup-frames 0 --target-fps 30 --width 256 --height 256

# commit and push git
git add . && git commit -m "progress" && git push

# connect to droplet
ssh root@161.35.226.156

# copy data from digital ocean to local
scp -r root@161.35.226.156:~/mojo_codex/outputs/ c:/temp/remote_mujo
