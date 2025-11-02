# commit and push git
git add . && git commit -m "progress" && git push

# copy data from digital ocean to local
scp -r root@161.35.226.156:~/mojo_codex/outputs/ c:/temp/remote_mujo
