---
date: 2019-11-15 15:27:27+11:00
draft: true
title: Remote Jupyter Console
---

On the server: jupyter console --hb 47181 --shell 47182 --iopub 47183 --stdin 47184 -f ~/.jupyter/connection.json "$@"

On the client:
ssh -NfL 47181:localhost:47181 -L 47182:localhost:47182 -L 47183:localhost:47183 -L 47184:localhost:47184 $SERVER

scp ${SERVER}:.jupyter/connection.json ~/.jupyter/connection.json
jupyter console --existing ~/.jupyter/connection.json

Emacs:
(setq python-shell-interpreter "jupyter")
(setq python-shell-interpreter-args "console --simple-prompt --existing /home/eross/.jupyter/connection.json")

Drawbacks:
Can't kind kill signal