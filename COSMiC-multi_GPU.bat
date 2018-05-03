;comment out extra GPUs if you have fewer or add if you have more.
;make sure GPU # DEVICES=# are right. GPU# is for the terminal name.
echo 25 0 > 0xbtc.conf 
start "GPU 0" cmd /c "set CUDA_VISIBLE_DEVICES=0 && for /l %%n in () do echo pool mine cuda | index-win.exe pause"
echo 24 0 > 0xbtc.conf 
start "GPU 1" cmd /c "set CUDA_VISIBLE_DEVICES=1 && for /l %%n in () do echo pool mine cuda | index-win.exe pause"