#cmd='../../msk144gensim/_bld/msk144gensim --center-freq=1520 --signal-level=575 --noise-level=0 --use-throttle=1 --on-frames=1 --off-frames=30 '

cmd='../../msk144gensim/_bld/msk144gensim --signal-level=20 --noise-level=0 --use-throttle=1 --on-frames=2 --off-frames=30 --mode=2 --num-messages=2'
#cmd2="cat aa"
#cmd3="cat aa4test.raw "
${cmd} | ./msk144cudecoder --search-width=100  --read-mode=2
#${cmd} | ../../msk144decoder/_bld/msk144decoder 
#${cmd} > aa


#--signal-level=400 --noise-level=1000