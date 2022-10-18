#FSHJAR002 MAKEFILE ML ASS 3

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
	
venv:
	test -d venv || python3 -m venv venv
	
clean:
	rm -rf venv
	find -iname "*.pyc" -delete

runSkip_recent:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/backup_final_model" -s

runSkip_recent_sim:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/backup_simple_model" -s


runSkipSim:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/simple_model" -s -p oversample normalize
	

runSkip:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -s

runRNN:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -ls --m "RNN" -e 10 -p oversample normalize


runTrain:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -b 64 -e 10 -ls

runExperiment:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -x

runCNN:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 10 -m "1D_CNN" -ls -p oversample

runLSTM_deep:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -b 32 -e 5 -m "LSTM_deep" -n 0.3 -ls
runLSTM_deep_skip:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/LSTM3_HighDrop" -s -b 128 -m "LSTM_deep3"


runLSTM:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 5 -m "LSTM"

runBiLSTM:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 100 -m "BiLSTM"

runBiLSTM_pool:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/biLSTM_pool" -ls -e 50 -m "BiLSTM_pool" -p oversample normalize -ls


runLSTMSkip:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" --model_choice "LSTM"

runGAN:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/GAN"

runResume:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/final_model" -r

runResume_sim:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -r
	
runShai:
	venv/bin/python3 src/wrapper.py 'data/' "./trained_models/final_model" -m 1D_HYBRID --n 1 -e 1 -b 32 -n 1 -ts 0.3 -ls

runHelp:
	venv/bin/python3 src/wrapper.py -h