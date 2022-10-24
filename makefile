#FSHJAR002, ARNSHA011, RSNJOS005 MAKEFILE AI ASS 2

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv
	
clean:
	rm -rf venv
	find -iname "*.pyc" -delete

runSimple:
	venv/bin/python3 src/wrapper.py "./trained_models/simple_model" -m "simple" -e 20

runLSTM:
	venv/bin/python3 src/wrapper.py "./trained_models/lstm_model" -m "LSTM" -e 20

runCNN:
	venv/bin/python3 src/wrapper.py "./trained_models/lstm_model" -m "LSTM" -e 20

runLSTMLoad:
	venv/bin/python3 src/wrapper.py "./trained_models/lstm_model" -m "LSTM" -ls -e 20

runHelp:
	venv/bin/python3 src/wrapper.py -h
