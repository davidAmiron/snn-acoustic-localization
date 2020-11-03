CC=gcc

models: model_IHC_BEZ2018.o model_Synapse_BEZ2018.o complex.o
	$(CC) -shared model_IHC_BEZ2018.o model_Synapse_BEZ2018.o complex.o -o libcochlea.so

model_IHC_BEZ2018.o: model_IHC_BEZ2018.c
	$(CC) -c -fPIC model_IHC_BEZ2018.c -o model_IHC_BEZ2018.o

model_Synapse_BEZ2018.o: model_Synapse_BEZ2018.c
	$(CC) -c -fPIC model_Synapse_BEZ2018.c -o model_Synapse_BEZ2018.o

complex.o: complex.c complex.h
	$(CC) -c -fPIC complex.c -o complex.o

clean:
	rm complex.o model_IHC_BEZ2018.o libcochlea.so
