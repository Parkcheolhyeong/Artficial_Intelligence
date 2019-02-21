#pragma once
#include <iostream>

typedef double NeuronD;

class Neuron
{
public:
	int numInput;
	int numOutput;
	int numLayers;
	int numWeight;
	int numNeuron;
	int *numLayerNeuron;

	NeuronD *input;
	NeuronD *output;
	NeuronD *weight;
	NeuronD *neuronGrad;
	NeuronD *neuronAct;

	NeuronD alpha;
	NeuronD bias;

	int findWeightNum(const int &i, const int &j, const int &k);
	int findNeuronNum(const int &i);

	NeuronD getSigmoid(const NeuronD &x);

	NeuronD getAct(const NeuronD &x);

	NeuronD getSigmoidGrad(const NeuronD &y);

	NeuronD getActGrad(const NeuronD &y);

	void init();
	void propForward(const NeuronD input_[]);
	void propBackward(const NeuronD target[]);
	void saveWeight();
	void loadWeight();
	void print();
	void fin();
};