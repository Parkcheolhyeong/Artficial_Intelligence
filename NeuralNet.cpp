#pragma warning(disable: 4996) 
#define _CRT_SECURE_NO_DEPRECATE


#include "StdAfx.h"
#include "NeuralNet.h"

#include <ostream>
#include <fstream>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // This contains utilities for modifying  c-style string
#include <cstring> // This is the C++ version of the library above.  It effectively just puts everything in std::
#include <string>// this contains the std::string class. 

using namespace std;

#define _CRT_SECURE_NO_WARNINGS
#define D_MAX_ARRAY_SIZE 12000
#define MAX_SIZE 20000

int Neuron::findWeightNum(const int &layer, const int &before, const int &after)
{
	int sum = 0;

	for (int i = 0; i < layer; i++)
	{
		sum += numLayerNeuron[i] * (numLayerNeuron[i + 1] - 1);
	}

	return sum + before * (numLayerNeuron[layer + 1] - 1) + after;
}

int Neuron::findNeuronNum(const int &i)
{
	int sum = 0;

	for (int x = 0; x < i; x++)
	{
		sum += numLayerNeuron[x];
	}

	return sum;
}

void Neuron::init()
{
	numInput = 784;
	numOutput = 10;
	numLayers = 3; //num of hidden layers : 1

	//Enter each hidden layers num of nueron
	numLayerNeuron = new int[numLayers];
	numLayerNeuron[0] = numInput + 1; //input layer
	numLayerNeuron[1] = 100 + 1; //hidden layer
	numLayerNeuron[2] = numOutput + 1; //output layer

	//find total num about weight,nueron
	numWeight = findWeightNum(numLayers - 1, 0, 0);
	numNeuron = findNeuronNum(numLayers);
	
	//Dynamic allocation
	input = new NeuronD[numInput];
	output = new NeuronD[numOutput];
	weight = new NeuronD[numWeight];
	neuronGrad = new NeuronD[numNeuron];
	neuronAct = new NeuronD[numNeuron];

	//alpha, bias 
	alpha = 0.3;
	bias = 1.0;

	//weight init
	for (int i = 0; i < numWeight; i++)
		weight[i] = (NeuronD)rand() / RAND_MAX * 0.1;

	//bias init
	for (int i = 0; i < numLayers; i++)
	{
		int n_idx = findNeuronNum(i) + numLayerNeuron[i] - 1;
		neuronAct[n_idx] = bias;
	}

}

void Neuron::fin()
{
	delete[]numLayerNeuron;
	delete[]input;
	delete[]output;
	delete[]weight;
	delete[]neuronGrad;
	delete[]neuronAct;
}

void Neuron::propForward(NeuronD const input[])
{
	//insert input
	for (int i = 0; i < numInput; i++)
		neuronAct[i] = input[i];
	
	//feed forward
	for (int h = 0; h < numLayers - 1; h++)
	{
		for (int i = 0; i < numLayerNeuron[h + 1] - 1; i++)
		{
			NeuronD sum = 0;

			for (int j = 0; j < numLayerNeuron[h]; j++)
			{
				int w_idx = findWeightNum(h, j, i);
				sum += weight[w_idx] * neuronAct[findNeuronNum(h) + j];
			}

			neuronAct[findNeuronNum(h + 1) + i] = getAct(sum);
		}
	}

	//extract output
	for (int i = 0; i < numOutput; i++)
		output[i] = neuronAct[findNeuronNum(numLayers - 1) + i];
}

void Neuron::print()
{
	double tmpMax = output[0];
	int index = 0;
	printf("Output : ");

	for (int i = 0; i < numOutput; i++)
	{
		if (tmpMax < output[i])
		{
			tmpMax = output[i];
			index = i;
		}
	}

	printf("%d숫자입니다.(%.2lf%%)\n", index, tmpMax * 100);
	printf("(");
	for (int i = 0; i < numOutput; i++)
	{
		printf("[%d]: %.2lf ", i, output[i]);
	}
	printf(")\n");



	printf("\n");

	CString str;
	str.Format(_T("숫자는 %d"), index);
	AfxMessageBox(str);
}

void Neuron::propBackward(NeuronD const target[])
{
	//get gradient
	for (int i = 0; i < numOutput; i++) //output gradient
	{
		neuronGrad[findNeuronNum(numLayers - 1) + i] = (output[i] - target[i]) * getActGrad(output[i]);
	}
	
	for (int h = numLayers - 2; h >= 1; h--) //hidden layers gradients
	{
		for (int i = 0; i < numLayerNeuron[h] - 1; i++)
		{
			NeuronD sum = 0;

			for (int j = 0; j < numLayerNeuron[h + 1] - 1; j++)
			{
				int w_idx = findWeightNum(h, i, j);

				sum += weight[w_idx] * neuronGrad[findNeuronNum(h + 1) + j];
			}

			neuronGrad[findNeuronNum(h) + i] = sum * getActGrad(neuronAct[findNeuronNum(h) + i]);
		}
	}
	
	//start weight update
	for (int h = numLayers - 2; h >= 0; h--)
	{
		for (int i = 0; i < numLayerNeuron[h]; i++)
		{
			for (int j = 0; j < numLayerNeuron[h + 1] - 1; j++)
			{
				int w_idx = findWeightNum(h, i, j);

				weight[w_idx] -= alpha * neuronGrad[findNeuronNum(h + 1) + j] * neuronAct[findNeuronNum(h) + i];
			}
		}
	}

	
}

void Neuron::saveWeight()
{
	char filename[20] = "weight.txt";
	ofstream fout;
	fout.open(filename);

	for (int h = numLayers - 2; h >= 0; h--)
	{
		for (int i = 0; i < numLayerNeuron[h]; i++)
		{
			for (int j = 0; j < numLayerNeuron[h + 1] - 1; j++)
			{
				int w_idx = findWeightNum(h, i, j);

				fout << weight[w_idx] << "\t";
			}
		}
	}

	fout.close();
}

void Neuron::loadWeight()
{
	FILE *pFile = NULL;

	pFile = fopen("weight.txt", "r");
	if (pFile == NULL)
	{
		//에러 처리
	}
	else
	{
		int nCount;
		string fRatio;
		char strDesc[255];

		while (!feof(pFile))
		{
			for (int h = numLayers - 2; h >= 0; h--)
			{
				for (int i = 0; i < numLayerNeuron[h]; i++)
				{
					for (int j = 0; j < numLayerNeuron[h + 1] - 1; j++)
					{
						int w_idx = findWeightNum(h, i, j);
						fscanf(pFile, "%s\t\n", strDesc);
						weight[w_idx] = stod(strDesc);
						//cout << weight_[w_idx];
					}
				}
			}
			//탭으로 분리된 파일 읽기

			//printf("탭으로 분리 : %s\t\n", strDesc);

		}

		fclose(pFile);
	}
}


NeuronD Neuron::getSigmoid(const NeuronD &x)
{
	return 1.0 / (1.0 + exp(-x));
}

NeuronD Neuron::getAct(const NeuronD &x)
{
	return getSigmoid(x);
}

NeuronD Neuron::getSigmoidGrad(const NeuronD &x)
{
	NeuronD y = getSigmoid(x);

	return (1.0 - y) * y;
}

NeuronD Neuron::getActGrad(const NeuronD &x)
{
	return getSigmoidGrad(x);
}