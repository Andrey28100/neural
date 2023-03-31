using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Network
{
    public MiniBatch[] miniBatch = new MiniBatch[] { };
    public int[] sizes;
    public int num_layers;
    public double[][] biases;
    public double[][][] weights;
    public Network(int[] sizes)
    {
        this.sizes = sizes;
        num_layers = sizes.Length;
        weights = RandomWeights();
        biases = RandomBiases();
    }
    double[][] RandomBiases()
    {
        double[][] array = new double[sizes.Length - 1][];
        for (int i = 0; i < array.Length; i++)
            array[i] = new double[sizes[i + 1]];
        for (int i = 0; i < array.Length; i++)
            for (int j = 0; j < array[i].Length; j++)
                array[i][j] = Random.Range(-1f, 1f);
                //array[i][j] = i+j;
        return array;
    }
    double[][][] RandomWeights()
    {
        double[][][] array = new double[num_layers - 1][][];
        for (int i = 0; i < array.Length; i++)
            array[i] = new double[sizes[i + 1]][];
        for (int i = 0; i < array.Length; i++)
            for (int j = 0; j < array[i].Length; j++)
                array[i][j] = new double[sizes[i]];
        for (int i = 0; i < array.Length; i++)
            for (int j = 0; j < array[i].Length; j++)
                for (int k = 0; k < array[i][j].Length; k++)
                    array[i][j][k] = Random.Range(-1f, 1f);
                    //array[i][j][k] = i+j+k;
        return array;
    }
    double[][][] FeedForward(double[] input)
    {
        double[][] a = new double[num_layers][];
        double[][] z = new double[num_layers - 1][];
        for (int i = 0; i < num_layers; i++)
            a[i] = new double[sizes[i]];
        for (int i = 1; i < num_layers; i++)
            z[i - 1] = new double[sizes[i]];
        a[0] = input;
        for (int i = 1; i < a.Length; i++)  
        {
            for (int j = 0; j < a[i].Length; j++)
            {
                for (int k = 0; k < a[i - 1].Length; k++)
                    a[i][j] += a[i - 1][k] * weights[i - 1][j][k];
                z[i - 1][j] = a[i][j] + biases[i - 1][j];
                a[i][j] = Sigmoid(z[i - 1][j]);
            }
        }
        return new double[][][] { a, z };
    }
    double[][][][] BackPropagation(double[] input, double[] y)
    {
        double[][][] AandZ = FeedForward(input);
        double[][] a = AandZ[0];
        double[][] z = AandZ[1];
        double[][] delta = new double[num_layers - 1][];

        for (int i = 1; i < num_layers; i++)
            delta[i - 1] = new double[sizes[i]];

        double[][][] nablaW = new double[num_layers - 1][][];

        for (int i = 0; i < nablaW.Length; i++)
            nablaW[i] = new double[sizes[i + 1]][];
        for (int i = 0; i < nablaW.Length; i++)
            for (int j = 0; j < nablaW[i].Length; j++)
                nablaW[i][j] = new double[sizes[i]];

        double[][] nablaB = new double[num_layers - 1][];
        for (int i = 0; i < nablaB.Length; i++)
            nablaB[i] = new double[sizes[i + 1]];

        // цикл по нейронам выходного слоя
        for (int i = 0; i < sizes[num_layers - 1]; i++)
        {
            delta[num_layers - 2][i] = (a[num_layers - 1][i] - y[i]) * SigmoidPrime(z[num_layers - 2][i]);
            nablaB[num_layers - 2][i] = delta[num_layers - 2][i];
            for (int j = 0; j < sizes[num_layers - 2]; j++)
                nablaW[num_layers - 2][i][j] = delta[num_layers - 2][i] * a[num_layers - 2][j];
        }

        // цикл по слоям (от предпоследнего до первого)
        for (int i = num_layers - 2; i > 0; i--)
        {
            // цикл по нейроном слоя i
            for (int j = 0; j < sizes[i]; j++)
            {
                // цикл по нейроном слоя i+1
                for (int k = 0; k < sizes[i+1]; k++)
                    delta[i - 1][j] += weights[i][k][j] * delta[i][k];
                delta[i - 1][j] *= SigmoidPrime(z[i - 1][j]);
            }
            nablaB[i - 1] = delta[i - 1];
            // цикл по нейроном слоя i
            for (int j = 0; j < sizes[i]; j++)
                // цикл по нейроном слоя i+1
                for (int k = 0; k < sizes[i-1]; k++)
                    nablaW[i-1][j][k] = delta[i-1][j] * a[i-1][k];
        }
        return new double[][][][] { nablaW, new double[][][] { nablaB } };
    }
    void UpdateMiniBatch(MiniBatch[] miniBatch, double eta)
    {
        double[][][] nablaW = new double[num_layers - 1][][];
        for (int i = 0; i < nablaW.Length; i++)
            nablaW[i] = new double[sizes[i + 1]][];
        for (int i = 0; i < nablaW.Length; i++)
            for (int j = 0; j < nablaW[i].Length; j++)
                nablaW[i][j] = new double[sizes[i]];
            
        double[][] nablaB = new double[num_layers - 1][];
        for (int i = 0; i < nablaB.Length; i++)
            nablaB[i] = new double[sizes[i + 1]];

        foreach (var test in miniBatch)
        {
            double[][][][] WandB = BackPropagation(test.input, test.output);
            double[][][] nablaWW = WandB[0];
            double[][] nablaBB = WandB[1][0];
            for (int i = 0; i < nablaB.Length; i++)
                for (int j = 0; j < nablaB[i].Length; j++)
                    nablaB[i][j] += nablaBB[i][j];
            for (int i = 0; i < nablaW.Length; i++)
                for (int j = 0; j < nablaW[i].Length; j++)
                    for (int k = 0; k < nablaW[i][j].Length; k++)
                        nablaW[i][j][k] += nablaWW[i][j][k];
        }
        for (int i = 0; i < nablaB.Length; i++)
            for (int j = 0; j < nablaB[i].Length; j++)
                biases[i][j] -= eta / miniBatch.Length * nablaB[i][j];
        for (int i = 0; i < nablaW.Length; i++)
                for (int j = 0; j < nablaW[i].Length; j++)
                    for (int k = 0; k < nablaW[i][j].Length; k++)
                        weights[i][j][k] -= eta / miniBatch.Length * nablaW[i][j][k];
    }
    public string[] SGD(MiniBatch[] training_data, int epochs, int miniBatchSize, double eta, MiniBatch[] test_data)
    {
        int n_test = test_data.Length;
        int n = training_data.Length;
        List<string> results = new List<string>();
        for (int i = 0; i < epochs; i++)
        {
            training_data = ShuffleData(training_data);
            MiniBatch[] miniBatch = new MiniBatch[miniBatchSize];
            for (int k = 0; k < n; k += miniBatchSize)
            {
                for (int p = k; p < k + miniBatchSize; p++)
                    miniBatch[p - k] = training_data[p];
                UpdateMiniBatch(miniBatch, eta);
            }
            results.Add($"Epoch {i + 1}: {Evaluate(test_data)} / {n_test}");
        }
        return results.ToArray();
    }
    public int Evaluate(MiniBatch[] test_data)
    {
        double[] a;
        int rightAnswers = 0;
        foreach (var data in test_data)
        {
            a = FeedForward(data.input)[0][num_layers - 1];
            int max = a.ToList().IndexOf(a.Max());
            rightAnswers += data.output[max] == 0 ? 0 : 1;
        }
        return rightAnswers;
    }
    public MiniBatch[] ShuffleData(MiniBatch[] array)
    {
        int n = array.Length;
        while (n > 1) 
        {
            int k = Random.Range(0, n--);
            MiniBatch temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }
        return array;
    }
    double Sigmoid(double z)
    {
        return 1 / (1 + Mathf.Exp((float)-z));
    }
    double SigmoidPrime(double z)
    {
        return Sigmoid(z) * (1 - Sigmoid(z));
    }
}
