using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class StartUp : MonoBehaviour
{
    public Network net;
    public List<MiniBatch> data = new List<MiniBatch>();
    public List<MiniBatch> training_data = new List<MiniBatch>();
    public List<MiniBatch> test_data = new List<MiniBatch>();
    public Text resultsText;
    public Text weightsText;
    public Text biasesText;
    public Text errorText;
    public int[] sizes;
    public int trainingDataSize;
    public int miniBatchSize;
    public int epochs = 3;
    public double eta = 0.1d;
    public string path = "Assets/tests.txt";
    void Start()
    {
        
    }
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
            StartNetwork();
    }
    void StartNetwork()
    {
        net = new Network(sizes);
        data = new List<MiniBatch>();
        training_data = new List<MiniBatch>();
        test_data = new List<MiniBatch>();
        resultsText.text = "";
        weightsText.text = "Weights:\n";
        biasesText.text = "Biases:\n";
        errorText.text = "";
        try
        {
            ReadData(path);
            for (int i = 0; i < trainingDataSize; i++)
                training_data.Add(data[i]);
            for (int i = trainingDataSize; i < data.Count; i++)
                test_data.Add(data[i]);
            
            string[] results = net.SGD(training_data.ToArray(), epochs, miniBatchSize, eta, test_data.ToArray());
            foreach (string i in results)
                resultsText.text += i + "\n";
            string s = "";
            int layer = 0;
            foreach (double[][] i in net.weights)
            {
                s = "weights[" + layer.ToString() + "]: ";
                foreach (double[] j in i)
                {
                    foreach (double k in j)
                        s += System.Math.Round((decimal)k, 2) + ", ";
                    s += '|';
                }
                layer++;
                weightsText.text += s;
            }
            s = "";
            layer = 0;
            foreach (double[] i in net.biases)
            {
                s = "biases[" + layer.ToString() + "]: ";
                foreach (double j in i)
                    s += System.Math.Round((decimal)j, 2) + ", ";
                layer++;
                biasesText.text += s;
            }
        }
        catch (System.Exception e)
        {
            errorText.text = e.ToString();
        }
    }
    void ReadData(string filePath)
    {
        string[] text = System.IO.File.ReadAllLines(filePath);
        List<double> resList = new List<double>();
        double res = 0;
        string[] preAns;
        double[] ans = new double[3];
        for (int i = 0; i < text.Length; i += 5)
        {
            resList.Clear();
            for (int line = i; line < i + 4; line++)
            {
                string[] rgb = text[line].Split(' ');
                int r = int.Parse(rgb[0]);
                int g = int.Parse(rgb[1]);
                int b = int.Parse(rgb[2]);
                res = (r + g + b) / 765d;
                resList.Add(res);
            }
            preAns = text[i + 4].Split(' ');
            ans = new double[3] {int.Parse(preAns[0]), int.Parse(preAns[1]), int.Parse(preAns[2]) };
            data.Add(new MiniBatch(resList.ToArray(), ans));
        }
    }
    public void OnEtaChanges(Dropdown dropdown)
    {
        eta = double.Parse(dropdown.options[dropdown.value].text.Replace(".", ","));
    }
    public void OnMiniBatchSizeChanges(InputField field)
    {
        int.TryParse(field.text, out miniBatchSize);
    }
    public void OnTrainingDataSizeChanges(InputField field)
    {
        int.TryParse(field.text, out trainingDataSize);
    }
    public void OnPathChanges(InputField field)
    {
        path = field.text;
    }
    public void OnEpochesChanges(InputField field)
    {
        int.TryParse(field.text, out epochs);
    }
}
