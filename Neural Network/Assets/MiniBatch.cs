[System.Serializable]
public class MiniBatch
{
    public double[] input;
    public double[] output;
    public MiniBatch(double[] input, double[] output)
    {
        this.input = input;
        this.output = output;
    }
}
