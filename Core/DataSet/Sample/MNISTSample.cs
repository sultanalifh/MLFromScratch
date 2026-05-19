class MNISTSample : Sample
{
    public double[] Image;

    public int Label;

    public override double[] Input => Image;

    public override double[] Target
    {
        get
        {
            double[] oneHot = new double[10];
            oneHot[Label] = 1;
            return oneHot;
        }
    }

    public MNISTSample(double[] image, int label)
    {
        Image = image;
        Label = label;
    }
}