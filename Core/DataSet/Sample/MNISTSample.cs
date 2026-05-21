class MNISTSample : Sample
{
    public Tensor Image;

    public int Label;

    public override Tensor Input => Image;

    public override Tensor Target
    {
        get
        {
            Tensor oneHot = new Tensor(10);
            oneHot[Label] = 1;
            return oneHot;
        }
    }

    public MNISTSample(Tensor image, int label, int width = 0)
    {
        if(width != 0)
        {
            image = image.Reshape(1, width, width);
        }
        
        Image = image;
        Label = label;
    }
}