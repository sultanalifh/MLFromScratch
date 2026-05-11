class SpiralDataset
{
    public int NumClass, NumSamplePerClass;
    public Matrix X;

    public Matrix Y;

    public SpiralDataset(int numClass, int numSamplePerClass)
    {
        NumClass = numClass;
        NumSamplePerClass = numSamplePerClass;
        int total = NumClass * NumSamplePerClass;

        X = new Matrix(total, 2);
        Y = new Matrix(total, numClass);
    }

    public void Generate()
    {
        for(int i = 0; i < NumClass; i++)
        {
            int label = i;
            double baseAngle = i * 2 * Math.PI / NumClass;

            for(int j = 0; j < NumSamplePerClass; j++)
            {
                double t = Utility.Random.NextDouble();
                double r = t;

                double theta = baseAngle + t * 4 * Math.PI;

                theta += Utility.RandomNormal() * 0.2;

                double x = r * Math.Cos(theta);
                double y = r * Math.Sin(theta);

                int idx = i * NumSamplePerClass + j;

                X[idx,0] = x;
                X[idx,1] = y;

                Y[idx,label] = 1;
            }
        }
    }
}