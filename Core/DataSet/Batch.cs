class Batch<TSample> where TSample : Sample
{
    public int Size;
    public Tensor X;

    public Tensor Y;

    public Batch(List<TSample> samples, int start, int size)
    {
        Size = size;

        int[] inputShape = samples[0].Input.Shape;
        int inputShapeSize = inputShape.Length;

        int[] targetShape = samples[0].Target.Shape;
        int targetShapeSize = targetShape.Length;

        int[] batchXShape = new int[inputShapeSize + 1];
        int[] batchYShape = new int[targetShapeSize + 1];

        batchXShape[0] = batchYShape[0] = size;

        Array.Copy(inputShape, 0, batchXShape, 1, inputShapeSize);
        Array.Copy(targetShape, 0, batchYShape, 1, targetShapeSize);

        X = new Tensor(batchXShape);
        Y = new Tensor(batchYShape);

        int inputSize = samples[0].Input.Data.Length;
        int targetSize = samples[0].Target.Data.Length;

        for(int i = start; i < start + size; i++)
        {
            int dx = i * inputSize;
            int dy = i * targetSize;

            Tensor input = samples[i].Input;
            Tensor target = samples[i].Target;

            for(int j = 0; j < inputSize; j++)
            {
                int x_pos = dx + j;

                X.Data[x_pos] = input.Data[j];
            }

            for(int j = 0; j < targetSize; j++)
            {
                int y_pos = dy + j;

                Y.Data[y_pos] = target.Data[j];
            }
        }
    }
}