using System;

class ML {
    

    static void Main()
    {
        Sequential sequential = new Sequential();

        DenseLayer denseLayer = new DenseLayer(2,7);

        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                double u1 = Utility.Random.NextDouble();
                double u2 = Utility.Random.NextDouble();
                double z = Math.Sqrt(-2*Math.Log(u1)) * Math.Cos(2*Math.PI*u2);
                denseLayer.Weights.Data[i,j] = z;
            }
        }

        sequential.Add(denseLayer);

        LayerNorm batchNormLayer = new LayerNorm(7,7);

        sequential.Add(batchNormLayer);

        ReLULayer reLULayer = new ReLULayer(7,7);
    
        sequential.Add(reLULayer);

        denseLayer = new DenseLayer(7,1);

        sequential.Add(denseLayer);

        SigmoidLayer sigmoid = new SigmoidLayer(1,1);

        sequential.Add(sigmoid);

        Matrix Testcase = new Matrix(4,2);
        Testcase[1,0] = 1;
        Testcase[2,1] = 1;
        Testcase[3,0] = Testcase[3,1] = 1;

        Matrix Ans = new Matrix(4,1);
        Ans[1,0] = Ans[2,0] = 1;

        for(int i = 0; i < 10000; i++)
        {
            sequential.Backward(sequential.Forward(Testcase), Ans);
            sequential.LearnGradient(0.01);
        }

        Matrix Pred = sequential.Predict(Testcase);

        for(int i = 0; i < 4; i++)
        {
            Console.WriteLine(Pred[i,0]);
        }
    }
}