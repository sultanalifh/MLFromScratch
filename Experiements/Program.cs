using System;

class ML {
    

    static void Main()
    {
        NeuralNetwork model = new NeuralNetwork(5,2);

        List<DataSet> dataSets = new List<DataSet>();

        Vector data1 = new Vector(2);
        data1[0] = data1[1] = 0;
        DataSet ds1 = new DataSet(data1, 0);

        Vector data2 = new Vector(2);
        data2[0] = 0; data2[1] = 1;
        DataSet ds2 = new DataSet(data2, 1);

        Vector data3 = new Vector(2);
        data3[0] = 1; data3[1] = 0;
        DataSet ds3 = new DataSet(data3, 1);

        Vector data4 = new Vector(2);
        data4[0] = 1; data4[1] = 1;
        DataSet ds4 = new DataSet(data4, 0);

        dataSets.Add(ds1);
        dataSets.Add(ds2);
        dataSets.Add(ds3);
        dataSets.Add(ds4);

        int BatchSize = 2;
        int epochs = 100000;

        for(int epoch = 0; epoch < epochs; epoch++)
        {
            dataSets = dataSets.OrderBy(_ => Utility.Random.Next()).ToList();

            for(int i = 0; i < dataSets.Count; i += BatchSize)
            {
                List<DataSet> Batch = dataSets.Skip(i).Take(BatchSize).ToList();
                model.TrainBatchSigmoid(Batch);
            }
        }

        model.ShowDebug = true;

        model.TrainBatchSigmoid(dataSets);
    }
}